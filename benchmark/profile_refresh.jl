using StaticAssembler
include(joinpath(@__DIR__, "..", "oracle", "StaticAssemblerOracle.jl"))
using .StaticAssemblerOracle
using GridapEmbedded

function timed(f, label)
    GC.gc()
    result = @timed f()
    println(label, ",", result.time * 1e3, ",", result.bytes)
    return result.value
end

function main(n=128, order=2)
    base = build_cut_reference(n, n; radius=0.42, center=(0.50, 0.50))
    moved = build_cut_reference(n, n; radius=0.39, center=(0.53, 0.48), model=base.model)
    roots = aggregate(AggregateAllCutCells(), base.cut_geometry)
    moved_roots = aggregate(AggregateAllCutCells(), moved.cut_geometry)
    element = order == 1 ? StaticAssembler.Q1ReferenceQuadrilateral() :
                           StaticAssembler.Q2ReferenceQuadrilateral()
    form = kelvin_voigt_form(base.parameters; element)
    plan = StaticSystem(base.model, base.cut_geometry, roots, n, n; form)
    ext = Base.get_extension(StaticAssembler, :StaticAssemblerGridapExt)
    cutdata = timed("gridap_conversion") do
        ext._cut_geometry_data(moved.cut_geometry, Float64)
    end
    background_cells = Base.OneTo(n^2)
    r, a = plan.refresh, plan.assemble
    timed("quadrature_stage_commit") do
        npoints = StaticAssembler._stage_cut_quadrature!(
            r.quadrature_buffer, background_cells, cutdata, plan.nx, plan.ny,
            plan.hx, plan.hy, plan.cut_quadrature)
        StaticAssembler._commit_quadrature!(r.quadrature,
            r.quadrature_buffer.pointers, r.quadrature_buffer.is_cut,
            r.quadrature_buffer.reference_r, r.quadrature_buffer.reference_s,
            r.quadrature_buffer.weight, npoints)
    end
    tables = timed("static_tables") do
        StaticAssembler._refresh_static_tables!(Float64, a.test_cell_dofs,
            a.test_constraints, r.element_tables.test_space, cutdata.bg_states,
            moved_roots, n, n)
    end
    E, _ = tables
    B = timed("sparse_pattern") do
        StaticAssembler._pattern_from_ragged_dofs!(a.B, a.pattern_rows,
            a.test_cell_dofs, a.test_cell_dofs, size(E, 2), size(E, 2))
    end
    timed("scatter_slots") do
        StaticAssembler._ragged_scatter_slots!(a.scatter_slots, B,
            a.test_cell_dofs, a.test_cell_dofs)
    end
    timed("coloring") do
        StaticAssembler._effective_cell_colors(a.test_cell_dofs, a.test_cell_dofs)
    end
    timed("element_tables") do
        StaticAssembler._regenerate_element_tables!(r.element_tables, plan.hx,
            plan.hy, r.quadrature, n^2, plan.interior_quadrature)
    end
end

main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128,
     length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2)
