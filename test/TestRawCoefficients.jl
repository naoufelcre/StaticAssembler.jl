# Standalone test for the raw-coefficient assembly path
# (`assemble!(...; extend_coefficients=false)`) of StaticAssembler.
#
# Run with the Stack environment:
#
#   julia --project="/Users/naoufel/Documents/Scientific stack/Stack" \
#         StaticAssembler.jl/test/TestRawCoefficients.jl
#
# It checks, for Q1 and Q2 scalar coefficient/test spaces:
#   1. the default path equals extend_coefficients=true;
#   2. the raw path matches an independent Gridap RHS that evaluates the raw
#      background polynomial on cut cells, while the default differs there;
#   3. wrong background-node counts are rejected in both modes;
#   4. the keyword is honoured after a structural geometry refresh.
#
# No new dependencies; uses Gridap/GridapEmbedded already in Stack.

using StaticAssembler
if !isdefined(Main, :StaticAssemblerOracle)
    include(joinpath(@__DIR__, "..", "oracle", "StaticAssemblerOracle.jl"))
end
using .StaticAssemblerOracle
using Gridap
using GridapEmbedded
using Gridap.Geometry: get_node_coordinates, get_glue
using LinearAlgebra
using Printf
using Test

# Non-polynomial field: its Q1/Q2 nodal interpolant differs from the root
# interpolation on cut cells, so the two coefficient modes are distinguishable.
raw_field(x, y) = exp(0.7x) * sin(1.3y + 0.2)

function lattice_values(reference, order::Int)
    nx, ny = reference.nx, reference.ny
    origin = first(get_node_coordinates(reference.model))
    step = order == 1 ? 1 : 2
    values = Float64[]
    for j in 0:(step * ny), i in 0:(step * nx)
        x = origin[1] + i * reference.hx / step
        y = origin[2] + j * reference.hy / step
        push!(values, raw_field(x, y))
    end
    return values
end

function raw_rhs_form(element)
    space = StaticAssembler.FiniteElement(element, 1)
    return StaticAssembler.ElementVariationalForm(
        space, (), (StaticAssembler.RhsSlot(scalar_value_op(), c -> c[1], 1.0),), 1;
        coefficient_elements = (element,))
end

# Gridap DOF -> background lattice node, for a scalar Q1/Q2 space on the
# active triangulation (same 1:1 layout as the engine's `_cell_nodes`).
function scalar_stable_map(order::Int, cell_dofs, cell_nodes, nstandard::Int)
    nn = order == 1 ? 4 : 9
    stable = zeros(Int, nstandard)
    for cell in eachindex(cell_dofs), local_dof in 1:nn
        node = cell_nodes[local_dof, cell]
        dof = cell_dofs[cell][local_dof]
        previous = stable[dof]
        (previous == 0 || previous == node) ||
            error("inconsistent local DOF ordering for order $order")
        stable[dof] = node
    end
    return stable
end

# Independent Gridap RHS for the raw coefficient: assemble ∫ c v dΩ on the
# standard active space with the raw background polynomial `c` (no AgFEM
# interpolation), reindex to the background lattice, and project with `Eᵀ`.
# This avoids Gridap's AgFEM DOF ordering and reuses only the E row map.
function gridap_raw_rhs(reference, E, order::Int)
    model, cutgeo = reference.model, reference.cut_geometry
    nx = reference.nx
    active = Triangulation(cutgeo, ACTIVE)
    physical = Triangulation(cutgeo, PHYSICAL)
    scalar_fe = ReferenceFE(lagrangian, Float64, order)
    standard = TestFESpace(active, scalar_fe, conformity = :H1)
    background = FESpace(model, scalar_fe, conformity = :H1)
    c = interpolate(interpolate(x -> raw_field(x[1], x[2]), background), standard)
    dΩ = Measure(physical, order == 1 ? 6 : 10)
    a(u, v) = ∫(u * v)dΩ
    l(v) = ∫(c * v)dΩ
    op = AffineFEOperator(a, l, TrialFESpace(standard), standard)
    b_gridap = get_vector(op)
    active_bg_cells = collect(get_glue(active, Val(2)).tface_to_mface)
    element = order == 1 ? StaticAssembler.Q1ReferenceQuadrilateral() :
                           StaticAssembler.Q2ReferenceQuadrilateral()
    lattice = StaticAssembler._cell_nodes(element, Int32.(active_bg_cells), nx)
    stable = scalar_stable_map(order, collect(get_cell_dof_ids(standard)),
                               lattice, num_free_dofs(standard))
    b_stable = zeros(size(E, 1))
    for dof in eachindex(stable)
        stable[dof] == 0 && continue
        b_stable[stable[dof]] = b_gridap[dof]
    end
    return transpose(E) * b_stable
end

function check_order(nx::Int, ny::Int, order::Int)
    reference = build_cut_reference(nx, ny)
    roots = aggregate(AggregateAllCutCells(), reference.cut_geometry)
    system = StaticSystem(reference.model, reference.cut_geometry, roots, nx, ny;
                          form = raw_rhs_form(order == 1 ? StaticAssembler.Q1ReferenceQuadrilateral() :
                                                       StaticAssembler.Q2ReferenceQuadrilateral()))
    values = lattice_values(reference, order)

    _, b_default = assemble!(system, (values,); threaded = false)
    b_default = copy(b_default)
    _, b_true = assemble!(system, (values,); threaded = false, extend_coefficients = true)
    b_true = copy(b_true)
    _, b_raw = assemble!(system, (values,); threaded = false, extend_coefficients = false)
    b_raw = copy(b_raw)

    # 1. default unchanged.
    @test b_default == b_true

    # 2. independent Gridap RHS for the raw coefficient.
    b_oracle_raw = gridap_raw_rhs(reference, system.refresh.E, order)
    @test b_raw ≈ b_oracle_raw rtol = 1e-10
    # The raw and default paths must actually differ on this cut geometry.
    @test norm(b_raw - b_default) > 1e-8 * norm(b_default)

    # 3. wrong background-node counts rejected in both modes.
    @test_throws DimensionMismatch assemble!(system, (values[1:end-1],);
                                             threaded = false, extend_coefficients = false)
    @test_throws DimensionMismatch assemble!(system, (values[1:end-1],);
                                             threaded = false)

    # 4. geometry refresh honours the keyword: a fresh system on the new
    # cut geometry reproduces the refreshed raw path exactly.
    shifted = build_cut_reference(nx, ny; center = (0.46, 0.53), model = reference.model)
    shifted_roots = aggregate(AggregateAllCutCells(), shifted.cut_geometry)
    refresh_geometry!(system, shifted.cut_geometry, shifted_roots)
    shifted_values = lattice_values(shifted, order)
    _, b_refreshed_raw = assemble!(system, (shifted_values,);
                                   threaded = false, extend_coefficients = false)
    b_refreshed_raw = copy(b_refreshed_raw)
    _, b_refreshed_true = assemble!(system, (shifted_values,);
                                    threaded = false, extend_coefficients = true)
    b_refreshed_true = copy(b_refreshed_true)
    fresh = StaticSystem(shifted.model, shifted.cut_geometry, shifted_roots, nx, ny;
                         form = raw_rhs_form(order == 1 ? StaticAssembler.Q1ReferenceQuadrilateral() :
                                                      StaticAssembler.Q2ReferenceQuadrilateral()))
    _, b_fresh_raw = assemble!(fresh, (shifted_values,);
                               threaded = false, extend_coefficients = false)
    @test b_refreshed_raw ≈ copy(b_fresh_raw) rtol = 1e-14
    @test norm(b_refreshed_raw - b_refreshed_true) > 1e-8 * norm(b_refreshed_true)

    return (order = order, raw_vs_default = norm(b_raw - b_default) / norm(b_default))
end

@testset "raw-coefficient assembly" begin
    for order in (1, 2)
        info = check_order(4, 3, order)
        @printf("  order %d: raw/default relative difference = %.3e\n",
                info.order, info.raw_vs_default)
    end
end
