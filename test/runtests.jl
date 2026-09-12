using StaticAssembler
# Kelvin-Voigt PDE, Gridap oracle, and equivalence helpers live outside the
# library; include them as test support.
include(joinpath(@__DIR__, "..", "oracle", "StaticAssemblerOracle.jl"))
using .StaticAssemblerOracle
# White-box access to assembly internals under test.
using StaticAssembler: validate_sparse_slots
using Gridap
using GridapEmbedded
# Gridap adapter (extension); loaded since Gridap/GridapEmbedded are in scope.
const StaticAssemblerGridapExt = Base.get_extension(StaticAssembler, :StaticAssemblerGridapExt)
using Gridap.TensorValues
using Gridap.Geometry: get_node_coordinates
using LinearAlgebra
using Random
using SparseArrays
using Test

# Testset filtering: run the full suite with `Pkg.test()` or a subset with
# `Pkg.test(; test_args=["engine"])`, `["equivalence"]`, or individual testset
# names. Groups: "equivalence" (Gridap-oracle comparisons) and "engine"
# (structural invariants of the cut-quadrature refresh path).
const SELECTED = Set(ARGS)
# `do`-block syntax passes the anonymous function as the first argument.
function run_testset(body, name, group)
    if isempty(SELECTED) || name in SELECTED || group in SELECTED
        @testset "$name" begin
            body()
        end
    end
end

function _add_moments!(moments, x, y, weight)
    moments[1] += weight
    moments[2] += x * weight
    moments[3] += y * weight
    moments[4] += x^2 * weight
    moments[5] += x * y * weight
    moments[6] += y^2 * weight
end

function _direct_moments(plan, domain)
    quadrature = plan.refresh.quadrature
    states = plan.refresh.cell_states
    xmin, _, ymin, _ = domain
    moments = zeros(6)
    a = inv(sqrt(3.0))
    lo, hi = (1-a)/2, (1+a)/2
    for bg_cell in 1:(plan.nx * plan.ny)
        i = (bg_cell - 1) % plan.nx
        j = (bg_cell - 1) ÷ plan.nx
        first_point = quadrature.pointers[bg_cell]
        last_point = quadrature.pointers[bg_cell + 1] - 1
        if first_point > last_point
            states[bg_cell] == StaticAssembler.OUT && continue
            quadrature.is_cut[bg_cell] && continue
            for (r, s) in ((lo, lo), (hi, lo), (lo, hi), (hi, hi))
                x = xmin + (i + r) * plan.hx
                y = ymin + (j + s) * plan.hy
                _add_moments!(moments, x, y, plan.hx * plan.hy / 4)
            end
        else
            for q in first_point:last_point
                x = xmin + (i + quadrature.reference_r[q]) * plan.hx
                y = ymin + (j + quadrature.reference_s[q]) * plan.hy
                _add_moments!(moments, x, y, quadrature.weight[q])
            end
        end
    end
    return moments
end

function _gridap_moments(reference)
    measure = Measure(reference.domain, 2)
    functions = (
        x -> 1.0, x -> x[1], x -> x[2],
        x -> x[1]^2, x -> x[1] * x[2], x -> x[2]^2,
    )
    return [sum(integrate(CellField(f, reference.domain), measure)) for f in functions]
end

# Vector reaction-diffusion: a(u,v) = ∫ κ ∇u:∇v + c u·v, l(v) = ∫ f·v.
# Proves the assembly path is independent of the Kelvin-Voigt PDE, including
# state-dependent matrix coefficients: FieldCoefficient slots (linear in the
# fields) and StateFunction slots (nonlinear pointwise coefficient rules).
function reaction_diffusion_form()
    f = c -> VectorValue(c[3], c[4])
    space = StaticAssembler.FiniteElement(StaticAssembler.Q1ReferenceQuadrilateral(), 2)
    return StaticAssembler.ElementVariationalForm(
        space,
        (StaticAssembler.BilinearSlot(grad_op(), grad_op(),
                                 StaticAssembler.FieldCoefficient(1)),
         StaticAssembler.BilinearSlot(identity_op(), identity_op(),
                                 StaticAssembler.FieldCoefficient(2))),
        (StaticAssembler.RhsSlot(identity_op(), f, 1.0),),
        4)
end

function state_dependent_form()
    f = c -> VectorValue(c[3], c[4])
    space = StaticAssembler.FiniteElement(StaticAssembler.Q1ReferenceQuadrilateral(), 2)
    return StaticAssembler.ElementVariationalForm(
        space,
        (StaticAssembler.BilinearSlot(grad_op(), grad_op(),
                                 StaticAssembler.StateFunction(cq -> 1.0 + 0.5cq[1]^2)),
         StaticAssembler.BilinearSlot(identity_op(), identity_op(),
                                 StaticAssembler.StateFunction(cq -> 0.5 + 0.1cq[2]))),
        (StaticAssembler.RhsSlot(identity_op(), f, 1.0),),
        4)
end

struct ScaledValueOp <: StaticAssembler.AbstractDifferentialOperator
    scale::Float64
end

@inline function StaticAssembler._op_pair(test::ScaledValueOp, trial::ScaledValueOp,
                                          row::Int, col::Int,
                                          sh_t, gx_t, gy_t, sh_c, gx_c, gy_c)
    return test.scale * trial.scale * sh_t[row] * sh_c[col]
end

@inline function StaticAssembler._rhs_contract!(fe, op::ScaledValueOp, slot,
                                                shape, gx, gy, c, w, offset::Int)
    value = op.scale * slot.scale * slot.source(c) * w
    @inbounds for row in eachindex(shape)
        fe[offset + row] += value * shape[row]
    end
    return nothing
end

run_testset("differential operator instances", "engine") do
    space = StaticAssembler.FiniteElement(StaticAssembler.Q1ReferenceQuadrilateral(), 1)
    form = StaticAssembler.ElementVariationalForm(
        space,
        (StaticAssembler.BilinearSlot(
             ScaledValueOp(2.0), ScaledValueOp(3.0),
             StaticAssembler.ConstantCoefficient(5.0)),
         StaticAssembler.BilinearSlot(
             ScaledValueOp(2.0), ScaledValueOp(3.0),
             StaticAssembler.FieldCoefficient(1))),
        (StaticAssembler.RhsSlot(ScaledValueOp(4.0), c -> c[1], 3.0),),
        1)
    reference = build_cut_reference(1, 1; levelset_values=fill(-1.0, 4))
    system = StaticSystem(reference.model, reference.cut_geometry, Int32[1], 1, 1; form)

    B, b = assemble!(system, (fill(2.0, 4),); threaded=false)
    @test sum(B) ≈ 42.0
    @test sum(b) ≈ 24.0
end

run_testset("StaticAssembler", "equivalence") do
    errors = cut_equivalence_errors(8, 7; radius=0.34, center=(0.48, 0.51))
    @test errors.serial.matrix_relative < 5e-13
    @test errors.serial.vector_relative < 5e-13
    @test errors.threaded.matrix_relative < 5e-13
    @test errors.threaded.vector_relative < 5e-13

    parameters = KelvinVoigtParameters{Float64}(ξ=0.002, A=0.017, ν=0.43)
    custom_errors = cut_equivalence_errors(5, 4; parameters, radius=0.32)
    @test custom_errors.serial.matrix_relative < 5e-13
    @test custom_errors.serial.vector_relative < 5e-13
    @test custom_errors.threaded.matrix_relative < 5e-13
    @test custom_errors.threaded.vector_relative < 5e-13

    q2_errors = cut_equivalence_errors(4, 3; order=2)
    @test q2_errors.serial.matrix_relative < 5e-12
    @test q2_errors.serial.vector_relative < 5e-12
    @test q2_errors.threaded.matrix_relative < 5e-12
    @test q2_errors.threaded.vector_relative < 5e-12

    stokes_errors = cut_stokes_equivalence_errors(4, 3)
    @test stokes_errors.serial.matrix_relative < 5e-12
    @test stokes_errors.serial.vector_relative < 5e-12
    @test stokes_errors.threaded.matrix_relative < 5e-12
    @test stokes_errors.threaded.vector_relative < 5e-12

    rectangular_errors = cut_equivalence_errors(
        6, 5; radius=0.299, center=(0.451, 0.503), domain=(0.0, 0.9, 0.0, 1.0))
    @test rectangular_errors.serial.matrix_relative < 5e-13
    @test rectangular_errors.serial.vector_relative < 5e-13
    @test rectangular_errors.threaded.matrix_relative < 5e-13
    @test rectangular_errors.threaded.vector_relative < 5e-13

    zero_errors = _errors(
        spzeros(2, 2), zeros(2), spzeros(2, 2), zeros(2))
    @test zero_errors.matrix_relative == 0.0
    @test zero_errors.vector_relative == 0.0
    nonzero_errors = _errors(
        sparse([1], [1], [1.0], 2, 2), [1.0, 0.0], spzeros(2, 2), zeros(2))
    @test nonzero_errors.matrix_relative == Inf
    @test nonzero_errors.vector_relative == Inf
    nonfinite_errors = _errors(
        sparse([1], [1], [NaN], 2, 2), zeros(2), spzeros(2, 2), zeros(2))
    @test nonfinite_errors.matrix_relative == Inf
    @test nonfinite_errors.matrix_maximum == Inf

    reference = build_cut_reference(8, 7; radius=0.34, center=(0.48, 0.51))
    roots = aggregate(AggregateAllCutCells(), reference.cut_geometry)
    system = StaticSystem(reference.model, reference.cut_geometry, roots, 8, 7;
                          form=kelvin_voigt_form(KelvinVoigtParameters{Float64}()))
    points = vec(collect(get_node_coordinates(reference.model)))
    coefficients = ([density(p[1], p[2]) for p in points],
                    [strain_components(p[1], p[2])[1] for p in points],
                    [strain_components(p[1], p[2])[2] for p in points],
                    [strain_components(p[1], p[2])[3] for p in points])
    serial_matrix, serial_vector = assemble!(system, coefficients; threaded=false)
    serial_matrix = copy(serial_matrix)
    serial_vector = copy(serial_vector)
    threaded_matrix, threaded_vector = assemble!(system, coefficients; threaded=true)
    @test threaded_matrix ≈ serial_matrix rtol=1e-14 atol=1e-15
    @test threaded_vector ≈ serial_vector rtol=1e-14 atol=1e-15
    repeat_matrix = copy(threaded_matrix)
    repeat_vector = copy(threaded_vector)
    assemble!(system, coefficients; threaded=true)
    @test threaded_matrix == repeat_matrix
    @test threaded_vector == repeat_vector

    # Reaction-diffusion (FieldCoefficient) and state-dependent
    # (StateFunction) coefficient forms: the assembly path must be
    # independent of the Kelvin-Voigt PDE, including state-dependent
    # matrix coefficients.
    function check_coefficient_form(form, gridap_bilinear)
        nx, ny = 8, 7
        model = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (nx, ny))
        scalar_fe = ReferenceFE(lagrangian, Float64, 1)
        vector_fe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
        background_scalar = FESpace(model, scalar_fe, conformity=:H1)
        cutgeo = cut(model, disk(0.34, x0=Point(0.49, 0.51)))
        roots = aggregate(AggregateAllCutCells(), cutgeo)
        active = Triangulation(cutgeo, ACTIVE)
        physical = Triangulation(cutgeo, PHYSICAL)
        vector_standard = TestFESpace(active, vector_fe, conformity=:H1)
        scalar_standard = TestFESpace(active, scalar_fe, conformity=:H1)
        vector_test = AgFEMSpace(vector_standard, roots)
        vector_trial = TrialFESpace(vector_test)
        scalar_space = AgFEMSpace(scalar_standard, roots)

        coords = vec(collect(get_node_coordinates(model)))
        κ = [1.0 + 0.2x[1] - 0.1x[2] for x in coords]
        reaction = [0.5 + 0.1x[1] for x in coords]
        f1 = [sin(π * x[1]) * cos(π * x[2]) for x in coords]
        f2 = [cos(π * x[1]) * sin(π * x[2]) for x in coords]
        coefficients = (κ, reaction, f1, f2)

        κh = interpolate(FEFunction(background_scalar, κ), scalar_space)
        ch = interpolate(FEFunction(background_scalar, reaction), scalar_space)
        f1h = interpolate(FEFunction(background_scalar, f1), scalar_space)
        f2h = interpolate(FEFunction(background_scalar, f2), scalar_space)
        fh = Operation((a, b) -> VectorValue(a, b))(f1h, f2h)
        measure = Measure(physical, 6)
        grad_frob = Operation() do Ju, Jv
            Ju[1,1]*Jv[1,1] + Ju[1,2]*Jv[1,2] +
                Ju[2,1]*Jv[2,1] + Ju[2,2]*Jv[2,2]
        end
        a(u, v) = gridap_bilinear(κh, ch, grad_frob, measure)(u, v)
        l(v) = integrate(fh ⋅ v, measure)
        operator = AffineFEOperator(a, l, vector_trial, vector_test)

        projected = StaticSystem(
            model, cutgeo, roots, nx, ny; form,
            interior_quadrature=StaticAssemblerOracle._gridap_interior_quadrature(6))
        B, b_m = assemble!(projected, coefficients; threaded=false)
        active_bg_cells = collect(get_glue(active, Val(2)).tface_to_mface)
        compact_to_stable = agfem_compact_map(
            vector_standard, vector_test, active_bg_cells, nx)
        ours = stable_to_master(projected.refresh.E)
        gridap_to_ours = ours[compact_to_stable]
        @test B[gridap_to_ours, gridap_to_ours] ≈ get_matrix(operator) rtol=8e-13 atol=2e-14
        @test b_m[gridap_to_ours] ≈ get_vector(operator) rtol=8e-13 atol=2e-14
        @test B ≈ transpose(B) rtol=1e-14 atol=1e-15
        @test diag(B) ≈ diag(get_matrix(operator)[gridap_to_ours, gridap_to_ours]) rtol=1e-12 atol=1e-13
        return nothing
    end

    check_coefficient_form(reaction_diffusion_form(),
        (κh, ch, grad_frob, measure) ->
            (u, v) -> integrate(κh * grad_frob(∇(u), ∇(v)) + ch * (u ⋅ v), measure))
    check_coefficient_form(state_dependent_form(),
        (κh, ch, grad_frob, measure) ->
            (u, v) -> integrate(Operation(x -> 1.0 + 0.5x^2)(κh) * grad_frob(∇(u), ∇(v)) +
                               Operation(x -> 0.5 + 0.1x)(ch) * (u ⋅ v), measure))

    system_reference = build_cut_reference(8, 7; radius=0.34, center=(0.48, 0.51))
    system_roots = aggregate(AggregateAllCutCells(), system_reference.cut_geometry)
    system = StaticSystem(system_reference.model, system_reference.cut_geometry,
                          system_roots, 8, 7;
                          form=kelvin_voigt_form(KelvinVoigtParameters{Float64}()))
    cache = system.assemble
    @test eltype(cache.scatter_slots) == Vector{Int}
    @test eltype(system.refresh.quadrature.pointers) == Int
    @test !StaticAssembler._projection_structure_changed(
        Int8[StaticAssembler.IN, StaticAssembler.CUT, StaticAssembler.OUT],
        Int32[1, 1, 0],
        Int8[StaticAssembler.CUT, StaticAssembler.IN, StaticAssembler.OUT],
        Int32[1, 1, 99])
    @test StaticAssembler._projection_structure_changed(
        Int8[StaticAssembler.CUT], Int32[1],
        Int8[StaticAssembler.OUT], Int32[0])
    @test StaticAssembler._projection_structure_changed(
        Int8[StaticAssembler.CUT, StaticAssembler.IN], Int32[2, 2],
        Int8[StaticAssembler.CUT, StaticAssembler.IN], Int32[1, 2])
    @test validate_sparse_slots(cache.B, cache.test_cell_dofs,
                                cache.test_cell_dofs, cache.scatter_slots)
    @test all(isempty(system.assemble.test_constraints[cell])
              for cell in eachindex(system.refresh.cell_states)
              if system.refresh.cell_states[cell] == StaticAssembler.OUT)
    bad_slots = copy(cache.scatter_slots)
    bad_slots[1] = [0]
    @test_throws ArgumentError validate_sparse_slots(
        cache.B, cache.test_cell_dofs, cache.test_cell_dofs, bad_slots)
    B_pattern = cache.B
    dofs_pattern = cache.test_cell_dofs
    slots_pattern = cache.scatter_slots
    @test refresh_geometry!(system, system_reference.cut_geometry, system_roots) === system
    @test system.assemble.B === B_pattern
    @test system.assemble.test_cell_dofs === dofs_pattern
    @test system.assemble.scatter_slots === slots_pattern
    @test_throws ArgumentError build_cut_reference(0, 2)
    @test_throws ArgumentError build_cut_reference(2, 2; radius=0)
    @test_throws ArgumentError build_cut_reference(
        2, 2; parameters=KelvinVoigtParameters{Float64}(ξ=NaN, A=0.01, ν=0.48))
end

run_testset("direct cut-cell quadrature", "equivalence") do
    default_form = kelvin_voigt_form(KelvinVoigtParameters{Float64}())
    function quadrature_system(reference, nx, ny)
        # Identity roots: quadrature does not depend on aggregation, and
        # degenerate geometries (e.g. a lone cut cell with no interior
        # neighbor) have nothing to aggregate to.
        roots = Int32.(1:(nx * ny))
        return StaticSystem(reference.model, reference.cut_geometry, roots,
                            nx, ny; form=default_form)
    end

    # Defaults match the polynomial envelope of the corresponding tensor-product
    # interior rules: Q1-Q1-Q1 through degree 6 and Q2-Q2-Q1 through degree 10.
    for (element, degree, npoints) in (
        (StaticAssembler.Q1ReferenceQuadrilateral(), 6, 12),
        (StaticAssembler.Q2ReferenceQuadrilateral(), 10, 25),
    )
        points, weights = StaticAssembler.cut_rule(element)
        @test length(points) == length(weights) == npoints
        for i in 0:degree, j in 0:(degree - i)
            actual = sum(weights[q] * points[q][2]^i * points[q][3]^j
                         for q in eachindex(weights))
            exact = 2.0 * factorial(i) * factorial(j) / factorial(i + j + 2)
            @test actual ≈ exact rtol=5e-13 atol=5e-15
        end
    end

    triangle_reference = build_cut_reference(
        1, 1; levelset_values=[-1.0, 1.0, 1.0, 1.0])
    triangle_plan = quadrature_system(triangle_reference, 1, 1)
    quadrature = triangle_plan.refresh.quadrature
    @test fieldnames(typeof(quadrature)) ==
          (:pointers, :is_cut, :reference_r, :reference_s, :weight)
    @test quadrature.pointers == [1, 13]
    @test length(quadrature.weight) == 12
    @test sum(quadrature.weight) ≈ 1 / 8
    @test _direct_moments(triangle_plan, (0.0, 1.0, 0.0, 1.0)) ≈
          _gridap_moments(triangle_reference) rtol=5e-14 atol=5e-16

    scalar_space = StaticAssembler.FiniteElement(
        StaticAssembler.Q1ReferenceQuadrilateral(), 1)
    mass_form = StaticAssembler.ElementVariationalForm(
        scalar_space,
        (StaticAssembler.BilinearSlot(
            ScaledValueOp(1.0), ScaledValueOp(1.0),
            StaticAssembler.ConstantCoefficient(1.0)),),
        (), 0)
    mass_plan = StaticSystem(
        triangle_reference.model, triangle_reference.cut_geometry,
        Int32[1], 1, 1; form=mass_form)
    mass, _ = assemble!(mass_plan, (); threaded=false)
    @test mass[4, 4] ≈ 1 / 11520 rtol=5e-14

    bad_cut_rules = (
        ((), ()),
        (((1 / 3, 1 / 3, 1 / 3),), (0.5,)),
        (((0.2, 0.2, 0.2),), (1.0,)),
        (((1 / 3, 1 / 3, 1 / 3),), (-1.0,)),
    )
    for rule in bad_cut_rules
        @test_throws ArgumentError StaticSystem(
            triangle_reference.model, triangle_reference.cut_geometry,
            Int32[1], 1, 1; form=default_form, cut_quadrature=rule)
    end
    @test_throws ArgumentError StaticSystem(
        triangle_reference.model, triangle_reference.cut_geometry,
        Int32[1], 1, 1; form=default_form,
        interior_quadrature=(((1.5, 0.5),), (1.0,)))

    interior_domain = (-0.7, 1.3, 0.2, 1.4)
    interior_reference = build_cut_reference(
        4, 3; domain=interior_domain, radius=10.0, center=(0.3, 0.8))
    interior_plan = quadrature_system(interior_reference, 4, 3)
    @test isempty(interior_plan.refresh.quadrature.weight)
    @test all(==(1), interior_plan.refresh.quadrature.pointers)
    @test !any(interior_plan.refresh.quadrature.is_cut)
    @test count(==(StaticAssembler.IN), interior_plan.refresh.cell_states) == 12
    @test _direct_moments(interior_plan, interior_domain) ≈
          _gridap_moments(interior_reference) rtol=5e-14 atol=5e-14

    translated_domain = (-0.4, 1.7, 0.2, 1.1)
    translated_reference = build_cut_reference(
        7, 5; domain=translated_domain, radius=0.35, center=(0.55, 0.65))
    translated_plan = quadrature_system(translated_reference, 7, 5)
    direct_moments = _direct_moments(translated_plan, translated_domain)
    gridap_moments = _gridap_moments(translated_reference)
    @test direct_moments ≈ gridap_moments rtol=5e-13 atol=5e-14
    implicit_area = count(!, translated_plan.refresh.quadrature.is_cut) *
                    translated_plan.hx * translated_plan.hy
    @test implicit_area + sum(translated_plan.refresh.quadrature.weight) ≈ gridap_moments[1]

    node_crossing_reference = build_cut_reference(
        10, 10; radius=0.4, center=(0.5, 0.5))
    node_crossing_plan = quadrature_system(node_crossing_reference, 10, 10)
    @test _direct_moments(node_crossing_plan, (0.0, 1.0, 0.0, 1.0)) ≈
          _gridap_moments(node_crossing_reference) rtol=5e-13 atol=5e-14
end

run_testset("direct quadrature refresh transitions and reuse", "engine") do
    interior_reference = build_cut_reference(
        1, 1; levelset_values=fill(-1.0, 4))
    small_cut_reference = build_cut_reference(
        1, 1; levelset_values=[-1.0, 1.0, 1.0, 1.0],
        model=interior_reference.model)
    half_cut_reference = build_cut_reference(
        1, 1; levelset_values=[-1.0, -1.0, 1.0, 1.0],
        model=interior_reference.model)
    roots_of(ref) = Int32.(1:(ref.nx * ref.ny))
    plan = StaticSystem(interior_reference.model, interior_reference.cut_geometry,
                        roots_of(interior_reference), 1, 1;
                        form=kelvin_voigt_form(KelvinVoigtParameters{Float64}()))
    quadrature() = plan.refresh.quadrature
    staging() = plan.refresh.quadrature_buffer
    pointers = quadrature().pointers
    is_cut = quadrature().is_cut
    reference_r = quadrature().reference_r
    reference_s = quadrature().reference_s
    weights = quadrature().weight
    staging_r = staging().reference_r
    staging_s = staging().reference_s
    staging_weights = staging().weight
    staging_is_cut = staging().is_cut

    custom_interior_points = [(0.5, 0.5)]
    custom_interior_weights = [1.0]
    custom_cut_points = [(1 / 3, 1 / 3, 1 / 3)]
    custom_cut_weights = [1.0]
    custom = StaticSystem(
        interior_reference.model, interior_reference.cut_geometry,
        roots_of(interior_reference), 1, 1;
        form=kelvin_voigt_form(KelvinVoigtParameters{Float64}()),
        interior_quadrature=(points=custom_interior_points,
                             weights=custom_interior_weights),
        cut_quadrature=(points=custom_cut_points, weights=custom_cut_weights))
    @test custom.refresh.element_tables.interior_weight == [1.0]
    custom_interior_points[1] = (0.0, 0.0)
    custom_interior_weights[1] = 0.0
    custom_cut_points[1] = (1.0, 0.0, 0.0)
    custom_cut_weights[1] = 0.0
    refresh_geometry!(custom, small_cut_reference.cut_geometry,
                      roots_of(small_cut_reference))
    @test custom.refresh.quadrature.pointers == [1, 2]
    @test custom.refresh.quadrature.reference_r ≈ [1 / 6]
    @test custom.refresh.quadrature.reference_s ≈ [1 / 6]
    @test custom.refresh.quadrature.weight ≈ [1 / 8]

    refresh_geometry!(plan, small_cut_reference.cut_geometry,
                      roots_of(small_cut_reference))
    @test quadrature().pointers == [1, 13]
    @test quadrature().is_cut == [true]
    small_area = sum(quadrature().weight)
    refresh_geometry!(plan, half_cut_reference.cut_geometry,
                      roots_of(half_cut_reference))
    @test quadrature().pointers == [1, 37]
    @test sum(quadrature().weight) > small_area
    high_water = length(staging().weight)
    @test high_water == 36
    refresh_geometry!(plan, small_cut_reference.cut_geometry,
                      roots_of(small_cut_reference))
    @test length(quadrature().weight) == 12
    @test length(staging().weight) == high_water
    refresh_geometry!(plan, interior_reference.cut_geometry,
                      roots_of(interior_reference))
    @test quadrature().pointers == [1, 1]
    @test quadrature().is_cut == [false]
    @test isempty(quadrature().weight)
    @test length(staging().weight) == high_water
    half_roots = roots_of(half_cut_reference)
    refresh_geometry!(plan, half_cut_reference.cut_geometry, half_roots)
    warmed_bytes = @allocated refresh_geometry!(
        plan, half_cut_reference.cut_geometry, half_roots)
    @test warmed_bytes <= 4096
    @test quadrature().pointers === pointers
    @test quadrature().is_cut === is_cut
    @test quadrature().reference_r === reference_r
    @test quadrature().reference_s === reference_s
    @test quadrature().weight === weights
    @test staging().reference_r === staging_r
    @test staging().reference_s === staging_s
    @test staging().weight === staging_weights
    @test staging().is_cut === staging_is_cut

    # Independent Gridap-derived quadrature agrees cell by cell. Point order
    # within a cell is an implementation detail (assembly sums are
    # order-independent) and the two paths differ in the last ulp, so
    # segments compare as rounded sorted triples.
    function sorted_triples(pointers, r, s, w, bg)
        lo, hi = pointers[bg], pointers[bg + 1] - 1
        triples = [(round(r[q]; digits=12), round(s[q]; digits=12),
                    round(w[q]; digits=12)) for q in lo:hi]
        return sort(triples)
    end
    expected = naive_quadrature(half_cut_reference)
    @test expected.pointers == quadrature().pointers
    @test expected.is_cut == quadrature().is_cut
    for bg in 1:1
        got = sorted_triples(quadrature().pointers, quadrature().reference_r,
                             quadrature().reference_s, quadrature().weight, bg)
        want = sorted_triples(expected.pointers, expected.reference_r,
                              expected.reference_s, expected.weight, bg)
        @test length(got) == length(want)
        @test got == want
    end

    allocation_reference = build_cut_reference(48, 40; center=(0.501, 0.499))
    allocation_plan = StaticSystem(
        allocation_reference.model, allocation_reference.cut_geometry,
        roots_of(allocation_reference), 48, 40;
        form=kelvin_voigt_form(KelvinVoigtParameters{Float64}()))
    cutdata = StaticAssemblerGridapExt._cut_geometry_data(
        allocation_reference.cut_geometry, Float64)
    # Core staging/commit path must stay allocation-free; the adapter's
    # descriptor build (points copied once per refresh) is measured separately.
    staging = allocation_plan.refresh.quadrature_buffer
    cells = collect(Int32, 1:(allocation_reference.nx * allocation_reference.ny))
    StaticAssembler._stage_cut_quadrature!(
        staging, cells, cutdata,
        allocation_reference.nx, allocation_reference.ny,
        allocation_plan.hx, allocation_plan.hy, allocation_plan.cut_quadrature)
    allocation_bytes = @allocated StaticAssembler._stage_cut_quadrature!(
        staging, cells, cutdata,
        allocation_reference.nx, allocation_reference.ny,
        allocation_plan.hx, allocation_plan.hy, allocation_plan.cut_quadrature)
    @test allocation_bytes <= 4096
    @test allocation_bytes < allocation_plan.nx * allocation_plan.ny * sizeof(Int)
end

run_testset("direct quadrature refresh validation", "engine") do
    reference = build_cut_reference(
        1, 1; levelset_values=[-1.0, 1.0, 1.0, 1.0])
    roots = Int32[1]
    plan = StaticSystem(reference.model, reference.cut_geometry, roots, 1, 1;
                        form=kelvin_voigt_form(KelvinVoigtParameters{Float64}()))
    safe_coefficients = (fill(0.7, 4), zeros(4), zeros(4), zeros(4))
    snapshot() = (
        copy(plan.refresh.quadrature.pointers), copy(plan.refresh.quadrature.is_cut),
        copy(plan.refresh.quadrature.reference_r),
        copy(plan.refresh.quadrature.reference_s), copy(plan.refresh.quadrature.weight),
    )
    published = snapshot()
    B_published = plan.assemble.B
    assert_unchanged() = begin
        @test snapshot() == published
        @test plan.assemble.B === B_published
    end
    cut_geometry = reference.cut_geometry

    # Inconsistent descriptors (cell states disagreeing with the subcells
    # still attached to the cell) fail loudly in staging validation; the
    # committed tables stay untouched.
    bg_states = only(cut_geometry.ls_to_bgcell_to_inoutcut)
    old_state = bg_states[1]
    bg_states[1] = Int8(StaticAssembler.OUT)
    @test_throws ArgumentError refresh_geometry!(plan, cut_geometry, roots)
    bg_states[1] = old_state
    assert_unchanged()

    # A genuine transition to OUT (fresh geometry with no stale subcells) is
    # an accepted structural rebuild under the projected path, not an error.
    out_reference = build_cut_reference(1, 1; levelset_values=fill(1.0, 4),
                                        model=reference.model)
    @test refresh_geometry!(
        plan, out_reference.cut_geometry, [typemax(Int)]) === plan
    @test plan.refresh.cell_states == Int8[StaticAssembler.OUT]
    @test plan.refresh.last_roots == Int32[0]
    @test size(plan.assemble.B) == (0, 0)
    @test refresh_geometry!(plan, cut_geometry, roots) === plan
    published = snapshot()
    B_published = plan.assemble.B
    assert_unchanged()

    push!(cut_geometry.ls_to_bgcell_to_inoutcut, copy(bg_states))
    push!(cut_geometry.ls_to_subcell_to_inout,
          copy(only(cut_geometry.ls_to_subcell_to_inout)))
    @test_throws ArgumentError refresh_geometry!(plan, cut_geometry, roots)
    pop!(cut_geometry.ls_to_bgcell_to_inoutcut)
    pop!(cut_geometry.ls_to_subcell_to_inout)
    assert_unchanged()

    subcells = cut_geometry.subcells
    old_point = subcells.point_to_coords[1]
    subcells.point_to_coords[1] = Point(NaN, old_point[2])
    @test_throws ArgumentError refresh_geometry!(plan, cut_geometry, roots)
    subcells.point_to_coords[1] = old_point
    assert_unchanged()

    old_pointer = subcells.cell_to_points.ptrs[2]
    subcells.cell_to_points.ptrs[2] = old_pointer - 1
    @test_throws ArgumentError refresh_geometry!(plan, cut_geometry, roots)
    subcells.cell_to_points.ptrs[2] = old_pointer
    assert_unchanged()

    in_subcell = findfirst(==(IN), only(cut_geometry.ls_to_subcell_to_inout))
    first_point = subcells.cell_to_points.ptrs[in_subcell]
    point1 = subcells.cell_to_points.data[first_point]
    point2 = subcells.cell_to_points.data[first_point + 1]
    point3 = subcells.cell_to_points.data[first_point + 2]
    old_reference_point = subcells.point_to_rcoords[point2]
    old_physical_point = subcells.point_to_coords[point2]
    old_reference_point3 = subcells.point_to_rcoords[point3]
    old_physical_point3 = subcells.point_to_coords[point3]
    subcells.point_to_rcoords[point2] = old_reference_point3
    subcells.point_to_coords[point2] = old_physical_point3
    subcells.point_to_rcoords[point3] = old_reference_point
    subcells.point_to_coords[point3] = old_physical_point
    @test_throws ArgumentError refresh_geometry!(plan, cut_geometry, roots)
    subcells.point_to_rcoords[point2] = old_reference_point
    subcells.point_to_coords[point2] = old_physical_point
    subcells.point_to_rcoords[point3] = old_reference_point3
    subcells.point_to_coords[point3] = old_physical_point3
    assert_unchanged()

    subcells.point_to_coords[point2] = Point(
        old_physical_point[1] + 0.1, old_physical_point[2])
    @test_throws ArgumentError refresh_geometry!(plan, cut_geometry, roots)
    subcells.point_to_coords[point2] = old_physical_point
    assert_unchanged()

    changed_mesh = build_cut_reference(
        1, 1; levelset_values=[-1.0, 1.0, 1.0, 1.0])
    @test_throws ArgumentError refresh_geometry!(
        plan, changed_mesh.cut_geometry, roots)
    assert_unchanged()
    # Reference metadata lies (domain inconsistent with the model) do not
    # affect refresh: only the cut geometry itself is consumed.
    wrong_metrics = build_cut_reference(
        1, 1; domain=(0.0, 2.0, 0.0, 1.0),
        levelset_values=[-1.0, 1.0, 1.0, 1.0], model=reference.model)
    @test refresh_geometry!(
        plan, wrong_metrics.cut_geometry, Int32[1]) === plan
    @test refresh_geometry!(plan, cut_geometry, roots) === plan
    published = snapshot()
    B_published = plan.assemble.B
    assert_unchanged()
    wrong_partition = CartesianDiscreteModel(Point(0.0, 0.0), Point(2.0, 1.0), (2, 1))
    wrong_cutgeo = cut(wrong_partition,
                       GridapEmbedded.LevelSetCutters.DiscreteGeometry(
                           fill(-1.0, 6), vec(collect(get_node_coordinates(wrong_partition)))))
    @test_throws ArgumentError StaticSystem(
        wrong_partition, wrong_cutgeo, Int32[1, 1], 1, 1;
        form=kelvin_voigt_form(KelvinVoigtParameters{Float64}()))
    assert_unchanged()
    # Malformed aggregate roots are plain-vector inputs, so these guards are
    # reachable with no geometry sabotage. Aggregate validation runs before
    # staging, so the committed tables stay untouched.
    @test_throws ArgumentError refresh_geometry!(plan, cut_geometry, Int32[])
    assert_unchanged()
    @test_throws ArgumentError refresh_geometry!(plan, cut_geometry, Int32[2])
    assert_unchanged()
    # A well-crafted aggregation engine already outputs valid partitions:
    # every non-OUT cell reaches an in-domain, non-OUT, self-rooted root in
    # one hop. This locks the assumption the validator relies on, checked
    # independently of the validator itself.
    partition_reference = build_cut_reference(
        4, 3; radius=0.34, center=(0.5, 0.5))
    partition_bg = only(partition_reference.cut_geometry.ls_to_bgcell_to_inoutcut)
    partition_roots = aggregate(
        AggregateAllCutCells(), partition_reference.cut_geometry)
    @test length(partition_roots) == length(partition_bg)
    for cell in eachindex(partition_bg)
        partition_bg[cell] == StaticAssembler.OUT && continue
        root = Int(partition_roots[cell])
        @test 1 <= root <= length(partition_bg)
        @test partition_bg[root] != StaticAssembler.OUT
        @test Int(partition_roots[root]) == root
    end
    B_after, _ = assemble!(plan, safe_coefficients; threaded=false)
    @test B_after == B_published
end

run_testset("projection-neutral geometry refresh", "engine") do
    nx, ny = 2, 1
    interior = build_cut_reference(nx, ny; levelset_values=fill(-1.0, 6))
    cut = build_cut_reference(
        nx, ny; levelset_values=[1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        model=interior.model)
    roots = Int32[2, 2]
    form = kelvin_voigt_form(KelvinVoigtParameters{Float64}())
    plan = StaticSystem(interior.model, interior.cut_geometry, roots, nx, ny; form)
    E = plan.refresh.E
    P = plan.refresh.coefficient_prolongation
    B = plan.assemble.B
    dofs = plan.assemble.test_cell_dofs
    constraints = plan.assemble.test_constraints
    slots = plan.assemble.scatter_slots

    refresh_geometry!(plan, cut.cut_geometry, roots)
    @test plan.refresh.E === E
    @test plan.refresh.coefficient_prolongation === P
    @test plan.assemble.B === B
    @test plan.assemble.test_cell_dofs === dofs
    @test plan.assemble.test_constraints === constraints
    @test plan.assemble.scatter_slots === slots
    @test size(E, 2) < size(E, 1)

    fresh = StaticSystem(cut.model, cut.cut_geometry, roots, nx, ny; form)
    values = (
        collect(range(0.7, 1.2; length=6)),
        collect(range(-0.1, 0.1; length=6)),
        collect(range(0.2, -0.1; length=6)),
        collect(range(-0.05, 0.08; length=6)),
    )
    actual_B, actual_b = assemble!(plan, values; threaded=false)
    expected_B, expected_b = assemble!(fresh, values; threaded=false)
    @test actual_B == expected_B
    @test actual_b == expected_b

    aliased_roots = plan.refresh.last_roots
    aliased_roots .= Int32[1, 2]
    refresh_geometry!(plan, cut.cut_geometry, aliased_roots)
    unaggregated = StaticSystem(cut.model, cut.cut_geometry, Int32[1, 2], nx, ny; form)
    @test plan.refresh.E == unaggregated.refresh.E
    @test plan.assemble.B == unaggregated.assemble.B

    overlap = build_cut_reference(3, 1; levelset_values=fill(-1.0, 8))
    overlap_plan = StaticSystem(
        overlap.model, overlap.cut_geometry, Int32[2, 2, 3], 3, 1; form)
    overlap_roots = view(overlap_plan.refresh.last_roots, [3, 1, 3])
    expected_roots = copy(overlap_roots)
    refresh_geometry!(overlap_plan, overlap.cut_geometry, overlap_roots)
    overlap_fresh = StaticSystem(
        overlap.model, overlap.cut_geometry, expected_roots, 3, 1; form)
    @test overlap_plan.refresh.last_roots == expected_roots
    @test overlap_plan.refresh.E == overlap_fresh.refresh.E
end

run_testset("runtime background coefficients", "equivalence") do
    nx, ny = 6, 4
    reference = build_cut_reference(nx, ny; radius=0.34, center=(0.52, 0.48))
    roots = aggregate(AggregateAllCutCells(), reference.cut_geometry)
    system = StaticSystem(reference.model, reference.cut_geometry, roots, nx, ny;
                          form=kelvin_voigt_form(KelvinVoigtParameters{Float64}()))
    @test size(system.refresh.coefficient_nodes) == (4, nx * ny)
    @test system.refresh.coefficient_nodes[:, 1] == Int32[1, 2, nx + 2, nx + 3]
    @test :form ∉ fieldnames(typeof(system))

    rng = MersenneTwister(20260831)
    buffers = ntuple(2) do _
        n = (nx + 1) * (ny + 1)
        (0.55 .+ 0.35 .* rand(rng, n),
         0.08 .* randn(rng, n), 0.08 .* randn(rng, n), 0.04 .* randn(rng, n))
    end

    results = Tuple[]
    for values in (buffers[1], buffers[2], buffers[1])
        B, b = assemble!(system, values; threaded=true)
        push!(results, (copy(B), copy(b)))
    end
    @test results[1][1] == results[3][1]
    @test results[1][2] == results[3][2]
    @test results[1][2] != results[2][2]
end


run_testset("stable projected cut AgFEM", "equivalence") do
    nx, ny = 8, 7
    model = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (nx, ny))
    parameters = KelvinVoigtParameters{Float64}(ξ=0.002, A=0.017, ν=0.43)
    rng = MersenneTwister(20260901)
    nnodes = (nx + 1) * (ny + 1)
    values = (0.55 .+ 0.35 .* rand(rng, nnodes),
              0.08 .* randn(rng, nnodes), 0.08 .* randn(rng, nnodes),
              0.04 .* randn(rng, nnodes))

    function check_projected(plan, reference, roots)
        op = cut_agfem_operator(reference, roots, values)
        B, b_m = assemble!(plan, values; threaded=false)
        E = plan.refresh.E
        n_m = size(E, 2)
        ours = stable_to_master(E)
        @test sort(op.compact_to_stable) == sort(findall(!iszero, ours))
        @test size(B) == (n_m, n_m)
        gridap_to_ours = ours[op.compact_to_stable]
        @test B[gridap_to_ours, gridap_to_ours] ≈ op.matrix rtol=8e-13 atol=2e-14
        @test b_m[gridap_to_ours] ≈ op.vector rtol=8e-13 atol=2e-14
        @test B ≈ transpose(B) rtol=1e-14 atol=1e-15

        # E prolongation of master DOFs reproduces the Gridap full field.
        u_m = randn(rng, n_m)
        u_full = E * u_m
        u_gridap = u_m[invperm(gridap_to_ours)]
        expected_recovery = zeros(2nnodes)
        standard_function = interpolate(
            FEFunction(op.vector_test, u_gridap), op.vector_standard)
        expected_recovery[op.standard] .= get_free_dof_values(standard_function)
        @test u_full ≈ expected_recovery rtol=5e-14 atol=5e-14
        @test mul!(similar(u_full), E, u_m) == u_full
        @test restrict_solution!(similar(u_m), plan, u_full) == u_m
        # Eᵀ is the true transpose: <E*u_m, v> == <u_m, Eᵀ*v>.
        v_full = randn(rng, 2nnodes)
        @test dot(u_full, v_full) ≈ dot(u_m, E' * v_full) rtol=5e-14 atol=5e-14
        return nothing
    end

    reference = build_cut_reference(8, 7; radius=0.34, center=(0.49, 0.51),
                                    parameters=parameters)
    cutgeo, model = reference.cut_geometry, reference.model
    roots = aggregate(AggregateAllCutCells(), cutgeo)
    plan = StaticSystem(
        model, cutgeo, roots, nx, ny;
        form=kelvin_voigt_form(parameters),
        interior_quadrature=StaticAssemblerOracle._gridap_interior_quadrature(6))
    check_projected(plan, reference, roots)

    pointers = plan.refresh.quadrature.pointers
    initial_states = copy(plan.refresh.cell_states)
    moved_reference = build_cut_reference(8, 7; radius=0.31, center=(0.54, 0.48),
                                          model=model, parameters=parameters)
    moved_cutgeo = moved_reference.cut_geometry
    moved_roots = aggregate(AggregateAllCutCells(), moved_cutgeo)
    @test only(moved_cutgeo.ls_to_bgcell_to_inoutcut) != initial_states
    @test refresh_geometry!(plan, moved_cutgeo, moved_roots) === plan
    @test plan.refresh.quadrature.pointers === pointers
    check_projected(plan, moved_reference, moved_roots)

    # Same aggregate roots with the same geometry: the static assembly tables
    # (B pattern, cell DOFs, scatter slots) are reused, only values change.
    stable_roots = copy(plan.refresh.last_roots)
    B_pattern = plan.assemble.B
    cell_dofs = plan.assemble.test_cell_dofs
    scatter_slots = plan.assemble.scatter_slots
    @test refresh_geometry!(plan, moved_cutgeo, stable_roots) === plan
    @test plan.assemble.B === B_pattern
    @test plan.assemble.test_cell_dofs === cell_dofs
    @test plan.assemble.scatter_slots === scatter_slots
    check_projected(plan, moved_reference, stable_roots)

    cut_rooted = copy(moved_roots)
    cut_cell = findfirst(
        ==(GridapEmbedded.Interfaces.CUT),
        only(moved_cutgeo.ls_to_bgcell_to_inoutcut))
    cut_rooted[cut_cell] = cut_cell
    @test refresh_geometry!(plan, moved_cutgeo, cut_rooted) === plan
    check_projected(plan, moved_reference, cut_rooted)
end
