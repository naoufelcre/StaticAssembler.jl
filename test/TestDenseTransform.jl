# Focused SA4 tests for the dense AgFEM transform. The identity and
# permutation branches are unchanged; the genuine dense branch replaces the
# scalar `C·K·Cᵀ` / `C·f` loops with `LinearAlgebra.mul!` on views sized from
# `m = size(C, 1)` and `n = size(C, 2)`. These tests:
#
#   * compare the dense kernel against an independent scalar reference with a
#     nonsymmetric `K` over square, tall (`m > n`) and wide (`m < n`) shapes;
#   * use oversized scratch buffers with sentinel padding and assert the
#     padding is never written;
#   * freeze the identity and permutation arithmetic exactly and check the
#     dense `mul!` branch against a permutation matrix with scale-aware
#     tolerance only for that cross-branch comparison;
#   * exercise actual Q1/Q2 cut-cell constraints and require genuine dense
#     cells with `m > n`;
#   * compare whole assembled matrices and loads against an independent
#     per-cell scalar transform, serial and threaded, for Q1/Q2, boundary and
#     supported mixed forms, and across structural/neutral refreshes.
#
# Runnable standalone from the repository root:
#
#     julia --project=@v1.11 test/TestDenseTransform.jl
#
# When wired into `test/runtests.jl` the oracle include below is skipped
# because the oracle module is already loaded by the caller.

using Test
using LinearAlgebra
using Random
using GridapEmbedded
using StaticAssembler
using StaticAssembler: ThreadBuffer, _transform_agfem!, _classify_constraint,
    _assemble_cell!, _scatter_projected!,
    CONSTRAINT_DENSE, CONSTRAINT_PERMUTATION, CONSTRAINT_IDENTITY, OUT

if !isdefined(Main, :StaticAssemblerOracle)
    include(joinpath(@__DIR__, "..", "oracle", "StaticAssemblerOracle.jl"))
end
using .StaticAssemblerOracle

# Sentinel written into every buffer slot that the transform must not touch.
const DENSE_SENTINEL = -9.87654321e30

# Scalar boundary operator for the standalone boundary-form test. The normal
# runtests.jl defines its own `ScaledValueOp`; a distinct name here avoids a
# redefinition clash when this file is included from the normal runner.
struct DenseBoundaryOp <: StaticAssembler.AbstractDifferentialOperator
    scale::Float64
end

@inline function StaticAssembler._op_pair(test::DenseBoundaryOp, trial::DenseBoundaryOp,
                                          row::Int, col::Int,
                                          sh_t, gx_t, gy_t, sh_c, gx_c, gy_c)
    return test.scale * trial.scale * sh_t[row] * sh_c[col]
end

@inline function StaticAssembler._rhs_contract!(fe, op::DenseBoundaryOp, slot,
                                                shape, gx, gy, c, w, offset::Int)
    value = op.scale * slot.scale * slot.source(c) * w
    @inbounds for row in eachindex(shape)
        fe[offset + row] += value * shape[row]
    end
    return nothing
end

# Evidence ledger: every comparison records the observed absolute difference
# and the tolerance it was checked against. Printed at the end of the file.
const SA4_RECORDS = NamedTuple[]
const SA4_ALLOCATIONS = NamedTuple[]
function record_sa4!(label, left_error, left_tol, matrix_error, matrix_tol,
                     vector_error, vector_tol)
    push!(SA4_RECORDS, (; label, left_error, left_tol, matrix_error, matrix_tol,
                        vector_error, vector_tol))
    return nothing
end

_absmax(a) = maximum(abs, a; init=0.0)

# Absolute-product scales for the three dense products. `left = C·K`,
# `matrix = C·K·Cᵀ`, `vector = C·f`:
#     S_left   = max |C| |K|
#     S_matrix = max |C| |K| |C|ᵀ
#     S_vector = max |C| |f|
function dense_scales(C, K, f)
    absC = abs.(C)
    left_abs = absC * abs.(K)
    matrix_abs = left_abs * transpose(absC)
    vector_abs = absC * abs.(f)
    return _absmax(left_abs), _absmax(matrix_abs), _absmax(vector_abs)
end

# Separate scale-aware roundoff bounds for left, final matrix, and vector. Each
# result is a sum of `n` products, so the standard accumulation factor is
# `gamma = factor · n · eps`; apply it to the matching absolute-product scale.
# The vector bound uses `|C| |f|`, not the stiffness scale. `factor` is a modest
# constant on top of `gamma_n`.
function dense_tolerances(C, K, f; factor=4.0)
    n = size(C, 2)
    left_scale, matrix_scale, vector_scale = dense_scales(C, K, f)
    gamma = factor * max(n, 1) * eps(Float64)
    return (gamma * max(1.0, left_scale),
            gamma * max(1.0, matrix_scale),
            gamma * max(1.0, vector_scale))
end

# Independent scalar reference. Deliberately does not call the production
# kernel: `left = C·K`, `eff = left·Cᵀ`, `v = C·f`, with explicit triple loops
# and the original accumulation order.
function scalar_reference(C, K, f)
    T = float(eltype(K))
    m, n = size(C)
    left = zeros(T, m, n)
    @inbounds for raw_column in 1:n, row in 1:m
        value = zero(T)
        for k in 1:n
            value += C[row, k] * K[k, raw_column]
        end
        left[row, raw_column] = value
    end
    effective = zeros(T, m, m)
    @inbounds for column in 1:m, row in 1:m
        value = zero(T)
        for k in 1:n
            value += left[row, k] * C[column, k]
        end
        effective[row, column] = value
    end
    vector = zeros(T, m)
    @inbounds for row in 1:m
        value = zero(T)
        for k in 1:n
            value += C[row, k] * f[k]
        end
        vector[row] = value
    end
    return left, effective, vector
end

# In-place independent scalar transform on a production scratch, mirroring the
# pre-SA4 dense loops. Used to build the whole-assembly reference.
function scalar_transform!(scratch, C)
    T = eltype(C)
    m, n = size(C)
    raw = scratch.raw_matrix
    left = scratch.left_matrix
    effective = scratch.effective_matrix
    @inbounds for raw_column in 1:n, row in 1:m
        value = zero(T)
        for k in 1:n
            value += C[row, k] * raw[k, raw_column]
        end
        left[row, raw_column] = value
    end
    @inbounds for column in 1:m, row in 1:m
        value = zero(T)
        for k in 1:n
            value += left[row, k] * C[column, k]
        end
        effective[row, column] = value
    end
    @inbounds for row in 1:m
        value = zero(T)
        for k in 1:n
            value += C[row, k] * scratch.raw_vector[k]
        end
        scratch.effective_vector[row] = value
    end
    return nothing
end

# Build an oversized scratch, run the production dense kernel, and return it.
# The active raw block is (n × n); every other slot stays at the sentinel.
function production_kernel(C, K, f; pad_m=3, pad_n=2)
    m, n = size(C)
    raw = fill(DENSE_SENTINEL, n + pad_n, n + pad_n)
    raw[1:n, 1:n] .= K
    raw_vector = fill(DENSE_SENTINEL, n + pad_n)
    raw_vector[1:n] .= f
    left = fill(DENSE_SENTINEL, m + pad_m, n + pad_n)
    effective = fill(DENSE_SENTINEL, m + pad_m, m + pad_m)
    effective_vector = fill(DENSE_SENTINEL, m + pad_m)
    scratch = ThreadBuffer(raw, raw_vector, left, effective, effective_vector)
    _transform_agfem!(scratch, C, CONSTRAINT_DENSE, Int32[])
    return scratch
end

# Assert every padding slot is still the sentinel (relative to active sizes).
function check_padding(scratch, m, n)
    raw = scratch.raw_matrix
    left = scratch.left_matrix
    effective = scratch.effective_matrix
    @test all(==(DENSE_SENTINEL), raw[n+1:end, :])
    @test all(==(DENSE_SENTINEL), raw[:, n+1:end])
    @test all(==(DENSE_SENTINEL), scratch.raw_vector[n+1:end])
    @test all(==(DENSE_SENTINEL), left[m+1:end, :])
    @test all(==(DENSE_SENTINEL), left[:, n+1:end])
    @test all(==(DENSE_SENTINEL), effective[m+1:end, :])
    @test all(==(DENSE_SENTINEL), effective[:, m+1:end])
    @test all(==(DENSE_SENTINEL), scratch.effective_vector[m+1:end])
    return nothing
end

# ---------------------------------------------------------------------------
# Independent scalar reference assemblies, mirroring assemble_serial!/threaded!
# but calling the test-local scalar transform for every cell. This is the
# independent reference for the whole matrix and load.
# ---------------------------------------------------------------------------

function reference_serial!(plan, coefficients, facet_fields)
    a = plan.assemble
    r = plan.refresh
    fill!(a.B.nzval, zero(eltype(a.B)))
    fill!(a.b_m, zero(eltype(a.b_m)))
    scratch = a.thread_buffers[Threads.threadid()]
    @inbounds for cell in 1:(plan.nx * plan.ny)
        r.cell_states[cell] == OUT && continue
        _assemble_cell!(scratch, plan, cell, coefficients, facet_fields)
        scalar_transform!(scratch, a.test_constraints[cell])
        _scatter_projected!(plan, scratch, cell)
    end
    return a.B, a.b_m
end

function reference_threaded!(plan, coefficients, facet_fields)
    a = plan.assemble
    r = plan.refresh
    fill!(a.B.nzval, zero(eltype(a.B)))
    fill!(a.b_m, zero(eltype(a.b_m)))
    for color in a.colors
        Threads.@threads :static for index in eachindex(color)
            cell = Int(color[index])
            r.cell_states[cell] == OUT && continue
            scratch = a.thread_buffers[Threads.threadid()]
            _assemble_cell!(scratch, plan, cell, coefficients, facet_fields)
            scalar_transform!(scratch, a.test_constraints[cell])
            _scatter_projected!(plan, scratch, cell)
        end
    end
    return a.B, a.b_m
end

# Compare production assembly against the independent scalar reference, serial
# and threaded. Returns the four absolute differences.
function compare_assembly!(plan, coefficients, facet_fields; label="")
    a = plan.assemble
    Bp, bp = assemble!(plan, coefficients; facet_fields=facet_fields, threaded=false)
    prod_serial = copy(Bp); prod_serial_b = copy(bp)
    Bs, bs = reference_serial!(plan, a.extended_coefficients, facet_fields)
    ref_serial = copy(Bs); ref_serial_b = copy(bs)

    Bt, bt = assemble!(plan, coefficients; facet_fields=facet_fields, threaded=true)
    prod_thread = copy(Bt); prod_thread_b = copy(bt)
    Bst, bst = reference_threaded!(plan, a.extended_coefficients, facet_fields)
    ref_thread = copy(Bst); ref_thread_b = copy(bst)

    dB_serial = _absmax(Matrix(prod_serial - ref_serial))
    db_serial = _absmax(prod_serial_b - ref_serial_b)
    dB_thread = _absmax(Matrix(prod_thread - ref_thread))
    db_thread = _absmax(prod_thread_b - ref_thread_b)
    tol_B = 1e-12 * max(1.0, _absmax(Matrix(ref_serial)),
                        _absmax(Matrix(ref_thread)))
    tol_b = 1e-12 * max(1.0, _absmax(ref_serial_b), _absmax(ref_thread_b))

    @test dB_serial <= tol_B
    @test db_serial <= tol_b
    @test dB_thread <= tol_B
    @test db_thread <= tol_b

    worst_B = max(dB_serial, dB_thread)
    worst_b = max(db_serial, db_thread)
    record_sa4!(label, NaN, NaN, worst_B, tol_B, worst_b, tol_b)
    return (; dB_serial, db_serial, dB_thread, db_thread)
end

# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------

function build_kelvin_voigt(nx, ny; element, radius=0.42, center=(0.5, 0.5))
    reference = build_cut_reference(nx, ny; radius, center)
    roots = aggregate(AggregateAllCutCells(), reference.cut_geometry)
    form = kelvin_voigt_form(KelvinVoigtParameters{Float64}(); element)
    plan = StaticSystem(reference.model, reference.cut_geometry, roots, nx, ny;
                        form=form)
    return reference, plan
end

# True when the plan has at least one genuine dense constraint whose master
# count exceeds its raw count (the unioned-root case the views must handle).
function has_tall_dense(plan)
    a = plan.assemble
    return any(eachindex(a.constraint_kinds)) do cell
        a.constraint_kinds[cell] == CONSTRAINT_DENSE || return false
        m, n = size(a.test_constraints[cell])
        return m > n
    end
end

# True when some genuine dense cell also carries embedded-boundary quadrature,
# so the dense transform and the boundary tables are exercised on one cell.
function has_dense_boundary_cell(plan)
    a = plan.assemble
    bq = plan.refresh.boundary_quadrature
    return any(eachindex(a.constraint_kinds)) do cell
        a.constraint_kinds[cell] == CONSTRAINT_DENSE || return false
        return Int(bq.pointers[cell]) < Int(bq.pointers[cell + 1])
    end
end

# ---------------------------------------------------------------------------
# Kernel-level tests.
# ---------------------------------------------------------------------------

@testset "SA4 dense kernel vs independent scalar C·K·Cᵀ and C·f" begin
    rng = MersenneTwister(20261007)
    # square, m > n (unioned roots), and m < n.
    shapes = ((1, 1), (2, 2), (4, 4), (9, 9), (16, 16),
              (2, 5), (3, 7), (6, 16), (48, 18),
              (5, 2), (7, 3), (16, 6), (18, 48))
    for (m, n) in shapes
        C = randn(rng, m, n)
        K = randn(rng, n, n)              # deliberately nonsymmetric
        f = randn(rng, n)
        left_ref, effective_ref, vector_ref = scalar_reference(C, K, f)
        scratch = production_kernel(C, K, f)
        tol_left, tol_matrix, tol_vector = dense_tolerances(C, K, f)
        left_error = _absmax(view(scratch.left_matrix, 1:m, 1:n) .- left_ref)
        matrix_error = _absmax(view(scratch.effective_matrix, 1:m, 1:m) .- effective_ref)
        vector_error = _absmax(view(scratch.effective_vector, 1:m) .- vector_ref)
        @test left_error <= tol_left
        @test matrix_error <= tol_matrix
        @test vector_error <= tol_vector
        check_padding(scratch, m, n)
        record_sa4!("kernel (m=$m, n=$n)", left_error, tol_left,
                    matrix_error, tol_matrix, vector_error, tol_vector)
    end
end

@testset "SA4 operand-scaled tolerances separate K and f" begin
    # The vector bound must track |C| |f|, not the stiffness scale, and the
    # matrix bound must track |C| |K| |Cᵀ|, not the load scale. Reuses the same
    # scalar reference and production kernel, with the operands rescaled.
    rng = MersenneTwister(20261014)
    for (m, n) in ((5, 3), (3, 5))
        C = randn(rng, m, n)
        Kbase = randn(rng, n, n)          # nonsymmetric
        fbase = randn(rng, n)
        for (k_scale, f_scale) in ((1e-9, 1e9), (1e9, 1e-9))
            K = k_scale .* Kbase
            f = f_scale .* fbase
            left_ref, effective_ref, vector_ref = scalar_reference(C, K, f)
            scratch = production_kernel(C, K, f)
            tol_left, tol_matrix, tol_vector = dense_tolerances(C, K, f)
            left_error = _absmax(view(scratch.left_matrix, 1:m, 1:n) .- left_ref)
            matrix_error = _absmax(view(scratch.effective_matrix, 1:m, 1:m) .- effective_ref)
            vector_error = _absmax(view(scratch.effective_vector, 1:m) .- vector_ref)
            @test left_error <= tol_left
            @test matrix_error <= tol_matrix
            @test vector_error <= tol_vector
            check_padding(scratch, m, n)
            record_sa4!("scaled (m=$m, n=$n, Kx$k_scale, fx$f_scale)",
                        left_error, tol_left, matrix_error, tol_matrix,
                        vector_error, tol_vector)
        end
        # Bounds are insensitive to the unrelated operand scale: f does not
        # enter the matrix bound and K does not enter the vector bound.
        _, matrix_small_f, _ = dense_tolerances(C, 1e9 .* Kbase, 1e-9 .* fbase)
        _, matrix_large_f, _ = dense_tolerances(C, 1e9 .* Kbase, 1e9 .* fbase)
        @test matrix_small_f == matrix_large_f
        _, _, vector_small_k = dense_tolerances(C, 1e-9 .* Kbase, 1e9 .* fbase)
        _, _, vector_large_k = dense_tolerances(C, 1e9 .* Kbase, 1e9 .* fbase)
        @test vector_small_k == vector_large_k
        # The vector bound still grows with the load scale.
        _, _, vector_tiny_f = dense_tolerances(C, 1e9 .* Kbase, 1e-9 .* fbase)
        _, _, vector_huge_f = dense_tolerances(C, 1e9 .* Kbase, 1e9 .* fbase)
        @test vector_huge_f > 1e6 * vector_tiny_f
    end
end

@testset "SA4 identity and permutation branches are frozen" begin
    rng = MersenneTwister(20261008)
    for n in (1, 2, 4, 9, 18)
        K = randn(rng, n, n)
        f = randn(rng, n)

        Cid = Matrix{Float64}(I, n, n)
        kind, perm = _classify_constraint(Cid)
        @test kind == CONSTRAINT_IDENTITY
        scratch = ThreadBuffer(copy(K), copy(f), zeros(n, n), zeros(n, n), zeros(n))
        _transform_agfem!(scratch, Cid, kind, perm)
        @test scratch.effective_matrix == K
        @test scratch.effective_vector == f

        p = randperm(rng, n)
        if p != collect(1:n)
            Cp = zeros(Float64, n, n)
            for row in 1:n
                Cp[row, p[row]] = 1.0
            end
            kind, perm = _classify_constraint(Cp)
            @test kind == CONSTRAINT_PERMUTATION
            scratch = ThreadBuffer(copy(K), copy(f), zeros(n, n), zeros(n, n), zeros(n))
            _transform_agfem!(scratch, Cp, kind, perm)
            effective_ref = zeros(Float64, n, n)
            for column in 1:n, row in 1:n
                effective_ref[row, column] = K[perm[row], perm[column]]
            end
            @test scratch.effective_matrix == effective_ref
            @test scratch.effective_vector == f[perm]

            # Cross-branch only: the dense mul! kernel on the same permutation
            # matrix must agree, but BLAS may reassociate, so compare with the
            # scale-aware tolerance and record the observed difference.
            dense = ThreadBuffer(copy(K), copy(f), zeros(n, n), zeros(n, n), zeros(n))
            _transform_agfem!(dense, Cp, CONSTRAINT_DENSE, Int32[])
            matrix_error = _absmax(dense.effective_matrix .- effective_ref)
            vector_error = _absmax(dense.effective_vector .- f[perm])
            _, tol_matrix, tol_vector = dense_tolerances(Cp, K, f)
            @test matrix_error <= tol_matrix
            @test vector_error <= tol_vector
            record_sa4!("permutation cross-branch (n=$n)", NaN, NaN,
                        matrix_error, tol_matrix, vector_error, tol_vector)
        end
    end

    # Near-permutation stays dense and is projected by the dense kernel.
    near = Matrix{Float64}(I, 4, 4)
    near[1, 2] = 1e-14
    @test _classify_constraint(near)[1] == CONSTRAINT_DENSE
    K = randn(rng, 4, 4)
    f = randn(rng, 4)
    _, effective_ref, vector_ref = scalar_reference(near, K, f)
    scratch = production_kernel(near, K, f)
    _, tol_matrix, tol_vector = dense_tolerances(near, K, f)
    @test _absmax(view(scratch.effective_matrix, 1:4, 1:4) .- effective_ref) <=
          tol_matrix
    @test _absmax(view(scratch.effective_vector, 1:4) .- vector_ref) <=
          tol_vector
end

@testset "SA4 actual Q1/Q2 cut constraints" begin
    rng = MersenneTwister(20261009)
    for (label, element) in (("Q1", Q1ReferenceQuadrilateral()),
                            ("Q2", Q2ReferenceQuadrilateral()))
        _, plan = build_kelvin_voigt(6, 5; element)
        a = plan.assemble
        dense_cells = findall(==(CONSTRAINT_DENSE), a.constraint_kinds)
        @test !isempty(dense_cells)
        @test any(==(CONSTRAINT_PERMUTATION), a.constraint_kinds)

        shapes = Set{Tuple{Int,Int}}()
        worst_left = 0.0
        worst_matrix = 0.0
        worst_vector = 0.0
        worst_tol_left = 0.0
        worst_tol_matrix = 0.0
        worst_tol_vector = 0.0
        for cell in dense_cells
            C = a.test_constraints[cell]
            m, n = size(C)
            push!(shapes, (m, n))
            K = randn(rng, n, n)
            f = randn(rng, n)
            left_ref, effective_ref, vector_ref = scalar_reference(C, K, f)
            scratch = production_kernel(C, K, f)
            left_error = _absmax(view(scratch.left_matrix, 1:m, 1:n) .- left_ref)
            matrix_error = _absmax(view(scratch.effective_matrix, 1:m, 1:m) .- effective_ref)
            vector_error = _absmax(view(scratch.effective_vector, 1:m) .- vector_ref)
            tol_left, tol_matrix, tol_vector = dense_tolerances(C, K, f)
            @test left_error <= tol_left
            @test matrix_error <= tol_matrix
            @test vector_error <= tol_vector
            worst_left = max(worst_left, left_error)
            worst_matrix = max(worst_matrix, matrix_error)
            worst_vector = max(worst_vector, vector_error)
            worst_tol_left = max(worst_tol_left, tol_left)
            worst_tol_matrix = max(worst_tol_matrix, tol_matrix)
            worst_tol_vector = max(worst_tol_vector, tol_vector)
        end
        # Aggregated cut cells have more master DOFs than raw DOFs.
        @test any(shape -> shape[1] > shape[2], shapes)
        record_sa4!("$label actual cut constraints ($(length(dense_cells)) dense, " *
                    "shapes $(sort(collect(shapes))))",
                    worst_left, worst_tol_left, worst_matrix, worst_tol_matrix,
                    worst_vector, worst_tol_vector)
    end
end

# ---------------------------------------------------------------------------
# Whole-assembly tests.
# ---------------------------------------------------------------------------

@testset "SA4 whole assembly vs independent scalar reference" begin
    rng = MersenneTwister(20261010)
    for (label, element) in (("Q1", Q1ReferenceQuadrilateral()),
                            ("Q2", Q2ReferenceQuadrilateral()))
        reference, plan = build_kelvin_voigt(6, 5; element)
        a = plan.assemble
        @test any(==(CONSTRAINT_DENSE), a.constraint_kinds)
        @test has_tall_dense(plan)
        nnodes = (reference.nx + 1) * (reference.ny + 1)
        values = (0.55 .+ 0.35 .* rand(rng, nnodes),
                  0.08 .* randn(rng, nnodes),
                  0.08 .* randn(rng, nnodes),
                  0.04 .* randn(rng, nnodes))
        compare_assembly!(plan, values, (); label="$label whole assembly 6x5")
    end
end

@testset "SA4 boundary and supported mixed forms" begin
    # Q1 scalar boundary form on a larger cut grid so genuine dense cells
    # appear (the 1x1 regression plan is identity-only).
    reference = build_cut_reference(4, 4; radius=0.38, center=(0.5, 0.5))
    roots = aggregate(AggregateAllCutCells(), reference.cut_geometry)
    space = StaticAssembler.FiniteElement(Q1ReferenceQuadrilateral(), 1)
    boundary_coefficient = StaticAssembler.StateFunction() do point
        nx, ny = boundary_normal(point)
        gx, gy = coefficient_gradient(point, 1)
        facet_value(point, 1) + point[1] + 0.2 * (gx * nx + gy * ny)
    end
    boundary_source = point -> begin
        nx, ny = boundary_normal(point)
        tx, ty = boundary_tangent(point)
        gx, gy = coefficient_gradient(point, 1)
        facet_value(point, 1) - point[1] + gx * tx + gy * ty + 2nx + 3ny
    end
    boundary_form = StaticAssembler.ElementVariationalForm(
        space, (), (), 1;
        boundary_bilinear=(StaticAssembler.BilinearSlot(
            DenseBoundaryOp(1.0), DenseBoundaryOp(1.0), boundary_coefficient),),
        boundary_rhs=(StaticAssembler.RhsSlot(
            DenseBoundaryOp(1.0), boundary_source, 1.0),),
        nfacetfields=1)
    boundary_plan = StaticSystem(
        reference.model, reference.cut_geometry, roots, 4, 4; form=boundary_form)
    nfacets = boundary_plan.refresh.boundary_quadrature.nfacets
    @test nfacets > 0
    @test any(==(CONSTRAINT_DENSE), boundary_plan.assemble.constraint_kinds)
    @test has_tall_dense(boundary_plan)
    @test has_dense_boundary_cell(boundary_plan)
    rng = MersenneTwister(20261011)
    nnodes = (4 + 1) * (4 + 1)
    boundary_values = (0.5 .+ 0.2 .* rand(rng, nnodes),)
    boundary_fields = ([0.5 + 0.01 * i for i in 1:nfacets],)
    compare_assembly!(boundary_plan, boundary_values, boundary_fields;
                      label="boundary Q1 4x4 (nfacets=$nfacets)")

    # Taylor-Hood Q2/Q1 Stokes: mixed vector/scalar blocks, no boundary terms.
    stokes_reference = build_cut_stokes_reference(4, 3; radius=0.42)
    stokes_roots = aggregate(AggregateAllCutCells(), stokes_reference.cut_geometry)
    stokes_plan = StaticSystem(
        stokes_reference.model, stokes_reference.cut_geometry, stokes_roots, 4, 3;
        form=stokes_form(StokesParameters{Float64}()),
        interior_quadrature=StaticAssemblerOracle._gridap_interior_quadrature(10))
    @test any(==(CONSTRAINT_DENSE), stokes_plan.assemble.constraint_kinds)
    @test has_tall_dense(stokes_plan)
    compare_assembly!(stokes_plan, (), (); label="stokes Q2/Q1 4x3")

    # Article Q2/Q1/Q1 form: mixed blocks plus embedded boundary terms.
    article_reference = build_cut_reference(4, 3; radius=0.38, center=(0.49, 0.51))
    article_roots = aggregate(
        AggregateAllCutCells(), article_reference.cut_geometry)
    article_data = StaticAssemblerOracle._article_data(article_reference)
    article_plan = StaticSystem(
        article_reference.model, article_reference.cut_geometry, article_roots, 4, 3;
        form=article_form(ArticleParameters{Float64}()),
        interior_quadrature=StaticAssemblerOracle._gridap_interior_quadrature(4),
        cut_quadrature=StaticAssemblerOracle._gridap_cut_quadrature(4))
    @test any(==(CONSTRAINT_DENSE), article_plan.assemble.constraint_kinds)
    @test has_tall_dense(article_plan)
    @test article_plan.refresh.boundary_quadrature.nfacets > 0
    @test has_dense_boundary_cell(article_plan)
    compare_assembly!(article_plan, article_data.coefficients,
                      article_data.facet_fields; label="article Q2/Q1/Q1 4x3")
end

@testset "SA4 refresh: structural and neutral" begin
    rng = MersenneTwister(20261012)
    reference, plan = build_kelvin_voigt(6, 5; element=Q2ReferenceQuadrilateral())
    nnodes = (6 + 1) * (5 + 1)
    values = (0.55 .+ 0.35 .* rand(rng, nnodes),
              0.08 .* randn(rng, nnodes),
              0.08 .* randn(rng, nnodes),
              0.04 .* randn(rng, nnodes))
    @test any(==(CONSTRAINT_DENSE), plan.assemble.constraint_kinds)
    initial_kinds = plan.assemble.constraint_kinds
    initial_perms = plan.assemble.constraint_perms
    compare_assembly!(plan, values, (); label="refresh initial Q2 6x5")

    # Structural refresh: move the level set, reclassify, compare again. The
    # structural rebuild must hand back fresh classification storage.
    moved = build_cut_reference(6, 5; radius=0.30, center=(0.55, 0.47),
                                model=reference.model)
    moved_roots = aggregate(AggregateAllCutCells(), moved.cut_geometry)
    refresh_geometry!(plan, moved.cut_geometry, moved_roots)
    @test plan.assemble.constraint_kinds !== initial_kinds
    @test plan.assemble.constraint_perms !== initial_perms
    @test any(==(CONSTRAINT_DENSE), plan.assemble.constraint_kinds)
    compare_assembly!(plan, values, (); label="refresh structural Q2 6x5")

    # Neutral refresh preserves classification storage and assembly values.
    kinds = plan.assemble.constraint_kinds
    perms = plan.assemble.constraint_perms
    refresh_geometry!(plan, moved.cut_geometry, moved_roots)
    @test plan.assemble.constraint_kinds === kinds
    @test plan.assemble.constraint_perms === perms
    compare_assembly!(plan, values, (); label="refresh neutral Q2 6x5")
end

@testset "SA4 warmed transform allocations" begin
    # Correctness-related evidence only. The dense kernel must not allocate per
    # call once warmed. The scalar-reference kernel on identical inputs is the
    # comparison point. No timing benchmarks are run here.
    rng = MersenneTwister(20261013)
    for (m, n) in ((8, 8), (48, 18), (18, 48))
        C = randn(rng, m, n)
        K = randn(rng, n, n)
        f = randn(rng, n)
        production = ThreadBuffer(copy(K), copy(f), zeros(m, n), zeros(m, m), zeros(m))
        reference = ThreadBuffer(copy(K), copy(f), zeros(m, n), zeros(m, m), zeros(m))
        _transform_agfem!(production, C, CONSTRAINT_DENSE, Int32[])
        scalar_transform!(reference, C)
        production_bytes = @allocated _transform_agfem!(
            production, C, CONSTRAINT_DENSE, Int32[])
        reference_bytes = @allocated scalar_transform!(reference, C)
        println("SA4 warmed dense transform allocation (m=$m, n=$n): " *
                "production=$production_bytes bytes, " *
                "scalar_reference=$reference_bytes bytes, " *
                "julia_threads=$(Threads.nthreads())")
        push!(SA4_ALLOCATIONS, (; label="(m=$m, n=$n)",
                                production_bytes, reference_bytes,
                                julia_threads=Threads.nthreads()))
        # Standard-library mul! on views and the scalar loops should allocate
        # nothing after warmup for Float64 matrices on this runtime. Gate on
        # zero; if a platform ever forces a tiny allocation, record it here
        # rather than silently widening the bound.
        @test production_bytes == 0
        @test reference_bytes == 0
    end
end

# ---------------------------------------------------------------------------
# Recorded absolute differences.
# ---------------------------------------------------------------------------

println("\n===== SA4 recorded absolute differences (observed / tolerance) =====")
for record in SA4_RECORDS
    println(rpad(record.label, 56),
            " left=", record.left_error, " / ", record.left_tol,
            "  matrix=", record.matrix_error, " / ", record.matrix_tol,
            "  vector=", record.vector_error, " / ", record.vector_tol)
end

println("\n===== SA4 warmed allocations (bytes) =====")
for allocation in SA4_ALLOCATIONS
    println(rpad(allocation.label, 16),
            " production=", allocation.production_bytes,
            " scalar_reference=", allocation.reference_bytes,
            " julia_threads=", allocation.julia_threads)
end
