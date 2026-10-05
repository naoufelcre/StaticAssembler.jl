# Focused tests for the SA1 identity/permutation fast path in the AgFEM
# transform. They exercise the classifier, the fast path against the dense
# `C·K·Cᵀ` reference, and real refreshed `StaticSystem` assemblies (fast cache
# vs forced dense cache). Runnable standalone from the repository root:
#
#     julia --project=@v1.11 test/TestTransformFastPath.jl
#
# When wired into `test/runtests.jl` the oracle include below is skipped
# because the oracle module is already loaded by the caller.

using Test
using LinearAlgebra
using Random
using GridapEmbedded
using StaticAssembler
using StaticAssembler: ThreadBuffer, _transform_agfem!, _classify_constraint,
    _classify_constraints!, CONSTRAINT_DENSE, CONSTRAINT_PERMUTATION,
    CONSTRAINT_IDENTITY

if !isdefined(Main, :StaticAssemblerOracle)
    include(joinpath(@__DIR__, "..", "oracle", "StaticAssemblerOracle.jl"))
end
using .StaticAssemblerOracle

# Run the transform through a fresh thread buffer and return copies of the
# effective matrix/vector. `perm` is only read on the permutation branch.
function run_transform(C, K, f, kind, perm)
    nt, nraw = size(C)
    scratch = ThreadBuffer(copy(K), copy(f), zeros(nt, nraw),
                           zeros(nt, nt), zeros(nt))
    _transform_agfem!(scratch, C, kind, perm)
    return copy(scratch.effective_matrix), copy(scratch.effective_vector)
end

@testset "SA1 constraint classification" begin
    # Identity is the special permutation with no stored index vector.
    kind, perm = _classify_constraint(Matrix{Float64}(I, 4, 4))
    @test kind == CONSTRAINT_IDENTITY
    @test isempty(perm)

    # Nontrivial permutation: one unit entry per row and column.
    C = zeros(4, 4)
    C[1, 2] = 1.0; C[2, 1] = 1.0; C[3, 3] = 1.0; C[4, 4] = 1.0
    kind, perm = _classify_constraint(C)
    @test kind == CONSTRAINT_PERMUTATION
    @test perm == Int32[2, 1, 3, 4]

    # Rectangular constraints cannot be a permutation.
    C = zeros(3, 4)
    C[1, 1] = 1.0; C[2, 2] = 1.0; C[3, 3] = 1.0
    @test _classify_constraint(C)[1] == CONSTRAINT_DENSE
    @test _classify_constraint(zeros(4, 3))[1] == CONSTRAINT_DENSE

    # Dense weights, duplicate columns, multiple entries per row, zero rows,
    # and non-unit entries must all stay on the dense fallback.
    @test _classify_constraint([1.0 0.5; 0.0 1.0])[1] == CONSTRAINT_DENSE
    @test _classify_constraint([1.0 0.0; 1.0 0.0])[1] == CONSTRAINT_DENSE
    @test _classify_constraint([1.0 1.0; 0.0 1.0])[1] == CONSTRAINT_DENSE
    @test _classify_constraint([1.0 0.0; 0.0 0.0])[1] == CONSTRAINT_DENSE
    @test _classify_constraint([0.5 0.0; 0.0 1.0])[1] == CONSTRAINT_DENSE

    # A near-permutation (tiny off-diagonal noise) is dense, not a permutation.
    near = Matrix{Float64}(I, 4, 4)
    near[1, 2] = 1e-14
    @test _classify_constraint(near)[1] == CONSTRAINT_DENSE

    # Batch classification fills both output vectors cell by cell.
    cells = Any[Matrix{Float64}(I, 2, 2), [1.0 0.0; 1.0 0.0], zeros(0, 0)]
    kinds = Vector{UInt8}(undef, 3)
    perms = Vector{Int32}[]
    resize!(perms, 3)
    _classify_constraints!(kinds, perms, cells)
    @test kinds == UInt8[CONSTRAINT_IDENTITY, CONSTRAINT_DENSE, CONSTRAINT_IDENTITY]
end

@testset "SA1 fast path equals C·K·Cᵀ and C·f" begin
    rng = MersenneTwister(20261003)
    for n in (1, 2, 4, 9)
        K = randn(rng, n, n); K = K + transpose(K)
        f = randn(rng, n)

        # Identity branch copies the raw block.
        Cid = Matrix{Float64}(I, n, n)
        E, v = run_transform(Cid, K, f, CONSTRAINT_IDENTITY, Int32[])
        @test E == K
        @test v == f
        @test E == Cid * K * transpose(Cid)
        @test v == Cid * f

        # Nontrivial permutation branch (also covers the vector nc=2 index
        # layout: a permutation matrix does not depend on component semantics).
        p = randperm(rng, n)
        perm = Int32.(p)
        Cp = zeros(n, n)
        for row in 1:n
            Cp[row, perm[row]] = 1.0
        end
        kind, stored = _classify_constraint(Cp)
        expected_kind = p == collect(1:n) ? CONSTRAINT_IDENTITY : CONSTRAINT_PERMUTATION
        @test kind == expected_kind
        E, v = run_transform(Cp, K, f, kind, stored)
        @test E == Cp * K * transpose(Cp)
        @test v == Cp * f
    end

    # The permutation rule does not assume a symmetric bilinear form.
    C = [0.0 0.0 1.0; 1.0 0.0 0.0; 0.0 1.0 0.0]
    K = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    f = [2.0, -3.0, 5.0]
    kind, perm = _classify_constraint(C)
    E, v = run_transform(C, K, f, kind, perm)
    @test E == C * K * transpose(C)
    @test v == C * f

    # Generic dense and rectangular constraints still use the original loops.
    for (nt, nraw) in ((3, 4), (4, 3), (4, 4))
        C = randn(rng, nt, nraw)
        K = randn(rng, nraw, nraw); K = K + transpose(K)
        f = randn(rng, nraw)
        kind, perm = _classify_constraint(C)
        @test kind == CONSTRAINT_DENSE
        E, v = run_transform(C, K, f, kind, perm)
        @test E ≈ C * K * transpose(C)
        @test v ≈ C * f
    end

    # A near-permutation must be projected with the dense reference, not the
    # reindex fast path.
    near = Matrix{Float64}(I, 4, 4)
    near[1, 2] = 1e-14
    K = randn(rng, 4, 4); K = K + transpose(K)
    f = randn(rng, 4)
    kind, perm = _classify_constraint(near)
    @test kind == CONSTRAINT_DENSE
    E, v = run_transform(near, K, f, kind, perm)
    @test E ≈ near * K * transpose(near)
    @test v ≈ near * f
end

@testset "SA1 block and mixed-element permutations" begin
    rng = MersenneTwister(20261004)
    n1, n2 = 4, 9                       # Q1 block, Q2 block
    p2 = Int32[3, 1, 4, 2, 9, 5, 7, 6, 8]
    Cmix = zeros(Float64, n1 + n2, n1 + n2)
    for i in 1:n1
        Cmix[i, i] = 1.0
    end
    for i in 1:n2
        Cmix[n1 + i, n1 + p2[i]] = 1.0
    end
    kind, perm = _classify_constraint(Cmix)
    @test kind == CONSTRAINT_PERMUTATION
    @test perm == Int32.(vcat(collect(1:n1), n1 .+ p2))
    K = randn(rng, n1 + n2, n1 + n2); K = K + transpose(K)
    f = randn(rng, n1 + n2)
    E, v = run_transform(Cmix, K, f, kind, perm)
    @test E ≈ Cmix * K * transpose(Cmix)
    @test v ≈ Cmix * f
end

@testset "SA1 refreshed systems: fast cache vs dense cache" begin
    function check_system(plan, values, nx, ny)
        a = plan.assemble
        @test length(a.constraint_kinds) == length(a.constraint_perms) ==
              length(a.test_constraints) == nx * ny
        # Stored classification agrees with a fresh classification.
        for cell in 1:(nx * ny)
            kind, perm = _classify_constraint(a.test_constraints[cell])
            @test kind == a.constraint_kinds[cell]
            @test perm == a.constraint_perms[cell]
        end
        Bfast, bfast = assemble!(plan, values; threaded=false)
        fastB = copy(Bfast); fastb = copy(bfast)
        Bfast_th, bfast_th = assemble!(plan, values; threaded=true)
        fastB_th = copy(Bfast_th); fastb_th = copy(bfast_th)
        saved = copy(a.constraint_kinds)
        a.constraint_kinds .= CONSTRAINT_DENSE
        try
            Bdense, bdense = assemble!(plan, values; threaded=false)
            @test fastB == Bdense
            @test fastb == bdense
            # Threaded accumulation follows the color order, not the cell
            # order, so compare threaded fast vs threaded dense (same order).
            Bdense_th, bdense_th = assemble!(plan, values; threaded=true)
            @test fastB_th == Bdense_th
            @test fastb_th == bdense_th
        finally
            a.constraint_kinds .= saved
        end
        return fastB, fastb
    end

    nx, ny = 6, 5
    reference = build_cut_reference(nx, ny; radius=0.34, center=(0.52, 0.48))
    parameters = KelvinVoigtParameters{Float64}()
    roots = aggregate(AggregateAllCutCells(), reference.cut_geometry)
    form = kelvin_voigt_form(parameters)
    plan = StaticSystem(reference.model, reference.cut_geometry, roots, nx, ny; form)
    @test any(==(StaticAssembler.OUT), plan.refresh.cell_states)
    @test any(==(StaticAssembler.CUT), plan.refresh.cell_states)
    @test any(k -> k == CONSTRAINT_IDENTITY || k == CONSTRAINT_PERMUTATION,
              plan.assemble.constraint_kinds)

    rng = MersenneTwister(20261005)
    nnodes = (nx + 1) * (ny + 1)
    values = (0.55 .+ 0.35 .* rand(rng, nnodes),
              0.08 .* randn(rng, nnodes), 0.08 .* randn(rng, nnodes),
              0.04 .* randn(rng, nnodes))
    check_system(plan, values, nx, ny)

    # Structural refresh after moving the level set reclassifies the tables.
    moved = build_cut_reference(nx, ny; radius=0.30, center=(0.55, 0.47),
                                model=reference.model)
    moved_roots = aggregate(AggregateAllCutCells(), moved.cut_geometry)
    refresh_geometry!(plan, moved.cut_geometry, moved_roots)
    check_system(plan, values, nx, ny)

    # A neutral refresh preserves classification storage and assembly values.
    kinds = plan.assemble.constraint_kinds
    perms = plan.assemble.constraint_perms
    B, b = assemble!(plan, values; threaded=false)
    saved_B, saved_b = copy(B), copy(b)
    refresh_geometry!(plan, moved.cut_geometry, moved_roots)
    @test plan.assemble.constraint_kinds === kinds
    @test plan.assemble.constraint_perms === perms
    B, b = assemble!(plan, values; threaded=false)
    @test B == saved_B
    @test b == saved_b
end

@testset "SA1 uncut self-rooted cells are all fast" begin
    nx, ny = 5, 4
    ncells = nx * ny
    interior = build_cut_reference(nx, ny;
                                   levelset_values=fill(-1.0, (nx + 1) * (ny + 1)))
    roots = collect(Int32, 1:ncells)          # every cell self-rooted
    parameters = KelvinVoigtParameters{Float64}()

    q1 = StaticSystem(interior.model, interior.cut_geometry, roots, nx, ny;
                      form=kelvin_voigt_form(parameters))
    @test all(==(CONSTRAINT_PERMUTATION), q1.assemble.constraint_kinds)
    kind, perm = _classify_constraint(q1.assemble.test_constraints[1])
    @test kind == CONSTRAINT_PERMUTATION
    @test sort(perm) == Int32.(1:8)     # Q1 vector: 4 nodes x 2 components

    q2 = StaticSystem(interior.model, interior.cut_geometry, roots, nx, ny;
                      form=kelvin_voigt_form(
                          parameters; element=StaticAssembler.Q2ReferenceQuadrilateral()))
    @test all(==(CONSTRAINT_PERMUTATION), q2.assemble.constraint_kinds)
    kind, perm = _classify_constraint(q2.assemble.test_constraints[1])
    @test kind == CONSTRAINT_PERMUTATION
    @test sort(perm) == Int32.(1:18)    # Q2 vector: 9 nodes x 2 components

    # Scalar Q1 is the true identity case (nc=1, raw and master orders agree).
    scalar_form = StaticAssembler.ElementVariationalForm(
        StaticAssembler.FiniteElement(StaticAssembler.Q1ReferenceQuadrilateral(), 1),
        (StaticAssembler.BilinearSlot(identity_op(), identity_op(),
                                      StaticAssembler.ConstantCoefficient(1.0)),),
        (), 0)
    scalar = StaticSystem(interior.model, interior.cut_geometry, roots, nx, ny;
                          form=scalar_form)
    @test all(==(CONSTRAINT_IDENTITY), scalar.assemble.constraint_kinds)

    # Fast cached assembly matches a dense forced cache on the uncut system.
    rng = MersenneTwister(20261006)
    nnodes = (nx + 1) * (ny + 1)
    values = (0.55 .+ 0.35 .* rand(rng, nnodes),
              0.08 .* randn(rng, nnodes), 0.08 .* randn(rng, nnodes),
              0.04 .* randn(rng, nnodes))
    Bfast, bfast = assemble!(q2, values; threaded=false)
    fastB = copy(Bfast); fastb = copy(bfast)
    saved = copy(q2.assemble.constraint_kinds)
    q2.assemble.constraint_kinds .= CONSTRAINT_DENSE
    try
        Bdense, bdense = assemble!(q2, values; threaded=false)
        @test fastB == Bdense
        @test fastb == bdense
    finally
        q2.assemble.constraint_kinds .= saved
    end
end
