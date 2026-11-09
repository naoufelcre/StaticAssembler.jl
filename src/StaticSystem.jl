# # Master-only projected AgFEM system: E and the solve operator B = EᵀAE.
#
# `StaticSystem` combines a static assembly layout (fixed quadrature tables,
# element tables, cached scatter slots — rebuilt only on a geometry refresh)
# with the AgFEM projection. The public math objects are:
#
#   E   : n_bg × n_m   extension operator (prolongation), a real sparse matrix
#   B   : n_m × n_m    projected solve operator = EᵀAE (SPD for coercive forms)
#   b_m : n_m          projected right-hand side = Eᵀb
#
# `assemble!` returns `(B, b_m)`. The solution lives on the n_m master
# (stable) DOFs; `E = extension(system)` maps it back to the full
# background lattice with one matmul (`u_full = E * u_m`, allocation-free via
# `mul!`; `restrict_solution!` extracts its master rows). `B` is a real sparse matrix, so
# `cholesky(B)` or `Preconditioners.jl` work directly.
#
# Systems are single-owner and non-reentrant: `B`, `b_m`, and the coefficient
# buffers are overwritten by the next `assemble!`, and `refresh_geometry!`
# rebuilds the extension tables and B's sparse pattern when the roots move.

"""
    StaticSystem(mesh, cutdata, aggregate_roots;
                 form, interior_quadrature=nothing, cut_quadrature=nothing,
                 boundary_quadrature=nothing)

Master-only projected AgFEM system. `mesh` and `cutdata` are the
background triangulation and cut-triangulation descriptors (see
Descriptors.jl) and `aggregate_roots` one root cell per background cell.
`form` is a construction-only input: the element variational form whose
element tables are generated at construction (see Forms.jl); only the
generated tables (finite elements and basis-at-quadrature data) are retained,
never the physics. Requires identical test and trial finite elements
(possibly mixed, as tuples of `FiniteElement` blocks). Accepts
aggregate-root changes on `refresh_geometry!`.

`interior_quadrature` supplies `(r,s)` points on `[0,1]²` and normalized
weights; `cut_quadrature` supplies barycentric triangle points and normalized
weights; `boundary_quadrature` supplies points on `[0,1]` and normalized
weights for each embedded segment. Each accepts a `(points, weights)` pair or an object exposing
`points` and `weights`. `nothing` selects the dominant element's conservative
default. Rules are validated and copied once; geometry refresh only remaps the
cached cut rule. The defaults integrate test/trial basis products multiplied by
a Q1 coefficient, upgraded to the Q2 rule when any coefficient field uses Q2;
nonlinear `StateFunction`s may require higher-order rules.
"""
struct RefreshCache{T,LK}
    element_tables::LK
    quadrature::CutQuadrature{T}
    quadrature_buffer::CutQuadratureStaging{T}
    boundary_quadrature::BoundaryQuadrature{T}
    boundary_quadrature_buffer::BoundaryQuadratureStaging{T}
    cell_states::Vector{Int8}
    last_roots::Vector{Int32}
    E::SparseMatrixCSC{T,Int}
    master_rows::Vector{Int}
    coefficient_prolongation::SparseMatrixCSC{T,Int}
    coefficient_nodes::Matrix{Int32}
    q2_coefficient_prolongation::SparseMatrixCSC{T,Int}
    q2_coefficient_nodes::Matrix{Int32}
end

struct AssembleCache{T,N,MT}
    B::MT
    b_m::Vector{T}
    test_cell_dofs::Vector{Vector{Int32}}
    test_constraints::Vector{Matrix{T}}
    scatter_slots::Vector{Vector{Int}}
    colors::Vector{Vector{Int32}}
    thread_buffers::Vector{ThreadBuffer{T}}
    extended_coefficients::NTuple{N,Vector{T}}
    # Per-cell constraint kind (dense / permutation / identity) and, for a
    # nontrivial permutation, the raw-local index of each effective row.
    constraint_kinds::Vector{UInt8}
    constraint_perms::Vector{Vector{Int32}}
end

mutable struct StaticSystem{T,N,LK,MT}
    nx::Int
    ny::Int
    hx::T
    hy::T
    mesh_id::Any
    interior_quadrature::_ReferenceQuadrature{2,T}
    cut_quadrature::_ReferenceQuadrature{3,T}
    boundary_rule::_ReferenceQuadrature{1,T}
    refresh::RefreshCache{T,LK}
    assemble::AssembleCache{T,N,MT}
end

function _validate_projected_aggregates(bg_states, aggregate_roots, ncells::Int)
    length(bg_states) == ncells ||
        throw(ArgumentError("level-set cell count does not match the background mesh"))
    length(aggregate_roots) == ncells ||
        throw(ArgumentError("one aggregate root is required per background cell"))
    for cell in 1:ncells
        state = bg_states[cell]
        _valid_bg_state(state) || throw(ArgumentError("invalid background-cell state"))
        state == OUT && continue
        root = Int(aggregate_roots[cell])
        1 <= root <= ncells ||
            throw(ArgumentError("active background cell $cell has no valid aggregate root"))
        bg_states[root] != OUT ||
            throw(ArgumentError("aggregate root $root is outside the active domain"))
        Int(aggregate_roots[root]) == root ||
            throw(ArgumentError("aggregate root $root is not self-rooted"))
    end
    return nothing
end

@inline function _projection_structure_changed(old_states, old_roots,
                                               new_states, new_roots)
    @inbounds for cell in eachindex(old_states)
        old_active = old_states[cell] != OUT
        new_active = new_states[cell] != OUT
        old_active == new_active || return true
        new_active && old_roots[cell] != new_roots[cell] && return true
    end
    return false
end

# Extension weights: the slave node's value is the root cell's basis
# interpolated at the slave node's position. On the structured background grid
# the slave node's reference position within the root cell is its lattice
# offset (integer for Q1, half-integer for Q2), so the weights are exactly
# `shape(el, offset)` — the same values the Gridap basis evaluation produced
# (see the projected-operator equivalence tests), without requiring Gridap at
# runtime.
@inline function _extension_weights(el::ReferenceFiniteElement,
                                    nx::Int, node::Int, root_cell::Int)
    nodes_x = node_grid(el, nx)
    i = (node - 1) % nodes_x + 1
    j = (node - 1) ÷ nodes_x + 1
    i_root = (root_cell - 1) % nx + 1
    j_root = (root_cell - 1) ÷ nx + 1
    scale = lattice_scale(el)
    r = scale * (i - 1) - (i_root - 1)
    s = scale * (j - 1) - (j_root - 1)
    return shape(el, r, s)
end

# Factored E rows for one lattice, used transiently by a structural refresh:
# `count[node]` nonzeros, `nodes[:, node]` support targets, `weights[:, node]`
# values. Master rows are unit rows; inactive rows are empty. `is_master` and
# `representative` are scratch for the scan. Nothing here persists past the
# refresh that materializes E and the per-cell C tables.
function _extension_row_supports!(count::Vector{UInt8}, nodes::Matrix{Int32},
                                  weights::Matrix{T}, is_master::BitVector,
                                  representative::Vector{Int32},
                                  el::ReferenceFiniteElement,
                                  bg_states, aggregate_roots, nx::Int,
                                  ny::Int) where {T}
    nnodes = length(count)
    nnode_el = nnode(el)
    fill!(is_master, false)
    fill!(representative, 0)
    fill!(count, 0)
    fill!(nodes, 0)
    fill!(weights, zero(T))

    # ponytail: scan the fixed mesh; switch to dirty-cell updates only if this is measurable.
    @inbounds for cell in 1:(nx * ny)
        bg_states[cell] == OUT && continue
        is_root = Int(aggregate_roots[cell]) == cell
        for local_node in 1:nnode_el
            node = Int(cell_nodes(el, cell, nx)[local_node])
            representative[node] = Int32(cell)
            is_master[node] |= is_root
        end
    end

    for node in 1:nnodes
        if is_master[node]
            count[node] = 1
            nodes[1, node] = Int32(node)
            weights[1, node] = one(T)
            continue
        end
        representative_cell = Int(representative[node])
        representative_cell == 0 && continue
        root = Int(aggregate_roots[representative_cell])
        extension = _extension_weights(el, nx, node, root)
        count[node] = UInt8(nnode_el)
        @inbounds for root_node in 1:nnode_el
            target = Int(cell_nodes(el, root, nx)[root_node])
            is_master[target] ||
                error("AgFEM extension targets a non-master background node")
            weight = T(extension[root_node])
            isfinite(weight) || throw(ArgumentError("AgFEM extension contains a non-finite weight"))
            nodes[root_node, node] = target
            weights[root_node, node] = weight
        end
    end
    return is_master
end

# Structural refresh core: build transient factored E rows per block, number
# the compact master DOFs block-major, fill the per-cell C tables in place,
# and materialize E (solution prolongation) and the Q1/Q2 coefficient
# prolongations as sparse matrices. Same math as the previous stored tables;
# only the storage changed.
function _coefficient_extension(::Type{T}, element::ReferenceFiniteElement,
                                bg_states, aggregate_roots,
                                nx::Int, ny::Int) where {T}
    nnodes = node_grid(element, nx) * node_grid(element, ny)
    count = zeros(UInt8, nnodes)
    nodes = zeros(Int32, nnode(element), nnodes)
    weights = zeros(T, nnode(element), nnodes)
    _extension_row_supports!(count, nodes, weights, falses(nnodes),
                             zeros(Int32, nnodes), element,
                             bg_states, aggregate_roots, nx, ny)
    I = Int[]
    J = Int[]
    V = T[]
    for node in 1:nnodes, support in 1:Int(count[node])
        push!(I, node)
        push!(J, Int(nodes[support, node]))
        push!(V, weights[support, node])
    end
    return sparse(I, J, V, nnodes, nnodes)
end

function _refresh_static_tables!(::Type{T}, cell_dofs, constraints, spaces,
                                  coefficient_elements,
                                  bg_states, aggregate_roots, nx::Int,
                                  ny::Int) where {T}
    nblocks = length(spaces)
    ncells = nx * ny
    rows = map(spaces) do space
        el = space.element
        nnodes = node_grid(el, nx) * node_grid(el, ny)
        nnode_el = nnode(el)
        count = zeros(UInt8, nnodes)
        nodes = zeros(Int32, nnode_el, nnodes)
        weights = zeros(T, nnode_el, nnodes)
        node_compact = zeros(Int32, nnodes)
        is_master = falses(nnodes)
        representative = zeros(Int32, nnodes)
        _extension_row_supports!(count, nodes, weights, is_master, representative,
                                 el, bg_states, aggregate_roots, nx, ny)
        (; count, nodes, weights, node_compact, is_master, el,
           nc=ncomp(space), nnode_el)
    end
    n_bg = sum(r.nc * length(r.count) for r in rows)

    I = Int[]
    J = Int[]
    V = T[]
    base = 1
    row_offset = 0
    for r in rows
        nnodes = length(r.count)
        rank = 0
        @inbounds for node in eachindex(r.is_master)
            if r.is_master[node]
                rank += 1
                r.node_compact[node] = Int32(base + (rank - 1) * r.nc)
            end
        end
        base += rank * r.nc
        @inbounds for node in 1:nnodes, comp in 1:r.nc
            row = row_offset + r.nc * (node - 1) + comp
            for support in 1:Int(r.count[node])
                push!(I, row)
                push!(J, Int(r.node_compact[Int(r.nodes[support, node])]) + comp - 1)
                push!(V, r.weights[support, node])
            end
        end
        row_offset += r.nc * nnodes
    end
    n_m = base - 1
    master_rows = Vector{Int}(undef, n_m)
    row_offset = 0
    for r in rows
        nnodes = length(r.count)
        @inbounds for node in 1:nnodes
            r.is_master[node] || continue
            for comp in 1:r.nc
                master_rows[Int(r.node_compact[node]) + comp - 1] =
                    row_offset + r.nc * (node - 1) + comp
            end
        end
        row_offset += r.nc * nnodes
    end
    E = sparse(I, J, V, n_bg, n_m)

    P = _coefficient_extension(
        T, Q1ReferenceQuadrilateral(), bg_states, aggregate_roots, nx, ny)
    P2 = any(element -> element isa Q2ReferenceQuadrilateral,
             coefficient_elements) ?
        _coefficient_extension(
            T, Q2ReferenceQuadrilateral(), bg_states, aggregate_roots, nx, ny) :
        spzeros(T, 0, 0)

    # Per-cell effective (master) DOFs and raw→effective constraint matrices.
    # `C[eff_local, raw_local]` is the weight with which raw local DOF
    # `raw_local` enters effective master `eff_local`; the transform `C·K·Cᵀ`
    # projects one cell's raw block onto the master DOFs.
    old_length = length(cell_dofs)
    resize!(cell_dofs, ncells)
    resize!(constraints, ncells)
    for cell in (old_length + 1):ncells
        cell_dofs[cell] = Int32[]
        constraints[cell] = Matrix{T}(undef, 0, 0)
    end
    masters = Int32[]
    for cell in 1:ncells
        eff_dofs = cell_dofs[cell]
        empty!(eff_dofs)
        if bg_states[cell] == OUT
            isempty(constraints[cell]) ||
                (constraints[cell] = Matrix{T}(undef, 0, 0))
            continue
        end
        C = constraints[cell]
        for b in 1:nblocks
            r = rows[b]
            nnode_el = r.nnode_el
            nc = r.nc
            raw_nodes = cell_nodes(r.el, cell, nx)
            empty!(masters)
            @inbounds for lnode in 1:nnode_el
                node = raw_nodes[lnode]
                for support in 1:Int(r.count[node])
                    m = r.nodes[support, node]
                    m in masters || push!(masters, m)
                end
            end
            sort!(masters)
            neff = length(masters) * nc
            nraw = nnode_el * nc
            Cb = if nblocks == 1 && size(C) == (neff, nraw)
                fill!(C, zero(T))
            else
                zeros(T, neff, nraw)
            end
            @inbounds for (rank, m) in enumerate(masters)
                base = r.node_compact[m]
                for comp in 1:nc
                    push!(eff_dofs, base + comp - 1)
                end
            end
            @inbounds for lnode in 1:nnode_el
                node = raw_nodes[lnode]
                for support in 1:Int(r.count[node])
                    m = r.nodes[support, node]
                    w = r.weights[support, node]
                    rank = searchsortedfirst(masters, m)
                    for comp in 1:nc
                        eff_local = (rank - 1) * nc + comp
                        raw_local = lnode + nnode_el * (comp - 1)
                        Cb[eff_local, raw_local] = w
                    end
                end
            end
            if b == 1
                C = Cb
            else
                Cnew = Matrix{T}(undef, size(C, 1) + neff, size(C, 2) + nraw)
                fill!(Cnew, zero(T))
                Cnew[1:size(C, 1), 1:size(C, 2)] .= C
                Cnew[size(C, 1)+1:end, size(C, 2)+1:end] .= Cb
                C = Cnew
            end
        end
        constraints[cell] = C
    end
    return E, master_rows, P, P2
end

# B sparse pattern from the per-cell effective DOF connectivity.
function _pattern_from_ragged_dofs!(matrix::SparseMatrixCSC{T,Int}, row_entries,
                                    test_dofs, trial_dofs,
                                    ntest::Int, ntrial::Int) where {T}
    while length(row_entries) < ntrial
        push!(row_entries, Int[])
    end
    for column in 1:ntrial
        empty!(row_entries[column])
    end
    for cell in eachindex(test_dofs)
        for column in trial_dofs[cell], row in test_dofs[cell]
            push!(row_entries[column], row)
        end
    end
    same_size = size(matrix) == (ntest, ntrial)
    colptr = matrix.colptr
    rowval = matrix.rowval
    nzval = matrix.nzval
    resize!(colptr, ntrial + 1)
    empty!(rowval)
    colptr[1] = 1
    for column in 1:ntrial
        rows = row_entries[column]
        sort!(rows)
        previous = 0
        for row in rows
            row == previous && continue
            push!(rowval, row)
            previous = row
        end
        colptr[column + 1] = length(rowval) + 1
    end
    resize!(nzval, length(rowval))
    fill!(nzval, zero(T))
    return same_size ? matrix : SparseMatrixCSC(ntest, ntrial, colptr, rowval, nzval)
end

"""
    refresh_geometry!(system::StaticSystem, cutdata, aggregate_roots)

Refresh quadrature, the AgFEM extension operator `E`, and B's sparse pattern
for a moved level set. The aggregate roots may change. Projection tables are
retained unless active-cell membership or an active root changes. Returns
`system`. A structural refresh invalidates matrices previously returned by
`assemble!`; reacquire them from the next assembly call.
"""
function refresh_geometry!(plan::StaticSystem{T}, cutdata::CutData{T},
                           aggregate_roots) where {T}
    cutdata.mesh_id === plan.mesh_id ||
        throw(ArgumentError("cut geometry belongs to a different background mesh"))
    ncells = plan.nx * plan.ny
    _validate_projected_aggregates(cutdata.bg_states, aggregate_roots, ncells)
    r = plan.refresh
    a = plan.assemble
    states_aliased = Base.mightalias(r.cell_states, cutdata.bg_states)
    roots_aliased = Base.mightalias(r.last_roots, aggregate_roots)
    bg_states = states_aliased ? copy(cutdata.bg_states) : cutdata.bg_states
    roots = roots_aliased ? collect(aggregate_roots) : aggregate_roots
    # Projection depends on active-vs-OUT membership and roots of active cells;
    # IN/CUT labels and roots stored on OUT cells only affect geometry data.
    structural = states_aliased || roots_aliased ||
                 _projection_structure_changed(
                     r.cell_states, r.last_roots, bg_states, roots)
    background_cells = Base.OneTo(ncells)
    spaces = r.element_tables.test_space
    npoints = _stage_cut_quadrature!(
        r.quadrature_buffer, background_cells, cutdata,
        plan.nx, plan.ny, plan.hx, plan.hy,
        plan.cut_quadrature)
    has_boundary = !isempty(r.element_tables.boundary_bilinear) ||
                   !isempty(r.element_tables.boundary_rhs)
    boundary_npoints = if has_boundary
        staged_boundary = _stage_boundary_quadrature!(
            r.boundary_quadrature_buffer, cutdata.boundary, bg_states,
            plan.nx, plan.ny, plan.hx, plan.hy, plan.boundary_rule)
        if r.boundary_quadrature_buffer.nfacets == 0 && any(==(CUT), bg_states)
            throw(ArgumentError("boundary-enabled form requires embedded boundary facets for cut cells"))
        end
        staged_boundary
    else
        fill!(r.boundary_quadrature_buffer.pointers, 1)
        r.boundary_quadrature_buffer.nfacets = 0
        0
    end
    # Structural rebuilds use fresh tables so a late failure throws before
    # any committed cache is touched; the swap below only runs on success.
    # ponytail: fresh B pattern per structural refresh; reuse it only if profiling blames this path.
    if structural
        new_cell_dofs = Vector{Vector{Int32}}()
        new_constraints = Vector{Matrix{T}}()
        E, master_rows, P, P2 = _refresh_static_tables!(
            T, new_cell_dofs, new_constraints, spaces,
            r.element_tables.coefficient_elements, bg_states, roots,
            plan.nx, plan.ny)
        constraint_kinds = Vector{UInt8}(undef, length(new_constraints))
        constraint_perms = Vector{Vector{Int32}}(undef, length(new_constraints))
        _classify_constraints!(constraint_kinds, constraint_perms, new_constraints)
        n_m = size(E, 2)
        pattern_rows = Vector{Vector{Int}}()
        B = _pattern_from_ragged_dofs!(
            spzeros(T, 0, 0), pattern_rows, new_cell_dofs, new_cell_dofs, n_m, n_m)
        slots = _ragged_scatter_slots!(
            Vector{Vector{Int}}(), B, new_cell_dofs, new_cell_dofs)
        colors = _effective_cell_colors(new_cell_dofs, new_cell_dofs)
        max_test = maximum(length, new_cell_dofs; init=0)
        ntest_raw = sum(ndofs(space) for space in spaces)
        buffers = [ThreadBuffer(
            zeros(T, ntest_raw, ntest_raw), zeros(T, ntest_raw),
            zeros(T, max_test, ntest_raw),
            zeros(T, max_test, max_test), zeros(T, max_test),
        ) for _ in 1:Threads.maxthreadid()]
        b_m = zeros(T, n_m)
        plan.refresh = RefreshCache(r.element_tables, r.quadrature, r.quadrature_buffer,
                                    r.boundary_quadrature,
                                    r.boundary_quadrature_buffer,
                                    r.cell_states, r.last_roots, E, master_rows, P,
                                    r.coefficient_nodes, P2,
                                    r.q2_coefficient_nodes)
        plan.assemble = AssembleCache(B, b_m, new_cell_dofs, new_constraints,
                                      slots, colors, buffers,
                                      a.extended_coefficients,
                                      constraint_kinds, constraint_perms)
        r = plan.refresh
        a = plan.assemble
    end
    _commit_quadrature!(
        r.quadrature, r.quadrature_buffer.pointers,
        r.quadrature_buffer.is_cut,
        r.quadrature_buffer.reference_r, r.quadrature_buffer.reference_s,
        r.quadrature_buffer.weight, npoints)
    _commit_boundary_quadrature!(
        r.boundary_quadrature, r.boundary_quadrature_buffer, boundary_npoints)
    @inbounds for cell in 1:ncells
        state = bg_states[cell]
        r.cell_states[cell] = state
        r.last_roots[cell] = state == OUT ? Int32(0) : Int32(roots[cell])
        r.quadrature.is_cut[cell] = state != IN
    end
    _regenerate_element_tables!(plan.refresh.element_tables, plan.hx, plan.hy,
                                 plan.refresh.quadrature, plan.nx * plan.ny,
                                 plan.interior_quadrature)
    _regenerate_boundary_tables!(plan.refresh.element_tables, plan.hx, plan.hy,
                                 plan.refresh.boundary_quadrature)
    return plan
end

# Validate the sparse scatter slots against the (possibly ragged) DOF maps.
function validate_sparse_slots(matrix::SparseMatrixCSC,
                               test_dofs::AbstractVector{<:AbstractVector},
                               trial_dofs::AbstractVector{<:AbstractVector},
                               slots::AbstractVector{<:AbstractVector{<:Integer}})
    length(test_dofs) == length(trial_dofs) == length(slots) ||
        throw(ArgumentError("cell connectivity and slot counts differ"))
    for cell in eachindex(test_dofs)
        length(slots[cell]) == length(test_dofs[cell]) * length(trial_dofs[cell]) ||
            throw(ArgumentError("invalid sparse slot count for cell $cell"))
        nt = length(test_dofs[cell])
        for (index, column) in enumerate(trial_dofs[cell])
            for (row_index, row) in enumerate(test_dofs[cell])
                slot = slots[cell][row_index + nt * (index - 1)]
                matrix.colptr[column] <= slot < matrix.colptr[column + 1] &&
                    matrix.rowval[slot] == row ||
                    throw(ArgumentError("invalid sparse slot for cell $cell"))
            end
        end
    end
    return true
end

function StaticSystem(
    mesh::Mesh{T}, cutdata::CutData{T}, aggregate_roots;
    form::ElementVariationalForm,
    interior_quadrature=nothing,
    cut_quadrature=nothing,
    boundary_quadrature=nothing,
) where {T<:AbstractFloat}
    T === Float64 || throw(ArgumentError("the projected path requires Float64"))
    form.ncoefficients >= 0 ||
        throw(ArgumentError("form.ncoefficients must be nonnegative"))
    form.nfacetfields >= 0 ||
        throw(ArgumentError("form.nfacetfields must be nonnegative"))
    form.test_space == form.trial_space ||
        throw(ArgumentError("the projected path requires identical test and trial spaces"))
    nx = mesh.nx
    ny = mesh.ny
    nx > 0 && ny > 0 || throw(ArgumentError("nx and ny must be positive"))
    cutdata.mesh_id === mesh.mesh_id ||
        throw(ArgumentError("cut geometry belongs to a different background mesh"))
    isfinite(mesh.hx) && isfinite(mesh.hy) && mesh.hx > 0 && mesh.hy > 0 ||
        throw(ArgumentError("background grid metrics must be finite and positive"))

    ncells = nx * ny
    element = _rule_element(form.test_space, form.trial_space,
                            form.coefficient_elements)
    interior_source = isnothing(interior_quadrature) ?
        interior_rule(element) : interior_quadrature
    cut_source = isnothing(cut_quadrature) ? cut_rule(element) : cut_quadrature
    boundary_source = isnothing(boundary_quadrature) ?
        boundary_rule(element) : boundary_quadrature
    interior = _canonical_quadrature(
        interior_source, Val(2), T, "interior quadrature")
    cut = _canonical_quadrature(cut_source, Val(3), T, "cut quadrature")
    boundary = _canonical_quadrature(
        boundary_source, Val(1), T, "boundary quadrature")
    element_tables = generate_element_tables(
        form, mesh.hx, mesh.hy, nothing, ncells, interior)
    quadrature = CutQuadrature(
        Vector{Int}(undef, ncells + 1), falses(ncells), T[], T[], T[])
    staging = CutQuadratureStaging(
        Vector{Int}(undef, ncells + 1), falses(ncells), T[], T[], T[])
    boundary_points = BoundaryQuadrature(
        Vector{Int}(undef, ncells + 1), Int32[], T[], T[], T[], T[], T[], 0)
    boundary_staging = BoundaryQuadratureStaging(
        Vector{Int}(undef, ncells + 1), Int32[], T[], T[], T[], T[], T[], 0)
    has_q2_coefficients = any(
        element -> element isa Q2ReferenceQuadrilateral,
        form.coefficient_elements)
    q2_coefficient_nodes = has_q2_coefficients ?
        _cell_nodes(Q2ReferenceQuadrilateral(), Int32.(1:ncells), nx) :
        Matrix{Int32}(undef, 0, 0)
    refresh = RefreshCache(
        element_tables, quadrature, staging, boundary_points, boundary_staging,
        zeros(Int8, ncells), zeros(Int32, ncells),
        spzeros(T, 0, 0), Int[], spzeros(T, 0, 0),
        _cell_nodes(Q1ReferenceQuadrilateral(), Int32.(1:ncells), nx),
        spzeros(T, 0, 0), q2_coefficient_nodes)
    assemble = AssembleCache(
        spzeros(T, 0, 0), T[],
        Vector{Vector{Int32}}(), Vector{Matrix{T}}(), Vector{Vector{Int}}(),
        Vector{Vector{Int32}}(), ThreadBuffer{T}[],
        ntuple(Val(Int(form.ncoefficients))) do field
            element = form.coefficient_elements[field]
            zeros(T, node_grid(element, nx) * node_grid(element, ny))
        end,
        UInt8[], Vector{Int32}[])
    plan = StaticSystem(
        nx, ny, mesh.hx, mesh.hy, mesh.mesh_id,
        interior, cut, boundary, refresh, assemble)
    return refresh_geometry!(plan, cutdata, aggregate_roots)
end

"""
    extension(system::StaticSystem)

Return the system-owned sparse operator mapping master DOFs to the background
lattice, without copying. Do not mutate it; reacquire it after a structural
`refresh_geometry!`, which may replace it and renumber master DOFs.
"""
extension(system::StaticSystem) = system.refresh.E

"""
    restrict_solution!(u_m, system, u_full)

Copy the master rows of a full-background solution into the compact projected
solution vector. This is the left inverse of extension for vectors in the
range of `extension(system)`.
"""
function restrict_solution!(u_m::AbstractVector, plan::StaticSystem,
                            u_full::AbstractVector)
    rows = plan.refresh.master_rows
    length(u_m) == length(rows) || throw(DimensionMismatch("invalid master solution length"))
    length(u_full) == size(plan.refresh.E, 1) ||
        throw(DimensionMismatch("invalid background solution length"))
    @inbounds for master in eachindex(rows)
        u_m[master] = u_full[rows[master]]
    end
    return u_m
end

@inline function _assemble_cell!(
    scratch::ThreadBuffer{T}, plan::StaticSystem{T}, cell::Int,
    coefficients::NTuple{N,AbstractVector},
    facet_fields::NTuple{M,AbstractVector},
) where {T,N,M}
    r = plan.refresh
    return _assemble_raw!(scratch.raw_matrix, scratch.raw_vector, r.element_tables,
                           r.quadrature, r.boundary_quadrature,
                           r.coefficient_nodes, r.q2_coefficient_nodes,
                           cell, coefficients, facet_fields)
end

@inline function _scatter_projected!(
    plan::StaticSystem, scratch::ThreadBuffer, cell::Int,
)
    a = plan.assemble
    test_dofs = a.test_cell_dofs[cell]
    nt = length(test_dofs)
    @inbounds for column in eachindex(test_dofs), row in eachindex(test_dofs)
        slot = a.scatter_slots[cell][row + nt * (column - 1)]
        a.B.nzval[slot] += scratch.effective_matrix[row, column]
    end
    @inbounds for row in eachindex(test_dofs)
        a.b_m[test_dofs[row]] += scratch.effective_vector[row]
    end
    return nothing
end

function assemble_serial!(plan::StaticSystem,
                          coefficients::NTuple{N,AbstractVector},
                          facet_fields::NTuple{M,AbstractVector}) where {N,M}
    a = plan.assemble
    r = plan.refresh
    fill!(a.B.nzval, zero(eltype(a.B)))
    fill!(a.b_m, zero(eltype(a.b_m)))
    scratch = a.thread_buffers[Threads.threadid()]
    @inbounds for cell in 1:(plan.nx * plan.ny)
        r.cell_states[cell] == OUT && continue
        _assemble_cell!(scratch, plan, cell, coefficients, facet_fields)
        _transform_agfem!(scratch, a.test_constraints[cell],
                          a.constraint_kinds[cell], a.constraint_perms[cell])
        _scatter_projected!(plan, scratch, cell)
    end
    return a.B, a.b_m
end

function assemble_threaded!(plan::StaticSystem,
                            coefficients::NTuple{N,AbstractVector},
                            facet_fields::NTuple{M,AbstractVector}) where {N,M}
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
            _transform_agfem!(scratch, a.test_constraints[cell],
                              a.constraint_kinds[cell], a.constraint_perms[cell])
            _scatter_projected!(plan, scratch, cell)
        end
    end
    return a.B, a.b_m
end

"""
    assemble!(system::StaticSystem, coefficients::NTuple{N,AbstractVector};
              facet_fields=(), threaded=true, extend_coefficients=true)

Assemble the projected operator `B = EᵀAE` (a real `n_m × n_m` sparse matrix)
and the projected right-hand side `b_m = Eᵀb` from background-node coefficient
arrays (one per element-table coefficient field). Every array holds one value
per background lattice node of that field's Q1/Q2 element; this node count is
validated in both modes. With `extend_coefficients=true` (default) each field
is prolonged to the full lattice with the stored coefficient prolongation
(the AgFEM root interpolation), so cut-cell nodes take their aggregate-root
values. With `extend_coefficients=false` the arrays are copied verbatim into
the assembly buffers, so each cut cell keeps its own raw nodal values. Use the
raw mode to feed a field that already lives on the full background lattice
(for example a previous Qm solution or an analytic field sampled at the
lattice nodes) without reprojecting it through the current aggregation.
Because the raw cut-cell values are the *current* cell polynomials, a previous
Qm solution on a shrinking domain must retain its old cut-cell polynomials for
the restriction (evaluating the old FE function, not re-interpolating the old
master values through the new roots) to be reproduced by this path.
`facet_fields` is a tuple of arrays in the current embedded boundary facet
order; these values are not prolonged.
Returns `(B, b_m)`, both owned by the system and overwritten by the next call
or invalidated by a structural `refresh_geometry!`.
Solve with `cholesky(B)` or any solver on `B`; `diag(B)` is the diagonal
preconditioner. Extend solutions with `extension(system) * u_m`.
"""
function assemble!(plan::StaticSystem,
                   coefficients::NTuple{N,AbstractVector};
                   facet_fields=(),
                   threaded::Bool=true,
                   extend_coefficients::Bool=true) where {N}
    a = plan.assemble
    r = plan.refresh
    length(coefficients) == length(a.extended_coefficients) ||
        throw(ArgumentError("element tables expect $(length(a.extended_coefficients)) coefficient fields, got $N"))
    facet_fields isa Tuple && all(values -> values isa AbstractVector, facet_fields) ||
        throw(ArgumentError("facet_fields must be a tuple of vectors"))
    expected_facet_fields = plan.refresh.element_tables.nfacetfields
    length(facet_fields) == expected_facet_fields ||
        throw(ArgumentError("element tables expect $expected_facet_fields facet fields, got $(length(facet_fields))"))
    nfacets = plan.refresh.boundary_quadrature.nfacets
    all(length(values) == nfacets for values in facet_fields) ||
        throw(DimensionMismatch("facet fields must contain one value per embedded boundary facet"))
    for field in eachindex(a.extended_coefficients)
        element = r.element_tables.coefficient_elements[field]
        P = element isa Q1ReferenceQuadrilateral ?
            r.coefficient_prolongation : r.q2_coefficient_prolongation
        length(coefficients[field]) == size(P, 1) ||
            throw(DimensionMismatch(
                "coefficient field $field has the wrong background-node count"))
        if extend_coefficients
            mul!(a.extended_coefficients[field], P, coefficients[field])
        else
            copyto!(a.extended_coefficients[field], coefficients[field])
        end
    end
    threaded ? assemble_threaded!(plan, a.extended_coefficients, facet_fields) :
               assemble_serial!(plan, a.extended_coefficients, facet_fields)
    return a.B, a.b_m
end
