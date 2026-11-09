# # Shared assembly kernels for the projected `StaticSystem`.
#
# Raw per-cell assembly, scatter-slot lookup, and the AgFEM `C·K·Cᵀ` transform.

struct ThreadBuffer{T}
    raw_matrix::Matrix{T}
    raw_vector::Vector{T}
    left_matrix::Matrix{T}
    effective_matrix::Matrix{T}
    effective_vector::Vector{T}
end

# Constraint classification for the AgFEM transform. The per-cell constraint
# `Cᵀ` maps master values to raw local values; `C` projects local loads.
# When `C` is an exact unit
# permutation, `C·K·Cᵀ` is only a reindex and the O(nt³) dense loops collapse
# to O(nt²); every uncut self-rooted cell has such a `C`.
const CONSTRAINT_DENSE = 0x00
const CONSTRAINT_PERMUTATION = 0x01
const CONSTRAINT_IDENTITY = 0x02

# Classify one constraint matrix once per structural refresh. The permutation
# fast path is a valid representation of `C` only when there is exactly one
# entry per row and per column and every entry is `one(T)`:
#     C[p, i] = δ(i, perm[p])   <=>  effective K[p, q] = raw K[perm[p], perm[q]].
# Rectangular, dense, weighted, or near-permutation tables return
# `CONSTRAINT_DENSE` and keep the original dense transform. Identity is the
# special permutation `perm[p] == p`, stored without a permutation vector.
function _classify_constraint(C::AbstractMatrix{T}) where {T}
    neff, nraw = size(C)
    neff == nraw || return (CONSTRAINT_DENSE, Int32[])
    n = neff
    n == 0 && return (CONSTRAINT_IDENTITY, Int32[])
    perm = Vector{Int32}(undef, n)
    seen = falses(n)
    identity = true
    @inbounds for row in 1:n
        column = 0
        for col in 1:n
            value = C[row, col]
            iszero(value) && continue
            value == one(T) || return (CONSTRAINT_DENSE, Int32[])
            column == 0 || return (CONSTRAINT_DENSE, Int32[])
            seen[col] && return (CONSTRAINT_DENSE, Int32[])
            seen[col] = true
            column = col
        end
        column == 0 && return (CONSTRAINT_DENSE, Int32[])
        perm[row] = column
        column == row || (identity = false)
    end
    return (identity ? CONSTRAINT_IDENTITY : CONSTRAINT_PERMUTATION,
            identity ? Int32[] : perm)
end

function _classify_constraints!(kinds::AbstractVector{UInt8},
                                perms::AbstractVector,
                                constraints)
    @inbounds for cell in eachindex(constraints)
        kinds[cell], perms[cell] = _classify_constraint(constraints[cell])
    end
    return nothing
end

@inline function _nz_position(matrix::SparseMatrixCSC, row::Integer, column::Integer)
    lo = Int(matrix.colptr[column])
    hi = Int(matrix.colptr[column + 1] - 1)
    position = searchsortedfirst(matrix.rowval, row, lo, hi, Base.Order.Forward)
    position <= hi && matrix.rowval[position] == row ||
        error("sparse pattern is missing entry ($row, $column)")
    return position
end

function _ragged_scatter_slots!(slots, matrix, test_dofs, trial_dofs)
    old_length = length(slots)
    resize!(slots, length(test_dofs))
    for cell in (old_length + 1):length(slots)
        slots[cell] = Int[]
    end
    for cell in eachindex(test_dofs)
        cell_slots = slots[cell]
        resize!(cell_slots, length(test_dofs[cell]) * length(trial_dofs[cell]))
        index = 0
        @inbounds for column in trial_dofs[cell], row in test_dofs[cell]
            index += 1
            cell_slots[index] = _nz_position(matrix, row, column)
        end
    end
    return slots
end

# Raw per-cell assembly core: fill element stiffness `Ke` / element load `fe`
# (Ciarlet `A^K` / `F^K`) from the element tables and the cut quadrature of
# `cell`. Shared by the projected `StaticSystem` (which reuses the raw block
# before projecting it).
@inline function _assemble_raw!(Ke, fe, kd, quadrature, boundary_quadrature,
                                 coefficient_nodes, q2_coefficient_nodes, cell::Int,
                                 coefficients::NTuple{N,AbstractVector},
                                 facet_fields::NTuple{M,AbstractVector}) where {N,M}
    T = eltype(Ke)
    fill!(Ke, zero(T))
    fill!(fe, zero(T))
    point_context = kd.coefficient_gradients
    has_q2 = _has_q2_elements(typeof(kd.coefficient_elements))
    if quadrature.pointers[cell] == quadrature.pointers[cell + 1]
        if !quadrature.is_cut[cell]
            copyto!(Ke, kd.const_interior)
            @inbounds for q in eachindex(kd.interior_weight)
                sh_t, gx_t, gy_t, sh_c, gx_c, gy_c = _point_bases(
                    kd.interior_shape_t, kd.interior_gx_t, kd.interior_gy_t,
                    kd.interior_shape_c, kd.interior_gx_c, kd.interior_gy_c,
                    q, kd.test_space, kd.trial_space)
                csh = _packed4(kd.interior_coefficient_shape, q)
                q2sh = _maybe_q2_basis(
                    Val(has_q2), kd.interior_q2_coefficient_shape, q)
                c = if point_context
                    q2gx = _maybe_q2_basis(
                        Val(has_q2), kd.interior_q2_coefficient_gx, q)
                    q2gy = _maybe_q2_basis(
                        Val(has_q2), kd.interior_q2_coefficient_gy, q)
                    _coefficient_point(
                        csh, _packed4(kd.interior_coefficient_gx, q),
                        _packed4(kd.interior_coefficient_gy, q),
                        q2sh, q2gx, q2gy, coefficient_nodes, q2_coefficient_nodes,
                        cell, coefficients, kd.coefficient_elements)
                else
                    _coefficient_values(
                        csh, q2sh, coefficient_nodes, q2_coefficient_nodes,
                        cell, coefficients, kd.coefficient_elements)
                end
                _accumulate_point!(kd, Ke, fe, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c, c,
                                   kd.interior_weight[q])
            end
        end
    else
        copyto!(Ke, kd.const_cut[kd.cut_cell_index[cell]])
        @inbounds for q in quadrature.pointers[cell]:(quadrature.pointers[cell + 1] - 1)
            sh_t, gx_t, gy_t, sh_c, gx_c, gy_c = _point_bases(
                kd.cut_shape_t, kd.cut_gx_t, kd.cut_gy_t,
                kd.cut_shape_c, kd.cut_gx_c, kd.cut_gy_c,
                q, kd.test_space, kd.trial_space)
            csh = _packed4(kd.cut_coefficient_shape, q)
            q2sh = _maybe_q2_basis(
                Val(has_q2), kd.cut_q2_coefficient_shape, q)
            c = if point_context
                q2gx = _maybe_q2_basis(
                    Val(has_q2), kd.cut_q2_coefficient_gx, q)
                q2gy = _maybe_q2_basis(
                    Val(has_q2), kd.cut_q2_coefficient_gy, q)
                _coefficient_point(
                    csh, _packed4(kd.cut_coefficient_gx, q),
                    _packed4(kd.cut_coefficient_gy, q), q2sh, q2gx, q2gy,
                    coefficient_nodes, q2_coefficient_nodes, cell,
                    coefficients, kd.coefficient_elements)
            else
                _coefficient_values(
                    csh, q2sh, coefficient_nodes, q2_coefficient_nodes,
                    cell, coefficients, kd.coefficient_elements)
            end
            _accumulate_point!(kd, Ke, fe, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c, c,
                               quadrature.weight[q])
        end
    end

    tables = kd.boundary
    @inbounds for q in boundary_quadrature.pointers[cell]:(boundary_quadrature.pointers[cell + 1] - 1)
        sh_t, gx_t, gy_t, sh_c, gx_c, gy_c = _point_bases(
            tables.shape_t, tables.gx_t, tables.gy_t,
            tables.shape_c, tables.gx_c, tables.gy_c,
            q, kd.test_space, kd.trial_space)
        q2sh = _maybe_q2_basis(
            Val(has_q2), tables.q2_coefficient_shape, q)
        q2gx = _maybe_q2_basis(
            Val(has_q2), tables.q2_coefficient_gx, q)
        q2gy = _maybe_q2_basis(
            Val(has_q2), tables.q2_coefficient_gy, q)
        c = _coefficient_point(
            _packed4(tables.coefficient_shape, q),
            _packed4(tables.coefficient_gx, q),
            _packed4(tables.coefficient_gy, q), q2sh, q2gx, q2gy,
            coefficient_nodes, q2_coefficient_nodes, cell,
            coefficients, kd.coefficient_elements)
        facet = Int(boundary_quadrature.facet[q])
        point = BoundaryPoint(
            c, _facet_values(facet_fields, facet),
            (boundary_quadrature.normal_x[q], boundary_quadrature.normal_y[q]))
        _accumulate_boundary_point!(
            kd, Ke, fe, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c, point,
            boundary_quadrature.weight[q])
    end
    return nothing
end

@inline function _transform_agfem!(
    scratch::ThreadBuffer{T}, C::AbstractMatrix{T},
    kind::UInt8 = CONSTRAINT_DENSE,
    perm::AbstractVector{Int32} = Int32[],
) where {T}
    nt = size(C, 1)
    if kind == CONSTRAINT_IDENTITY
        @inbounds for column in 1:nt, row in 1:nt
            scratch.effective_matrix[row, column] = scratch.raw_matrix[row, column]
        end
        @inbounds for row in 1:nt
            scratch.effective_vector[row] = scratch.raw_vector[row]
        end
        return nothing
    elseif kind == CONSTRAINT_PERMUTATION
        @inbounds for column in 1:nt, row in 1:nt
            scratch.effective_matrix[row, column] =
                scratch.raw_matrix[perm[row], perm[column]]
        end
        @inbounds for row in 1:nt
            scratch.effective_vector[row] = scratch.raw_vector[perm[row]]
        end
        return nothing
    end
    # Genuine dense constraints: K_eff = C·K·Cᵀ and f_eff = C·f, with
    #     (m×n)(n×n) -> (m×n), (m×n)(n×m) -> (m×m), (m×n)(n) -> (m).
    # Views size every product to this cell's constraint; the shared buffers may
    # be larger. Summation order can differ from the scalar loops, so callers
    # use roundoff-based tolerances rather than bit equality.
    m = size(C, 1)
    n = size(C, 2)
    raw_view = view(scratch.raw_matrix, 1:n, 1:n)
    left_view = view(scratch.left_matrix, 1:m, 1:n)
    effective_view = view(scratch.effective_matrix, 1:m, 1:m)
    mul!(left_view, C, raw_view)
    mul!(effective_view, left_view, transpose(C))
    mul!(view(scratch.effective_vector, 1:m), C, view(scratch.raw_vector, 1:n))
    return nothing
end
