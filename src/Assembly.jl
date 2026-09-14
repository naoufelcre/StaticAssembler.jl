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

@inline function _nz_position(matrix::SparseMatrixCSC, row::Integer, column::Integer)
    lo = Int(matrix.colptr[column])
    hi = Int(matrix.colptr[column + 1] - 1)
    position = searchsortedfirst(matrix.rowval, row, lo, hi, Base.Order.Forward)
    position <= hi && matrix.rowval[position] == row ||
        error("sparse pattern is missing entry ($row, $column)")
    return position
end

function _ragged_scatter_slots!(slots, matrix, test_dofs, trial_dofs)
    length(test_dofs) == length(trial_dofs) || error("test/trial cell count mismatch")
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
) where {T}
    nt = size(C, 1)
    nu = size(C, 1)
    nraw = size(scratch.raw_matrix, 1)
    nraw_col = size(scratch.raw_matrix, 2)
    @inbounds for row in 1:nt, raw_column in 1:nraw_col
        value = zero(T)
        for raw_row in 1:nraw
            value += C[row, raw_row] * scratch.raw_matrix[raw_row, raw_column]
        end
        scratch.left_matrix[row, raw_column] = value
    end
    @inbounds for column in 1:nu, row in 1:nt
        value = zero(T)
        for raw_column in 1:nraw_col
            value += scratch.left_matrix[row, raw_column] * C[column, raw_column]
        end
        scratch.effective_matrix[row, column] = value
    end
    @inbounds for row in 1:nt
        value = zero(T)
        for raw_row in 1:nraw
            value += C[row, raw_row] * scratch.raw_vector[raw_row]
        end
        scratch.effective_vector[row] = value
    end
    return nothing
end
