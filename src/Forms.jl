# Element variational-form description and generated element tables.
#
# The physics is written once as an `ElementVariationalForm` (Ciarlet: discrete
# variational formulation restricted to the element — find `u_h` such that
# `a_K(u_h, v_h) = L_K(v_h)`): a test space, a trial space
# (each a tuple of `FiniteElement` blocks), a tuple of bilinear slots (test
# differential operator × trial differential operator × coefficient rule, per
# block pair) and right-hand-side slots (test operator × source function ×
# scale, per test block).
# `generate_element_tables` turns the form into plain data — precomputed
# geometry tables — consumed by the assembly hot loop. The package contains no
# PDE and no callable kernel: forms are data, tables are data, differential
# operators are user types.

# ---- finite elements (Ciarlet: finite element (K, P, Sigma) type) -----------

"""
    FiniteElement{E,Nc}(element)

One block of a discrete variational formulation: `element` (a
`ReferenceFiniteElement`) times `Nc` components (1 scalar, 2 vector). Mixed
test/trial forms pair tuples of blocks; every block lives on the same
background grid and its own node lattice, so all finite elements are
compatible with the background triangulation by construction.
"""
struct FiniteElement{E<:ReferenceFiniteElement,Nc}
    element::E
end
const FESpace = FiniteElement
FiniteElement(element::E, ncomp::Int) where {E<:ReferenceFiniteElement} =
    FiniteElement{E,ncomp}(element)

@inline ncomp(::FiniteElement{E,Nc}) where {E,Nc} = Nc
@inline nnode(space::FiniteElement) = nnode(space.element)
@inline ndofs(space::FiniteElement) = ncomp(space) * nnode(space)

# Per-block node offsets inside the packed per-point basis tables.
@inline function _block_node_offsets(spaces::Tuple)
    return ntuple(Val(length(spaces))) do b
        acc = 0
        for i in 1:(b - 1)
            acc += nnode(spaces[i].element)
        end
        acc
    end
end

# Dominant reference element: the block with the most nodes sets the default
# quadrature formulas integrating every block pair.
@inline function _dominant_element(test_space::Tuple, trial_space::Tuple)
    best = test_space[1].element
    for b in 1:length(test_space)
        nnode(test_space[b].element) > nnode(best) && (best = test_space[b].element)
    end
    for b in 1:length(trial_space)
        nnode(trial_space[b].element) > nnode(best) && (best = trial_space[b].element)
    end
    return best
end

# ---- basis utilities (pure data-level) ------------------------------------

# Unpack 4 consecutive entries of a packed per-point array.
@inline function _packed4(v::Vector{T}, q::Int) where {T}
    @inbounds b = 4 * (q - 1)
    @inbounds return (v[b+1], v[b+2], v[b+3], v[b+4])
end

# Unpack `n` consecutive entries of one block of a packed per-point array;
# the packing is point-major with stride `stride` (total nodes per point
# across blocks), `n` and `offset` stay compile-time constants in the hot loop.
@inline function _packed_basis(v::Vector{T}, q::Int, stride::Int,
                               n::Int, offset::Int) where {T}
    @inbounds b = offset + stride * (q - 1)
    return ntuple(k -> v[b+k], Val(n))
end

# Unpack every block's basis at quadrature point `q` from packed tables.
# Generated: the block sizes and offsets are compile-time constants derived
# from the space types, so every `_packed_basis` call is emitted unrolled with
# literal arguments — allocation-free for any block structure (single- or
# multi-block, mixed spaces).
@generated function _point_bases(shape_t, gx_t, gy_t, shape_c, gx_c, gy_c,
                                 q::Int, test_space::Tuple, trial_space::Tuple)
    element(t) = t.parameters[1]
    nnodes_t = [nnode(element(t).instance) for t in test_space.parameters]
    nnodes_c = [nnode(element(t).instance) for t in trial_space.parameters]
    offsets(nnodes) = begin
        result = Int[]
        acc = 0
        for nn in nnodes
            push!(result, acc)
            acc += nn
        end
        result
    end
    offsets_t = offsets(nnodes_t)
    offsets_c = offsets(nnodes_c)
    stride_t = sum(nnodes_t)
    stride_c = sum(nnodes_c)
    unroll(v, stride, nnodes, offs) = Expr(
        :tuple, [Expr(:call, :_packed_basis, v, :q, stride, nn, off)
                 for (nn, off) in zip(nnodes, offs)]...)
    return quote
        sh_t = $(unroll(:shape_t, stride_t, nnodes_t, offsets_t))
        gx_t = $(unroll(:gx_t, stride_t, nnodes_t, offsets_t))
        gy_t = $(unroll(:gy_t, stride_t, nnodes_t, offsets_t))
        sh_c = $(unroll(:shape_c, stride_c, nnodes_c, offsets_c))
        gx_c = $(unroll(:gx_c, stride_c, nnodes_c, offsets_c))
        gy_c = $(unroll(:gy_c, stride_c, nnodes_c, offsets_c))
        return sh_t, gx_t, gy_t, sh_c, gx_c, gy_c
    end
end

# ---- differential operators (Ciarlet: differential operators applied to trial ------
# ---- and test functions; pairing evaluated at one quadrature node) ------------
#
# `AbstractDifferentialOperator` is the engine-side contract, not a physics
# library. User code subtypes it and defines the two pairings the engine calls:
#
#   _op_pair(otest, otrial, trow, tcol, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c)
#   _rhs_contract!(fe, otest, slot, sh_t, gx_t, gy_t, c, w, offset)
#
# Both receive block-local basis data (shape/gradient tuples at one quadrature
# point) and know nothing about the element; the engine ships no operator.

abstract type AbstractDifferentialOperator end
const AbstractOperator = AbstractDifferentialOperator

# Contract entry points: user code adds methods on its operator types; the
# generic fallbacks only make the names visible to the hot loop.
_op_pair(otest, otrial, row::Int, col::Int, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c) =
    error("operators $(typeof(otest))/$(typeof(otrial)) do not define `_op_pair`")
_rhs_contract!(fe, otest, slot, sh_t, gx_t, gy_t, c, w, offset::Int) =
    error("operator $(typeof(otest)) does not define `_rhs_contract!`")

# Boundary-specific operator methods are optional. Operators that do not use
# the interface frame retain the volume contraction.
_boundary_op_pair(otest, otrial, row::Int, col::Int,
                  sh_t, gx_t, gy_t, sh_c, gx_c, gy_c, point) =
    _op_pair(otest, otrial, row, col, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c)
_boundary_rhs_contract!(fe, otest, slot, sh_t, gx_t, gy_t,
                        point, w, offset::Int) =
    _rhs_contract!(fe, otest, slot, sh_t, gx_t, gy_t, point, w, offset)

struct PointContext{V,QX,QY,Q2X,Q2Y,N1,N2,C,E}
    values::V
    q1_gx::QX
    q1_gy::QY
    q2_gx::Q2X
    q2_gy::Q2Y
    q1_nodes::N1
    q2_nodes::N2
    cell::Int
    coefficients::C
    elements::E
end

@inline Base.length(point::PointContext) = length(point.values)
@inline Base.getindex(point::PointContext, index::Int) = point.values[index]
@inline Base.iterate(point::PointContext, state=1) =
    state > length(point) ? nothing : (point[state], state + 1)

struct BoundaryPoint{C,F,T}
    coefficients::C
    facet_values::F
    normal::NTuple{2,T}
end


@inline Base.length(point::BoundaryPoint) = length(point.coefficients)
@inline Base.getindex(point::BoundaryPoint, index::Int) = point.coefficients[index]
@inline Base.iterate(point::BoundaryPoint, state=1) = iterate(point.coefficients, state)

"""Physical `(dx, dy)` gradient of scalar coefficient field `index`."""
@inline function coefficient_gradient(point::PointContext, index::Int)
    element = point.elements[index]
    values = point.coefficients[index]
    return _coefficient_value(
               element, point.q1_gx, point.q2_gx,
               point.q1_nodes, point.q2_nodes, point.cell, values),
           _coefficient_value(
               element, point.q1_gy, point.q2_gy,
               point.q1_nodes, point.q2_nodes, point.cell, values)
end
@inline coefficient_gradient(point::BoundaryPoint, index::Int) =
    coefficient_gradient(point.coefficients, index)

"""Value of per-facet field `index` at a boundary quadrature point."""
@inline facet_value(point::BoundaryPoint, index::Int) = point.facet_values[index]

"""Outward unit normal at a boundary quadrature point."""
@inline boundary_normal(point::BoundaryPoint) = point.normal

"""Counter-clockwise tangent `(-n_y, n_x)` at a boundary quadrature point.

Gridap transcriptions must use `normal ⋅ TensorValue(0.0, -1.0, 1.0, 0.0)`:
`TensorValue` stores components column-major, so that order evaluates to
`(-n_y, n_x)` while `TensorValue(0.0, -1.0, 1.0, 0.0) ⋅ normal` evaluates to
the opposite tangent and flips single-tangent terms such as `Ft`.
"""
@inline boundary_tangent(point::BoundaryPoint) = (-point.normal[2], point.normal[1])

# ---- coefficient rules for bilinear slots ----------------------------------

abstract type AbstractCoefficient end
struct ConstantCoefficient{T} <: AbstractCoefficient
    m::T
end
struct FieldCoefficient{I} <: AbstractCoefficient
    index::I
end
struct StateFunction{F} <: AbstractCoefficient
    f::F
end

"""
    BilinearSlot{TestBlock,TrialBlock}(otest, otrial, coeff)

Bilinear contribution `∫ coeff · otest(v) ⊙ otrial(u)` restricted to test
block `TestBlock` rows and trial block `TrialBlock` columns (block indices
start at 1; `BilinearSlot(otest, otrial, coeff)` defaults to block pair
(1, 1)).
"""
struct BilinearSlot{TB,CB,Otest<:AbstractDifferentialOperator,Otrial<:AbstractDifferentialOperator,
                    C<:AbstractCoefficient}
    test_operator::Otest
    trial_operator::Otrial
    coefficient::C
end
BilinearSlot{TB,CB}(otest, otrial, coeff) where {TB,CB} =
    BilinearSlot{TB,CB,typeof(otest),typeof(otrial),typeof(coeff)}(otest, otrial, coeff)
BilinearSlot(otest, otrial, coeff) = BilinearSlot{1,1}(otest, otrial, coeff)

"""
    RhsSlot{TestBlock}(otest, source, scale)

Right-hand-side contribution `∫ scale · source(c) ⊙ otest(v)` on test block
`TestBlock` (`RhsSlot(otest, source, scale)` defaults to block 1).
"""
struct RhsSlot{TB,O<:AbstractDifferentialOperator,F}
    test_operator::O
    source::F
    scale::Float64
end
RhsSlot{TB}(otest, source, scale) where {TB} =
    RhsSlot{TB,typeof(otest),typeof(source)}(otest, source, scale)
RhsSlot(otest, source, scale) = RhsSlot{1}(otest, source, scale)

"""
    ElementVariationalForm(test_space, trial_space, bilinear, rhs, ncoefficients;
                           coefficient_elements=nothing,
                           coefficient_gradients=false,
                           boundary_bilinear=(), boundary_rhs=(), nfacetfields=0)

Discrete variational formulation restricted to the element (Ciarlet:
`a_K(u_h, v_h) = L_K(v_h)`), used to generate assembly data.
`test_space`/`trial_space` are `FiniteElement`s (or tuples of them, for mixed
spaces) on the same background grid; `bilinear` is a tuple of `BilinearSlot`s
contributing `∫ coeff · otest(v) ⊙ otrial(u)` to the element stiffness matrix;
`rhs` a tuple of `RhsSlot`s contributing `∫ scale · source(c) ⊙ otest(v)` to
the element load vector. `ncoefficients` is the number of scalar coefficient
fields consumed at each quadrature point (interpolated on the Q1 node lattice
unless `coefficient_elements` supplies one Q1/Q2 reference element
per field). Boundary slots receive a `BoundaryPoint` exposing coefficient
gradients, per-facet fields, and the oriented normal/tangent frame. Set
`coefficient_gradients=true` to expose the same lazy gradient access to volume
state/source callbacks without boundary slots.
"""
struct ElementVariationalForm{TS<:Tuple,TT<:Tuple,B<:Tuple,R<:Tuple,
                              BB<:Tuple,BR<:Tuple,CE<:Tuple}
    test_space::TS
    trial_space::TT
    bilinear::B
    rhs::R
    boundary_bilinear::BB
    boundary_rhs::BR
    ncoefficients::Int
    nfacetfields::Int
    coefficient_elements::CE
    coefficient_gradients::Bool
end
const LocalForm = ElementVariationalForm
function ElementVariationalForm(test_space::Tuple, trial_space::Tuple,
                                bilinear, rhs, ncoefficients;
                                coefficient_elements=nothing,
                                coefficient_gradients::Bool=false,
                                boundary_bilinear=(), boundary_rhs=(),
                                nfacetfields=0)
    ncoefficients >= 0 ||
        throw(ArgumentError("ncoefficients must be nonnegative"))
    nfacetfields >= 0 ||
        throw(ArgumentError("nfacetfields must be nonnegative"))
    elements = isnothing(coefficient_elements) ?
        ntuple(_ -> Q1ReferenceQuadrilateral(), ncoefficients) :
        coefficient_elements
    elements isa Tuple && length(elements) == ncoefficients &&
        all(element -> element isa Union{Q1ReferenceQuadrilateral,
                                        Q2ReferenceQuadrilateral}, elements) ||
        throw(ArgumentError("coefficient_elements must contain one Q1/Q2 element per coefficient"))
    return ElementVariationalForm(
        test_space, trial_space, bilinear, rhs,
        boundary_bilinear, boundary_rhs, ncoefficients, nfacetfields,
        elements, coefficient_gradients)
end
ElementVariationalForm(test_space::FiniteElement, trial_space::FiniteElement,
                       bilinear, rhs, ncoefficients; kwargs...) =
    ElementVariationalForm((test_space,), (trial_space,), bilinear, rhs,
                           ncoefficients; kwargs...)
ElementVariationalForm(test_space::FiniteElement, bilinear, rhs,
                       ncoefficients; kwargs...) =
    ElementVariationalForm((test_space,), (test_space,), bilinear, rhs,
                           ncoefficients; kwargs...)

# ---- generated element tables (Ciarlet: reference basis evaluated at ---------
# ---- quadrature nodes E_{\hat K} + precomputed constant-part element ---------
# ---- stiffness matrices A^K) --------------------------------------------------

"""
    ElementTables

Plain data produced by `generate_element_tables`: reference basis functions
evaluated at the quadrature nodes plus precomputed constant-part element
stiffness matrices.
Constant bilinear slots are summed into `const_interior` (shared by every
non-cut cell, which uses the dominant element's interior quadrature formula)
and per-cell `const_cut` tables. Field/state bilinear slots and RHS slots keep
the packed per-point basis values (one segment per finite-element block,
`cut_shape_*`/`cut_gx_*`/`cut_gy_*`) and are contracted in the hot loop. The
The coefficient tables hold Q1 and, when requested, Q2 bases at the same
points.
"""
mutable struct BoundaryTables{T}
    shape_t::Vector{T}
    gx_t::Vector{T}
    gy_t::Vector{T}
    shape_c::Vector{T}
    gx_c::Vector{T}
    gy_c::Vector{T}
    coefficient_shape::Vector{T}
    coefficient_gx::Vector{T}
    coefficient_gy::Vector{T}
    q2_coefficient_shape::Vector{T}
    q2_coefficient_gx::Vector{T}
    q2_coefficient_gy::Vector{T}
end

BoundaryTables{T}() where {T} = BoundaryTables{T}(
    T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[])

mutable struct ElementTables{T,TS<:Tuple,TT<:Tuple,B2<:Tuple,B<:Tuple,R<:Tuple,
                             BB<:Tuple,BR<:Tuple,CE<:Tuple}
    test_space::TS
    trial_space::TT
    const_bilinear::B2
    dynamic_bilinear::B
    rhs::R
    boundary_bilinear::BB
    boundary_rhs::BR
    nfacetfields::Int
    coefficient_elements::CE
    coefficient_gradients::Bool
    interior_shape_t::Vector{T}
    interior_gx_t::Vector{T}
    interior_gy_t::Vector{T}
    interior_shape_c::Vector{T}
    interior_gx_c::Vector{T}
    interior_gy_c::Vector{T}
    interior_coefficient_shape::Vector{T}
    interior_coefficient_gx::Vector{T}
    interior_coefficient_gy::Vector{T}
    interior_q2_coefficient_shape::Vector{T}
    interior_q2_coefficient_gx::Vector{T}
    interior_q2_coefficient_gy::Vector{T}
    interior_weight::Vector{T}
    cut_shape_t::Vector{T}
    cut_gx_t::Vector{T}
    cut_gy_t::Vector{T}
    cut_shape_c::Vector{T}
    cut_gx_c::Vector{T}
    cut_gy_c::Vector{T}
    cut_coefficient_shape::Vector{T}
    cut_coefficient_gx::Vector{T}
    cut_coefficient_gy::Vector{T}
    cut_q2_coefficient_shape::Vector{T}
    cut_q2_coefficient_gx::Vector{T}
    cut_q2_coefficient_gy::Vector{T}
    boundary::BoundaryTables{T}
    const_interior::Matrix{T}
    cut_cell_index::Vector{Int32}
    const_cut::Vector{Matrix{T}}
end
const LocalKernel = ElementTables

@inline _slot_value(c::ConstantCoefficient{T}, cq) where {T} = c.m
@inline _slot_value(c::FieldCoefficient, cq) = cq[c.index]
@inline _slot_value(c::StateFunction, cq) = c.f(cq)

@inline function _background_coefficient(shape, cell_nodes, cell, values)
    value = zero(eltype(values))
    @inbounds for node in eachindex(shape)
        value += shape[node] * values[cell_nodes[node, cell]]
    end
    return value
end

@inline _coefficient_value(::Q1ReferenceQuadrilateral, q1_shape, q2_shape,
                           q1_nodes, q2_nodes, cell, values) =
    _background_coefficient(q1_shape, q1_nodes, cell, values)
@inline _coefficient_value(::Q2ReferenceQuadrilateral, q1_shape, q2_shape,
                           q1_nodes, q2_nodes, cell, values) =
    _background_coefficient(q2_shape, q2_nodes, cell, values)

@inline function _rule_element(test_space::Tuple, trial_space::Tuple,
                                coefficient_elements::Tuple)
    best = _dominant_element(test_space, trial_space)
    best isa Q2ReferenceQuadrilateral && return best
    any(element -> element isa Q2ReferenceQuadrilateral,
        coefficient_elements) && return Q2ReferenceQuadrilateral()
    return best
end

# Recursive unrolling over the coefficient tuple: indexing a heterogeneous
# Q1/Q2 element tuple with a runtime index is type-unstable, so each field is
# resolved with its compile-time element type instead.
@inline _coefficient_values(q1_shape, q2_shape, q1_nodes, q2_nodes,
                            cell, ::Tuple{}, ::Tuple{}) = ()
@inline function _coefficient_values(q1_shape, q2_shape, q1_nodes, q2_nodes,
                                     cell, coefficients::Tuple,
                                     elements::Tuple)
    value = _coefficient_value(first(elements), q1_shape, q2_shape,
                               q1_nodes, q2_nodes, cell, first(coefficients))
    rest = _coefficient_values(q1_shape, q2_shape, q1_nodes, q2_nodes,
                               cell, Base.tail(coefficients),
                               Base.tail(elements))
    return (value, rest...)
end

@inline function _coefficient_point(q1_shape, q1_gx, q1_gy,
                                    q2_shape, q2_gx, q2_gy,
                                    q1_nodes, q2_nodes, cell,
                                    coefficients, elements)
    return PointContext(
        _coefficient_values(q1_shape, q2_shape, q1_nodes, q2_nodes,
                            cell, coefficients, elements),
        q1_gx, q1_gy, q2_gx, q2_gy, q1_nodes, q2_nodes, cell,
        coefficients, elements,
    )
end

# Compile-time Q2-coefficient presence from the element tuple type. Selecting
# the Q2 basis (or `nothing`) with a runtime `isempty` check merges the two
# into a heap-boxed Union for every quadrature point; resolving it from the
# type keeps the per-point contexts concrete and stack-allocated.
@generated _has_q2_elements(::Type{T}) where {T<:Tuple} =
    any(p -> p <: Q2ReferenceQuadrilateral, T.parameters) ? :(true) : :(false)
@inline _maybe_q2_basis(::Val{true}, values::Vector, q::Int) =
    _packed_basis(values, q, 9, 9, 0)
@inline _maybe_q2_basis(::Val{false}, ::Vector, ::Int) = nothing

@inline _facet_values(::Tuple{}, ::Int) = ()
@inline function _facet_values(fields::Tuple, facet::Int)
    @inbounds value = first(fields)[facet]
    rest = _facet_values(Base.tail(fields), facet)
    return (value, rest...)
end

# ---- per-point accumulation (hot loop) --------------------------------------

@inline function _accumulate_point!(kd::ElementTables{T}, Ke, fe,
                                    sh_t, gx_t, gy_t, sh_c, gx_c, gy_c,
                                    c, w::T) where {T}
    _accumulate_bilinear_slots!(
        Ke, kd.dynamic_bilinear, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c, c, w,
        kd.test_space, kd.trial_space)
    _accumulate_rhs_slots!(fe, kd.rhs, sh_t, gx_t, gy_t, c, w, kd.test_space)
    return nothing
end

@inline _accumulate_bilinear_slots!(Ke, ::Tuple{}, sh_t, gx_t, gy_t,
                                    sh_c, gx_c, gy_c, c, w,
                                    test_space, trial_space) = nothing
@inline function _accumulate_bilinear_slots!(
    Ke, slots::Tuple{S,Vararg{Any}}, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c,
    c, w, test_space, trial_space,
) where {S}
    _accumulate_bilinear!(Ke, first(slots), sh_t, gx_t, gy_t,
                          sh_c, gx_c, gy_c, c, w, test_space, trial_space)
    return _accumulate_bilinear_slots!(
        Ke, Base.tail(slots), sh_t, gx_t, gy_t, sh_c, gx_c, gy_c,
        c, w, test_space, trial_space)
end

@inline _accumulate_rhs_slots!(fe, ::Tuple{}, sh_t, gx_t, gy_t,
                               c, w, test_space) = nothing
@inline function _accumulate_rhs_slots!(
    fe, slots::Tuple{S,Vararg{Any}}, sh_t, gx_t, gy_t,
    c, w, test_space,
) where {S}
    _accumulate_rhs!(fe, first(slots), sh_t, gx_t, gy_t, c, w, test_space)
    return _accumulate_rhs_slots!(
        fe, Base.tail(slots), sh_t, gx_t, gy_t, c, w, test_space)
end

@inline function _accumulate_bilinear!(Ke, slot::BilinearSlot{TB,CB,Otest,Otrial,C},
                                       sh_t, gx_t, gy_t, sh_c, gx_c, gy_c,
                                       c, w, test_space::Tuple,
                                       trial_space::Tuple) where {TB,CB,Otest,Otrial,C}
    coeff = _slot_value(slot.coefficient, c)
    iszero(coeff) && return nothing
    offset_t = 0
    for b in 1:(TB - 1)
        offset_t += ndofs(test_space[b])
    end
    offset_c = 0
    for b in 1:(CB - 1)
        offset_c += ndofs(trial_space[b])
    end
    nt = ndofs(test_space[TB])
    nc = ndofs(trial_space[CB])
    sh_tb = sh_t[TB]
    gx_tb = gx_t[TB]
    gy_tb = gy_t[TB]
    sh_cb = sh_c[CB]
    gx_cb = gx_c[CB]
    gy_cb = gy_c[CB]
    @inbounds for col in 1:nc, row in 1:nt
        Ke[row + offset_t, col + offset_c] +=
            coeff * _op_pair(slot.test_operator, slot.trial_operator, row, col,
                             sh_tb, gx_tb, gy_tb, sh_cb, gx_cb, gy_cb) * w
    end
    return nothing
end

@inline function _accumulate_rhs!(fe, slot::RhsSlot{TB,O,F},
                                  sh_t, gx_t, gy_t, c, w,
                                  test_space::Tuple) where {TB,O,F}
    offset_t = 0
    for b in 1:(TB - 1)
        offset_t += ndofs(test_space[b])
    end
    return _rhs_contract!(fe, slot.test_operator, slot,
                           sh_t[TB], gx_t[TB], gy_t[TB], c, w, offset_t)
end

@inline function _accumulate_boundary_point!(kd::ElementTables, Ke, fe,
                                             sh_t, gx_t, gy_t,
                                             sh_c, gx_c, gy_c, point, w)
    _accumulate_boundary_bilinear_slots!(
        Ke, kd.boundary_bilinear, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c,
        point, w, kd.test_space, kd.trial_space)
    _accumulate_boundary_rhs_slots!(
        fe, kd.boundary_rhs, sh_t, gx_t, gy_t, point, w, kd.test_space)
    return nothing
end

@inline _accumulate_boundary_bilinear_slots!(
    Ke, ::Tuple{}, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c,
    point, w, test_space, trial_space) = nothing
@inline function _accumulate_boundary_bilinear_slots!(
    Ke, slots::Tuple{S,Vararg{Any}}, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c,
    point, w, test_space, trial_space,
) where {S}
    _accumulate_boundary_bilinear!(
        Ke, first(slots), sh_t, gx_t, gy_t, sh_c, gx_c, gy_c,
        point, w, test_space, trial_space)
    return _accumulate_boundary_bilinear_slots!(
        Ke, Base.tail(slots), sh_t, gx_t, gy_t, sh_c, gx_c, gy_c,
        point, w, test_space, trial_space)
end

@inline _accumulate_boundary_rhs_slots!(
    fe, ::Tuple{}, sh_t, gx_t, gy_t, point, w, test_space) = nothing
@inline function _accumulate_boundary_rhs_slots!(
    fe, slots::Tuple{S,Vararg{Any}}, sh_t, gx_t, gy_t,
    point, w, test_space,
) where {S}
    _accumulate_boundary_rhs!(
        fe, first(slots), sh_t, gx_t, gy_t, point, w, test_space)
    return _accumulate_boundary_rhs_slots!(
        fe, Base.tail(slots), sh_t, gx_t, gy_t, point, w, test_space)
end

@inline function _accumulate_boundary_bilinear!(
    Ke, slot::BilinearSlot{TB,CB,Otest,Otrial,C},
    sh_t, gx_t, gy_t, sh_c, gx_c, gy_c, point, w,
    test_space::Tuple, trial_space::Tuple,
) where {TB,CB,Otest,Otrial,C}
    coeff = _slot_value(slot.coefficient, point)
    iszero(coeff) && return nothing
    offset_t = 0
    for b in 1:(TB - 1)
        offset_t += ndofs(test_space[b])
    end
    offset_c = 0
    for b in 1:(CB - 1)
        offset_c += ndofs(trial_space[b])
    end
    nt = ndofs(test_space[TB])
    nc = ndofs(trial_space[CB])
    sh_tb, gx_tb, gy_tb = sh_t[TB], gx_t[TB], gy_t[TB]
    sh_cb, gx_cb, gy_cb = sh_c[CB], gx_c[CB], gy_c[CB]
    @inbounds for col in 1:nc, row in 1:nt
        Ke[row + offset_t, col + offset_c] += coeff *
            _boundary_op_pair(slot.test_operator, slot.trial_operator, row, col,
                              sh_tb, gx_tb, gy_tb, sh_cb, gx_cb, gy_cb, point) * w
    end
    return nothing
end

@inline function _accumulate_boundary_rhs!(
    fe, slot::RhsSlot{TB,O,F}, sh_t, gx_t, gy_t, point, w,
    test_space::Tuple,
) where {TB,O,F}
    offset_t = 0
    for b in 1:(TB - 1)
        offset_t += ndofs(test_space[b])
    end
    return _boundary_rhs_contract!(
        fe, slot.test_operator, slot, sh_t[TB], gx_t[TB], gy_t[TB],
        point, w, offset_t)
end

# ---- generation (once per geometry refresh, never in the hot loop) ----------

"""
    generate_element_tables(form, hx, hy, quadrature, ncells) -> ElementTables

Precompute the reference-basis-at-quadrature tables and constant-part element
stiffness matrices for `form` on a background triangulation with spacing
`(hx, hy)`. `quadrature` is the plan's `CutQuadrature` (or `nothing` for the
body-fitted path); `ncells` the number of cells the quadrature covers.
"""
# Rebuild the basis-at-quadrature and constant element-stiffness tables of an
# existing element-tables object after a quadrature refresh. Reuses the
# cut-stream buffers so a warmed refresh stays allocation-free.
function _regenerate_element_tables!(kd::ElementTables{T}, hx::T, hy::T,
                                   quadrature, ncells::Int,
                                   rule::_ReferenceQuadrature{2,T}) where {T}
    points, rule_weights = rule.points, rule.weights
    Nq = length(points)
    test_space = kd.test_space
    trial_space = kd.trial_space
    node_offsets_t = _block_node_offsets(test_space)
    node_offsets_c = _block_node_offsets(trial_space)
    ntot_t = node_offsets_t[end] + nnode(test_space[end].element)
    ntot_c = node_offsets_c[end] + nnode(trial_space[end].element)
    has_q2_coefficients = any(
        element -> element isa Q2ReferenceQuadrilateral,
        kd.coefficient_elements)
    resize!(kd.interior_shape_t, Nq * ntot_t)
    resize!(kd.interior_gx_t, Nq * ntot_t)
    resize!(kd.interior_gy_t, Nq * ntot_t)
    resize!(kd.interior_shape_c, Nq * ntot_c)
    resize!(kd.interior_gx_c, Nq * ntot_c)
    resize!(kd.interior_gy_c, Nq * ntot_c)
    resize!(kd.interior_coefficient_shape, 4Nq)
    resize!(kd.interior_coefficient_gx, 4Nq)
    resize!(kd.interior_coefficient_gy, 4Nq)
    resize!(kd.interior_q2_coefficient_shape, has_q2_coefficients ? 9Nq : 0)
    resize!(kd.interior_q2_coefficient_gx, has_q2_coefficients ? 9Nq : 0)
    resize!(kd.interior_q2_coefficient_gy, has_q2_coefficients ? 9Nq : 0)
    resize!(kd.interior_weight, Nq)
    @inbounds for q in 1:Nq
        r, s = points[q]
        for b in 1:length(test_space)
            sh = shape(test_space[b].element, r, s)
            gxq, gyq = gradients(test_space[b].element, r, s, hx, hy)
            base = ntot_t * (q - 1) + node_offsets_t[b]
            for n in 1:nnode(test_space[b].element)
                kd.interior_shape_t[base+n] = sh[n]
                kd.interior_gx_t[base+n] = gxq[n]
                kd.interior_gy_t[base+n] = gyq[n]
            end
        end
        for b in 1:length(trial_space)
            sh = shape(trial_space[b].element, r, s)
            gxq, gyq = gradients(trial_space[b].element, r, s, hx, hy)
            base = ntot_c * (q - 1) + node_offsets_c[b]
            for n in 1:nnode(trial_space[b].element)
                kd.interior_shape_c[base+n] = sh[n]
                kd.interior_gx_c[base+n] = gxq[n]
                kd.interior_gy_c[base+n] = gyq[n]
            end
        end
        csh = shape(Q1ReferenceQuadrilateral(), r, s)
        cgx, cgy = gradients(Q1ReferenceQuadrilateral(), r, s, hx, hy)
        cbase = 4 * (q - 1)
        for n in 1:4
            kd.interior_coefficient_shape[cbase+n] = csh[n]
            kd.interior_coefficient_gx[cbase+n] = cgx[n]
            kd.interior_coefficient_gy[cbase+n] = cgy[n]
        end
        if has_q2_coefficients
            q2sh = shape(Q2ReferenceQuadrilateral(), r, s)
            q2gx, q2gy = gradients(Q2ReferenceQuadrilateral(), r, s, hx, hy)
            q2base = 9 * (q - 1)
            for n in 1:9
                kd.interior_q2_coefficient_shape[q2base+n] = q2sh[n]
                kd.interior_q2_coefficient_gx[q2base+n] = q2gx[n]
                kd.interior_q2_coefficient_gy[q2base+n] = q2gy[n]
            end
        end
        kd.interior_weight[q] = T(rule_weights[q]) * hx * hy
    end

    fill!(kd.const_interior, zero(T))
    for slot in kd.const_bilinear
        @inbounds for q in 1:Nq
            sh_t, gx_t, gy_t, sh_c, gx_c, gy_c = _point_bases(
                kd.interior_shape_t, kd.interior_gx_t, kd.interior_gy_t,
                kd.interior_shape_c, kd.interior_gx_c, kd.interior_gy_c,
                q, test_space, trial_space)
            _accumulate_bilinear!(kd.const_interior, slot, sh_t, gx_t, gy_t,
                                  sh_c, gx_c, gy_c, nothing,
                                  kd.interior_weight[q], test_space, trial_space)
        end
    end

    fill!(kd.cut_cell_index, 0)
    if isnothing(quadrature)
        empty!(kd.cut_shape_t)
        empty!(kd.cut_gx_t)
        empty!(kd.cut_gy_t)
        empty!(kd.cut_shape_c)
        empty!(kd.cut_gx_c)
        empty!(kd.cut_gy_c)
        empty!(kd.cut_coefficient_shape)
        empty!(kd.cut_coefficient_gx)
        empty!(kd.cut_coefficient_gy)
        empty!(kd.cut_q2_coefficient_shape)
        empty!(kd.cut_q2_coefficient_gx)
        empty!(kd.cut_q2_coefficient_gy)
        empty!(kd.const_cut)
        return kd
    end
    P = length(quadrature.weight)
    resize!(kd.cut_shape_t, ntot_t * P)
    resize!(kd.cut_gx_t, ntot_t * P)
    resize!(kd.cut_gy_t, ntot_t * P)
    resize!(kd.cut_shape_c, ntot_c * P)
    resize!(kd.cut_gx_c, ntot_c * P)
    resize!(kd.cut_gy_c, ntot_c * P)
    resize!(kd.cut_coefficient_shape, 4P)
    resize!(kd.cut_coefficient_gx, 4P)
    resize!(kd.cut_coefficient_gy, 4P)
    resize!(kd.cut_q2_coefficient_shape, has_q2_coefficients ? 9P : 0)
    resize!(kd.cut_q2_coefficient_gx, has_q2_coefficients ? 9P : 0)
    resize!(kd.cut_q2_coefficient_gy, has_q2_coefficients ? 9P : 0)
    @inbounds for q in 1:P
        r = quadrature.reference_r[q]
        s = quadrature.reference_s[q]
        for b in 1:length(test_space)
            sh = shape(test_space[b].element, r, s)
            gxq, gyq = gradients(test_space[b].element, r, s, hx, hy)
            base = ntot_t * (q - 1) + node_offsets_t[b]
            for n in 1:nnode(test_space[b].element)
                kd.cut_shape_t[base+n] = sh[n]
                kd.cut_gx_t[base+n] = gxq[n]
                kd.cut_gy_t[base+n] = gyq[n]
            end
        end
        for b in 1:length(trial_space)
            sh = shape(trial_space[b].element, r, s)
            gxq, gyq = gradients(trial_space[b].element, r, s, hx, hy)
            base = ntot_c * (q - 1) + node_offsets_c[b]
            for n in 1:nnode(trial_space[b].element)
                kd.cut_shape_c[base+n] = sh[n]
                kd.cut_gx_c[base+n] = gxq[n]
                kd.cut_gy_c[base+n] = gyq[n]
            end
        end
        csh = shape(Q1ReferenceQuadrilateral(), r, s)
        cgx, cgy = gradients(Q1ReferenceQuadrilateral(), r, s, hx, hy)
        cbase = 4 * (q - 1)
        for n in 1:4
            kd.cut_coefficient_shape[cbase+n] = csh[n]
            kd.cut_coefficient_gx[cbase+n] = cgx[n]
            kd.cut_coefficient_gy[cbase+n] = cgy[n]
        end
        if has_q2_coefficients
            q2sh = shape(Q2ReferenceQuadrilateral(), r, s)
            q2gx, q2gy = gradients(Q2ReferenceQuadrilateral(), r, s, hx, hy)
            q2base = 9 * (q - 1)
            for n in 1:9
                kd.cut_q2_coefficient_shape[q2base+n] = q2sh[n]
                kd.cut_q2_coefficient_gx[q2base+n] = q2gx[n]
                kd.cut_q2_coefficient_gy[q2base+n] = q2gy[n]
            end
        end
    end
    ncut = 0
    @inbounds for cell in 1:ncells
        first_point = quadrature.pointers[cell]
        last_point = quadrature.pointers[cell + 1] - 1
        first_point > last_point && continue
        ncut += 1
        kd.cut_cell_index[cell] = Int32(ncut)
    end
    while length(kd.const_cut) < ncut
        push!(kd.const_cut, zeros(T, size(kd.const_interior)))
    end
    resize!(kd.const_cut, ncut)
    # Zero all cut tables once, then accumulate each constant slot over all
    # cells (the heterogeneous slot tuple is iterated once, not per cell).
    @inbounds for cell in 1:ncells
        idx = Int(kd.cut_cell_index[cell])
        idx == 0 && continue
        fill!(kd.const_cut[idx], zero(T))
    end
    for slot in kd.const_bilinear
        @inbounds for cell in 1:ncells
            idx = Int(kd.cut_cell_index[cell])
            idx == 0 && continue
            mcell = kd.const_cut[idx]
            first_point = quadrature.pointers[cell]
            last_point = quadrature.pointers[cell + 1] - 1
            for q in first_point:last_point
                sh_t, gx_t, gy_t, sh_c, gx_c, gy_c = _point_bases(
                    kd.cut_shape_t, kd.cut_gx_t, kd.cut_gy_t,
                    kd.cut_shape_c, kd.cut_gx_c, kd.cut_gy_c,
                    q, test_space, trial_space)
                _accumulate_bilinear!(mcell, slot, sh_t, gx_t, gy_t,
                                      sh_c, gx_c, gy_c, nothing,
                                      quadrature.weight[q], test_space, trial_space)
            end
        end
    end
    return kd
end

function _regenerate_element_tables!(kd::ElementTables{T}, hx::T, hy::T,
                                     quadrature, ncells::Int) where {T}
    el = _rule_element(kd.test_space, kd.trial_space, kd.coefficient_elements)
    rule = _canonical_quadrature(
        interior_rule(el), Val(2), T, "interior quadrature")
    return _regenerate_element_tables!(kd, hx, hy, quadrature, ncells, rule)
end

function _regenerate_boundary_tables!(kd::ElementTables{T}, hx::T, hy::T,
                                      quadrature) where {T}
    tables = kd.boundary
    test_space = kd.test_space
    trial_space = kd.trial_space
    node_offsets_t = _block_node_offsets(test_space)
    node_offsets_c = _block_node_offsets(trial_space)
    ntot_t = node_offsets_t[end] + nnode(test_space[end].element)
    ntot_c = node_offsets_c[end] + nnode(trial_space[end].element)
    P = length(quadrature.weight)
    has_q2_coefficients = any(
        element -> element isa Q2ReferenceQuadrilateral,
        kd.coefficient_elements)
    resize!(tables.shape_t, ntot_t * P)
    resize!(tables.gx_t, ntot_t * P)
    resize!(tables.gy_t, ntot_t * P)
    resize!(tables.shape_c, ntot_c * P)
    resize!(tables.gx_c, ntot_c * P)
    resize!(tables.gy_c, ntot_c * P)
    resize!(tables.coefficient_shape, 4P)
    resize!(tables.coefficient_gx, 4P)
    resize!(tables.coefficient_gy, 4P)
    resize!(tables.q2_coefficient_shape, has_q2_coefficients ? 9P : 0)
    resize!(tables.q2_coefficient_gx, has_q2_coefficients ? 9P : 0)
    resize!(tables.q2_coefficient_gy, has_q2_coefficients ? 9P : 0)
    @inbounds for q in 1:P
        r = quadrature.reference_r[q]
        s = quadrature.reference_s[q]
        for b in eachindex(test_space)
            sh = shape(test_space[b].element, r, s)
            gx, gy = gradients(test_space[b].element, r, s, hx, hy)
            base = ntot_t * (q - 1) + node_offsets_t[b]
            for node in 1:nnode(test_space[b].element)
                tables.shape_t[base + node] = sh[node]
                tables.gx_t[base + node] = gx[node]
                tables.gy_t[base + node] = gy[node]
            end
        end
        for b in eachindex(trial_space)
            sh = shape(trial_space[b].element, r, s)
            gx, gy = gradients(trial_space[b].element, r, s, hx, hy)
            base = ntot_c * (q - 1) + node_offsets_c[b]
            for node in 1:nnode(trial_space[b].element)
                tables.shape_c[base + node] = sh[node]
                tables.gx_c[base + node] = gx[node]
                tables.gy_c[base + node] = gy[node]
            end
        end
        sh = shape(Q1ReferenceQuadrilateral(), r, s)
        gx, gy = gradients(Q1ReferenceQuadrilateral(), r, s, hx, hy)
        base = 4 * (q - 1)
        for node in 1:4
            tables.coefficient_shape[base + node] = sh[node]
            tables.coefficient_gx[base + node] = gx[node]
            tables.coefficient_gy[base + node] = gy[node]
        end
        if has_q2_coefficients
            sh = shape(Q2ReferenceQuadrilateral(), r, s)
            gx, gy = gradients(Q2ReferenceQuadrilateral(), r, s, hx, hy)
            base = 9 * (q - 1)
            for node in 1:9
                tables.q2_coefficient_shape[base + node] = sh[node]
                tables.q2_coefficient_gx[base + node] = gx[node]
                tables.q2_coefficient_gy[base + node] = gy[node]
            end
        end
    end
    return kd
end

function generate_element_tables(form::ElementVariationalForm, hx::T, hy::T, quadrature,
                                 ncells::Int,
                                 rule::_ReferenceQuadrature{2,T}) where {T}
    const_bilinear = Tuple(s for s in form.bilinear
                           if s.coefficient isa ConstantCoefficient)
    dynamic_bilinear = Tuple(s for s in form.bilinear
                             if !(s.coefficient isa ConstantCoefficient))
    test_space = form.test_space
    trial_space = form.trial_space
    ntest_cell = sum(ndofs(space) for space in test_space)
    ntrial_cell = sum(ndofs(space) for space in trial_space)
    kd = ElementTables(
        test_space, trial_space,
        const_bilinear, dynamic_bilinear, form.rhs,
        form.boundary_bilinear, form.boundary_rhs, form.nfacetfields,
        form.coefficient_elements, form.coefficient_gradients,
        T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[],
        T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[],
        BoundaryTables{T}(),
        zeros(T, ntest_cell, ntrial_cell),
        zeros(Int32, max(ncells, 0)), Matrix{T}[])
    _regenerate_element_tables!(kd, hx, hy, quadrature, ncells, rule)
    return kd
end


function generate_element_tables(form::ElementVariationalForm, hx::T, hy::T, quadrature,
                                 ncells::Int) where {T}
    el = _rule_element(form.test_space, form.trial_space, form.coefficient_elements)
    rule = _canonical_quadrature(
        interior_rule(el), Val(2), T, "interior quadrature")
    return generate_element_tables(form, hx, hy, quadrature, ncells, rule)
end
const generate_local_kernel = generate_element_tables
