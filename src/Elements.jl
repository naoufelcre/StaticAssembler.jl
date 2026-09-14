# # Reference finite elements (Ciarlet: reference element
# # (\hat K, \hat P, \hat \Sigma), quadrilateral Q-family) and their basis /
# # quadrature / connectivity data.
#
# The assembly engine is parameterized by a `ReferenceFiniteElement`: the
# basis, quadrature rules, and background-grid connectivity are methods on the
# element type, so dispatch keeps the hot loop type-stable. Vector spaces are
# the element ⊗ R² with component-blocked DOF ordering; scalar coefficient
# fields interpolate on their declared Q1 or Q2 node lattice.

abstract type ReferenceFiniteElement end
const AbstractReferenceElement = ReferenceFiniteElement

struct Q1ReferenceQuadrilateral <: ReferenceFiniteElement end
struct Q2ReferenceQuadrilateral <: ReferenceFiniteElement end
const BilinearQuad = Q1ReferenceQuadrilateral
const BiquadraticQuad = Q2ReferenceQuadrilateral

# System-owned snapshot of a reference-cell quadrature rule. User-facing rules
# are plain `(points, weights)` pairs or objects exposing `.points`/`.weights`;
# canonicalizing once keeps their types out of refresh and assembly.
struct _ReferenceQuadrature{D,T}
    points::Vector{NTuple{D,T}}
    weights::Vector{T}
end

function _quadrature_parts(rule, name)
    if hasproperty(rule, :points) && hasproperty(rule, :weights)
        return getproperty(rule, :points), getproperty(rule, :weights)
    end
    rule isa Tuple && length(rule) == 2 ||
        throw(ArgumentError("$name must be a `(points, weights)` pair or expose `points` and `weights`"))
    return rule
end

function _canonical_quadrature(rule, ::Val{D}, ::Type{T}, name) where {D,T}
    source_points, source_weights = _quadrature_parts(rule, name)
    applicable(iterate, source_points) && applicable(iterate, source_weights) ||
        throw(ArgumentError("$name points and weights must be iterable"))
    raw_points = collect(source_points)
    raw_weights = collect(source_weights)
    !isempty(raw_points) || throw(ArgumentError("$name must contain at least one point"))
    length(raw_points) == length(raw_weights) ||
        throw(ArgumentError("$name point and weight counts differ"))

    tolerance = T(1024) * eps(T) * max(length(raw_points), 1)
    points = Vector{NTuple{D,T}}(undef, length(raw_points))
    weights = Vector{T}(undef, length(raw_weights))
    for q in eachindex(raw_points)
        point = raw_points[q]
        applicable(length, point) && length(point) == D ||
            throw(ArgumentError("$name point $q must have $D coordinates"))
        converted = ntuple(d -> T(point[d]), Val(D))
        all(isfinite, converted) ||
            throw(ArgumentError("$name contains a non-finite point"))
        all(x -> -tolerance <= x <= one(T) + tolerance, converted) ||
            throw(ArgumentError("$name point $q lies outside its reference cell"))
        if D == 3
            abs(sum(converted) - one(T)) <= tolerance ||
                throw(ArgumentError("$name point $q is not barycentric"))
        end
        points[q] = converted

        weight = T(raw_weights[q])
        isfinite(weight) || throw(ArgumentError("$name contains a non-finite weight"))
        weight >= zero(T) || throw(ArgumentError("$name weights must be nonnegative"))
        weights[q] = weight
    end
    abs(sum(weights) - one(T)) <= tolerance ||
        throw(ArgumentError("$name weights must sum to one"))
    return _ReferenceQuadrature{D,T}(points, weights)
end

@inline _triangle_orbit3(a) = ((a, a, 1 - 2a), (a, 1 - 2a, a), (1 - 2a, a, a))
@inline _triangle_orbit6(a, b, c) =
    ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a))

# Three-point Gauss rule on [0,1], matching Gridap's degree-4 segment rule.
@inline function boundary_rule(::ReferenceFiniteElement)
    a = sqrt(3 / 5) / 2
    return ((0.5 - a,), (0.5,), (0.5 + a,)), (5 / 18, 4 / 9, 5 / 18)
end

# ---- Q1 reference quadrilateral (Ciarlet Q1) ---------------------------------

@inline nnode(::Q1ReferenceQuadrilateral) = 4

@inline shape(::Q1ReferenceQuadrilateral, r, s) =
    ((1 - r) * (1 - s), r * (1 - s), (1 - r) * s, r * s)

@inline function gradients(::Q1ReferenceQuadrilateral, r, s, hx, hy)
    gx = (-(1 - s) / hx, (1 - s) / hx, -s / hx, s / hx)
    gy = (-(1 - r) / hy, -r / hy, (1 - r) / hy, r / hy)
    return gx, gy
end

# 2×2 Gauss quadrature formula on [0,1]² (Ciarlet E_{\hat K}); degree 3
# in each coordinate, including Q1-Q1 products with a Q1 coefficient.
@inline function interior_rule(::Q1ReferenceQuadrilateral)
    a = inv(sqrt(3.0))
    lo = (1 - a) / 2
    hi = (1 + a) / 2
    return ((lo, lo), (hi, lo), (lo, hi), (hi, hi)), (0.25, 0.25, 0.25, 0.25)
end

# Positive degree-6 triangle rule (12 points, Witherden-Vincent), sufficient for
# Q1-Q1 products with a Q1 coefficient. Barycentric weights are normalized to 1.
@inline function cut_rule(::Q1ReferenceQuadrilateral)
    a = 0.06308901449150223
    b = 0.24928674517091043
    c, d, e = 0.05314504984481694, 0.3103524510337844, 0.6365024991213987
    w1, w2, w3 = 0.050844906370206832, 0.1167862757263794, 0.08285107561837358
    points = (_triangle_orbit3(a)..., _triangle_orbit3(b)...,
              _triangle_orbit6(c, d, e)...)
    return points, (w1, w1, w1, w2, w2, w2, w3, w3, w3, w3, w3, w3)
end

# Node lattice along one direction and lattice-unit → cell-unit scale. Q1 nodes
# sit on cell corners (integer offsets); Q2 nodes add edge midpoints and the
# cell center (half-integer offsets on the refined 2nx+1 lattice).
@inline node_grid(::Q1ReferenceQuadrilateral, nx::Int) = nx + 1
@inline lattice_scale(::Q1ReferenceQuadrilateral) = 1.0

# Cell-node ids on the background grid, in the canonical cell order
# (corners first, then edge midpoints, then the center).
@inline function cell_nodes(::Q1ReferenceQuadrilateral, cell::Int, nx::Int)
    i = (cell - 1) % nx + 1
    j = (cell - 1) ÷ nx + 1
    lower_left = i + (j - 1) * (nx + 1)
    return (lower_left, lower_left + 1,
            lower_left + nx + 1, lower_left + nx + 2)
end

# ---- Q2 reference quadrilateral (Ciarlet Q2) ---------------------------------

@inline nnode(::Q2ReferenceQuadrilateral) = 9

# Tensor product of 1D quadratic Lagrange polynomials on {0, 1/2, 1}.
@inline _l2_0(x) = 2x^2 - 3x + 1
@inline _l2_1(x) = -4x^2 + 4x
@inline _l2_2(x) = 2x^2 - x
@inline _dl2_0(x) = 4x - 3
@inline _dl2_1(x) = -8x + 4
@inline _dl2_2(x) = 4x - 1

# (r-index, s-index) per canonical cell-node order: 4 corners, 4 edge
# midpoints (bottom, top, left, right), cell center.
const _Q2_NODE_ORDER = ((1, 1), (3, 1), (1, 3), (3, 3),
                        (2, 1), (2, 3), (1, 2), (3, 2), (2, 2))

@inline function shape(::Q2ReferenceQuadrilateral, r, s)
    lr = (_l2_0(r), _l2_1(r), _l2_2(r))
    ls = (_l2_0(s), _l2_1(s), _l2_2(s))
    return ntuple(k -> lr[_Q2_NODE_ORDER[k][1]] * ls[_Q2_NODE_ORDER[k][2]], 9)
end

@inline function gradients(::Q2ReferenceQuadrilateral, r, s, hx, hy)
    lr = (_l2_0(r), _l2_1(r), _l2_2(r))
    ls = (_l2_0(s), _l2_1(s), _l2_2(s))
    dlr = (_dl2_0(r), _dl2_1(r), _dl2_2(r))
    dls = (_dl2_0(s), _dl2_1(s), _dl2_2(s))
    gx = ntuple(k -> dlr[_Q2_NODE_ORDER[k][1]] * ls[_Q2_NODE_ORDER[k][2]] / hx, 9)
    gy = ntuple(k -> lr[_Q2_NODE_ORDER[k][1]] * dls[_Q2_NODE_ORDER[k][2]] / hy, 9)
    return gx, gy
end

# 3×3 Gauss quadrature formula on [0,1]² (Ciarlet E_{\hat K}); degree 5
# in each coordinate, including Q2-Q2 products with a Q1 coefficient.
@inline function interior_rule(::Q2ReferenceQuadrilateral)
    a = (1 - sqrt(3 / 5)) / 2
    b = (1 + sqrt(3 / 5)) / 2
    points = ((a, a), (a, 0.5), (a, b),
              (0.5, a), (0.5, 0.5), (0.5, b),
              (b, a), (b, 0.5), (b, b))
    wr = (5 / 18, 4 / 9, 5 / 18)
    weights = ntuple(k -> wr[((k - 1) ÷ 3) + 1] * wr[((k - 1) % 3) + 1], 9)
    return points, weights
end

# Positive degree-10 triangle rule (25 points, Witherden-Vincent), sufficient
# for Q2-Q2 products with a Q1 coefficient over affine cut subtriangles.
@inline function cut_rule(::Q2ReferenceQuadrilateral)
    a = 0.03205537321694352
    b = 0.14216110105656438
    c1, d1, e1 = 0.32181299528883545, 0.14813288578382056, 0.530054118927344
    c2, d2, e2 = 0.02961988948872979, 0.369146781827811, 0.6012333286834592
    c3, d3, e3 = 0.028367665339938453, 0.1637017337371825, 0.807930600922879
    w0 = 0.08174332914628598
    w1, w2 = 0.01335296881314957, 0.04595796360474474
    w3, w4, w5 = 0.06390490639642406, 0.034184648162959436, 0.025297757707288392
    points = ((1 / 3, 1 / 3, 1 / 3),
              _triangle_orbit3(a)..., _triangle_orbit3(b)...,
              _triangle_orbit6(c1, d1, e1)...,
              _triangle_orbit6(c2, d2, e2)...,
              _triangle_orbit6(c3, d3, e3)...)
    weights = (w0,
               w1, w1, w1, w2, w2, w2,
               w3, w3, w3, w3, w3, w3,
               w4, w4, w4, w4, w4, w4,
               w5, w5, w5, w5, w5, w5)
    return points, weights
end

@inline node_grid(::Q2ReferenceQuadrilateral, nx::Int) = 2nx + 1
@inline lattice_scale(::Q2ReferenceQuadrilateral) = 0.5

@inline function cell_nodes(::Q2ReferenceQuadrilateral, cell::Int, nx::Int)
    nodes_x = 2nx + 1
    i = (cell - 1) % nx + 1
    j = (cell - 1) ÷ nx + 1
    node(i, j) = i + (j - 1) * nodes_x
    return (node(2i - 1, 2j - 1), node(2i + 1, 2j - 1),
            node(2i - 1, 2j + 1), node(2i + 1, 2j + 1),
            node(2i, 2j - 1), node(2i, 2j + 1),
            node(2i - 1, 2j), node(2i + 1, 2j), node(2i, 2j))
end
