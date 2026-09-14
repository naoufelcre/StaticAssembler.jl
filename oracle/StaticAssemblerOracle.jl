# StaticAssembler oracle support: Kelvin-Voigt and Stokes PDEs on cut domains,
# Gridap reference builders, and StaticSystem equivalence helpers.
#
# NOT part of the StaticAssembler library. Included by the test suite;
# StaticAssembler itself contains no PDE or material model. Everything here
# targets cut domains: the equivalence harness compares the projected
# `StaticSystem` operator `B` against Gridap AgFEM reference operators.

module StaticAssemblerOracle

using Gridap
using GridapEmbedded
using Gridap.TensorValues
using Gridap.Geometry: get_cell_map, get_glue, get_node_coordinates
using Gridap.CellData: get_cell_quadrature
using Gridap.Fields: evaluate
using LinearAlgebra
using SparseArrays

using StaticAssembler
# Gridap adapter (extension); loaded since Gridap/GridapEmbedded are in scope.
const StaticAssemblerGridapExt = Base.get_extension(StaticAssembler, :StaticAssemblerGridapExt)

export KelvinVoigtParameters, kelvin_voigt_form, density, strain_components,
    build_cut_reference, build_cut_stokes_reference, _errors,
    cut_equivalence_errors, cut_stokes_equivalence_errors,
    cut_agfem_operator, cut_stokes_agfem_operator,
    stable_to_master, agfem_compact_map, naive_quadrature,
    StokesParameters, stokes_form, ArticleParameters, article_form,
    article_equivalence_errors
export identity_op, grad_op, symgrad_op, div_op, scalar_value_op,
    normal_op, tangent_op

# Physics-side differential operators: relocated from the engine, which now ships
# only the `AbstractDifferentialOperator` contract. The local vector DOF
# ordering these pairings assume is component-blocked: rows 1:N = x component
# at nodes 1:N, N+1:2N = y component at nodes 1:N, with N = length(shape)
# derived from the reference-basis data.
struct IdentityOp <: StaticAssembler.AbstractDifferentialOperator end
struct GradOp <: StaticAssembler.AbstractDifferentialOperator end
struct SymGradOp <: StaticAssembler.AbstractDifferentialOperator end
struct DivOp <: StaticAssembler.AbstractDifferentialOperator end
struct ScalarValueOp <: StaticAssembler.AbstractDifferentialOperator end
struct NormalOp <: StaticAssembler.AbstractDifferentialOperator end
struct TangentOp <: StaticAssembler.AbstractDifferentialOperator end

identity_op() = IdentityOp()
grad_op() = GradOp()
symgrad_op() = SymGradOp()
div_op() = DivOp()
scalar_value_op() = ScalarValueOp()
normal_op() = NormalOp()
tangent_op() = TangentOp()

@inline _row_node(row::Int, nnode::Int) = row <= nnode ? row : row - nnode
@inline _row_comp(row::Int, nnode::Int) = row <= nnode ? 1 : 2

# Contraction of two local operator rows: only same-component rows pair.
@inline function StaticAssembler._op_pair(::IdentityOp, ::IdentityOp, row::Int,
                                          col::Int, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c)
    nnode = length(sh_t)
    _row_comp(row, nnode) == _row_comp(col, nnode) || return zero(first(sh_t))
    return sh_t[_row_node(row, nnode)] * sh_c[_row_node(col, nnode)]
end

@inline function StaticAssembler._op_pair(::GradOp, ::GradOp, row::Int, col::Int,
                                          sh_t, gx_t, gy_t, sh_c, gx_c, gy_c)
    nnode = length(sh_t)
    _row_comp(row, nnode) == _row_comp(col, nnode) || return zero(first(sh_t))
    rnode = _row_node(row, nnode)
    cnode = _row_node(col, nnode)
    return gx_t[rnode] * gx_c[cnode] + gy_t[rnode] * gy_c[cnode]
end

@inline function StaticAssembler._op_pair(::SymGradOp, ::SymGradOp, row::Int,
                                          col::Int, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c)
    nnode = length(sh_t)
    rnode = _row_node(row, nnode)
    cnode = _row_node(col, nnode)
    # D(N e_x) = (gx, 0, gy/2), D(N e_y) = (0, gy, gx/2); cross-component
    # pairs share the xy entry: D(N e_x):D(N e_y) = gy·gx/2.
    if row <= nnode
        if col <= nnode
            return gx_t[rnode] * gx_c[cnode] + (gy_t[rnode] * gy_c[cnode]) / 2
        else
            return (gy_t[rnode] * gx_c[cnode]) / 2
        end
    else
        if col <= nnode
            return (gx_t[rnode] * gy_c[cnode]) / 2
        else
            return gy_t[rnode] * gy_c[cnode] + (gx_t[rnode] * gx_c[cnode]) / 2
        end
    end
end

# q div(u): test scalar q (row) against the divergence of trial velocity (col).
@inline function StaticAssembler._op_pair(::IdentityOp, ::DivOp, row::Int,
                                          col::Int, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c)
    nnode_c = length(sh_c)
    return sh_t[row] * (col <= nnode_c ? gx_c[col] : gy_c[col - nnode_c])
end

# div(v) p: test velocity divergence (row) against trial scalar p (col).
@inline function StaticAssembler._op_pair(::DivOp, ::IdentityOp, row::Int,
                                          col::Int, sh_t, gx_t, gy_t, sh_c, gx_c, gy_c)
    nnode_t = length(sh_t)
    return (row <= nnode_t ? gx_t[row] : gy_t[row - nnode_t]) * sh_c[col]
end

@inline function StaticAssembler._op_pair(::ScalarValueOp, ::ScalarValueOp,
                                          row::Int, col::Int,
                                          sh_t, gx_t, gy_t, sh_c, gx_c, gy_c)
    return sh_t[row] * sh_c[col]
end

@inline function StaticAssembler._boundary_op_pair(::NormalOp, ::NormalOp,
                                                    row::Int, col::Int,
                                                    sh_t, gx_t, gy_t,
                                                    sh_c, gx_c, gy_c, point)
    nnode_t = length(sh_t)
    nnode_c = length(sh_c)
    normal = StaticAssembler.boundary_normal(point)
    return normal[_row_comp(row, nnode_t)] * sh_t[_row_node(row, nnode_t)] *
           normal[_row_comp(col, nnode_c)] * sh_c[_row_node(col, nnode_c)]
end

@inline function StaticAssembler._boundary_op_pair(::TangentOp, ::TangentOp,
                                                    row::Int, col::Int,
                                                    sh_t, gx_t, gy_t,
                                                    sh_c, gx_c, gy_c, point)
    nnode_t = length(sh_t)
    nnode_c = length(sh_c)
    tangent = StaticAssembler.boundary_tangent(point)
    return tangent[_row_comp(row, nnode_t)] * sh_t[_row_node(row, nnode_t)] *
           tangent[_row_comp(col, nnode_c)] * sh_c[_row_node(col, nnode_c)]
end

@inline function StaticAssembler._rhs_contract!(fe, ::IdentityOp, slot, shape,
                                                gx, gy, c, w, offset::Int)
    src = slot.source(c)
    nnode = length(shape)
    @inbounds for row in 1:2nnode
        node = _row_node(row, nnode)
        v = _row_comp(row, nnode) == 1 ? src[1] : src[2]
        fe[offset + row] += slot.scale * v * shape[node] * w
    end
    return nothing
end

@inline function StaticAssembler._rhs_contract!(fe, ::SymGradOp, slot, shape,
                                                gx, gy, c, w, offset::Int)
    σ = slot.source(c)
    σ11, σ22, σ12 = σ[1, 1], σ[2, 2], σ[1, 2]
    nnode = length(shape)
    @inbounds for row in 1:2nnode
        node = _row_node(row, nnode)
        if row <= nnode
            fe[offset + row] += slot.scale * (σ11 * gx[node] + σ12 * gy[node]) * w
        else
            fe[offset + row] += slot.scale * (σ22 * gy[node] + σ12 * gx[node]) * w
        end
    end
    return nothing
end

@inline function StaticAssembler._rhs_contract!(fe, ::GradOp, slot, shape,
                                                gx, gy, c, w, offset::Int)
    F = slot.source(c)
    nnode = length(shape)
    @inbounds for row in 1:2nnode
        node = _row_node(row, nnode)
        if row <= nnode
            fe[offset + row] += slot.scale * (F[1, 1] * gx[node] + F[1, 2] * gy[node]) * w
        else
            fe[offset + row] += slot.scale * (F[2, 1] * gx[node] + F[2, 2] * gy[node]) * w
        end
    end
    return nothing
end


@inline function StaticAssembler._rhs_contract!(fe, ::ScalarValueOp, slot,
                                                shape, gx, gy, c, w, offset::Int)
    value = slot.scale * slot.source(c) * w
    @inbounds for row in eachindex(shape)
        fe[offset + row] += value * shape[row]
    end
    return nothing
end


@inline function StaticAssembler._boundary_rhs_contract!(
    fe, ::NormalOp, slot, shape, gx, gy, point, w, offset::Int)
    value = slot.scale * slot.source(point) * w
    normal = StaticAssembler.boundary_normal(point)
    nnode = length(shape)
    @inbounds for row in 1:2nnode
        fe[offset + row] += value * normal[_row_comp(row, nnode)] *
                            shape[_row_node(row, nnode)]
    end
    return nothing
end


@inline function StaticAssembler._boundary_rhs_contract!(
    fe, ::TangentOp, slot, shape, gx, gy, point, w, offset::Int)
    value = slot.scale * slot.source(point) * w
    tangent = StaticAssembler.boundary_tangent(point)
    nnode = length(shape)
    @inbounds for row in 1:2nnode
        fe[offset + row] += value * tangent[_row_comp(row, nnode)] *
                            shape[_row_node(row, nnode)]
    end
    return nothing
end


# Kelvin-Voigt variational form written once as a StaticAssembler
# ElementVariationalForm. The element tables are generated from this form by
# StaticAssembler; the hand-written Gridap reference in this file remains the
# independent oracle for equivalence tests.
"""
    kelvin_voigt_form(parameters) -> ElementVariationalForm

`a(u,v) = ∫ u·v + ξ D(u):D(v)` and `l(v) = -∫ σ(ε,ρ):D(v)` with the active
stress `A (1/(ρ-0.2) - 0.5) I`, consuming `(ρ, ε₁₁, ε₂₂, ε₁₂)`.
"""
struct KelvinVoigtStress{P}
    p::P
end

function (s::KelvinVoigtStress)(c)
    ρ, e11, e22, e12 = c
    p = s.p
    trace_e = e11 + e22
    active = p.A * (inv(ρ - 0.2) - 0.5)
    σ11 = (1.0 - 2p.ν) * e11 + p.ν * trace_e + active
    σ22 = (1.0 - 2p.ν) * e22 + p.ν * trace_e + active
    σ12 = (1.0 - 2p.ν) * e12
    return SymTensorValue(σ11, σ12, σ22)
end

function kelvin_voigt_form(parameters; element=StaticAssembler.Q1ReferenceQuadrilateral())
    space = StaticAssembler.FiniteElement(element, 2)
    return StaticAssembler.ElementVariationalForm(
        space,
        (StaticAssembler.BilinearSlot(identity_op(), identity_op(),
                                 StaticAssembler.ConstantCoefficient(1.0)),
         StaticAssembler.BilinearSlot(symgrad_op(), symgrad_op(),
                                 StaticAssembler.ConstantCoefficient(parameters.ξ))),
        (StaticAssembler.RhsSlot(symgrad_op(),
                            KelvinVoigtStress(parameters), -1.0),),
        4)
end

# Taylor-Hood Q2/Q1 Stokes: test and trial spaces are the velocity (Q2,
# vector) and pressure (Q1, scalar) blocks, all on the same background grid.
function stokes_form(parameters)
    velocity = StaticAssembler.FiniteElement(StaticAssembler.Q2ReferenceQuadrilateral(), 2)
    pressure = StaticAssembler.FiniteElement(StaticAssembler.Q1ReferenceQuadrilateral(), 1)
    μ = parameters.μ
    return StaticAssembler.ElementVariationalForm(
        (velocity, pressure),
        (velocity, pressure),
        (StaticAssembler.BilinearSlot{1,1}(grad_op(), grad_op(),
                                      StaticAssembler.ConstantCoefficient(μ)),
         StaticAssembler.BilinearSlot{2,1}(identity_op(), div_op(),
                                      StaticAssembler.ConstantCoefficient(-1.0)),
         StaticAssembler.BilinearSlot{1,2}(div_op(), identity_op(),
                                      StaticAssembler.ConstantCoefficient(-1.0))),
        (StaticAssembler.RhsSlot{1}(identity_op(),
                               c -> VectorValue(0.0, -1.0), 1.0),),
        0)
end

Base.@kwdef struct ArticleParameters{T <: AbstractFloat}
    η::T = T(0.8)
    ξ::T = T(0.07)
    Δt::T = T(0.2)
    D::T = T(0.03)
    a::T = T(0.4)
    asign::T = T(-0.6)
    σmax::T = T(0.5)
    Kb::T = T(0.5)
    nH::Int = 4
    γ0::T = T(0.9)
    γmax::T = T(0.4)
    Ks::T = T(0.5)
    mH::Int = 4
    mar::T = T(0.7)
    β::T = T(2.1)
    hΣ::T = T(0.15)
    slip::T = T(0.12)
    fric::T = T(0.35)
end

# Hill constitutive laws shared by the StaticAssembler transcription and the
# Gridap oracle, mirroring `Article_Cells/src/VarForms.jl` (`ζ̄`, `ζ′`, `γ`, `γ′`).
@inline _hill_zeta_prime(c, σmax, Kb, nH) =
    σmax * nH * Kb^nH * c^(nH - 1) / (Kb^nH + c^nH)^2
@inline _hill_gamma(c, γ0, γmax, Ks, mH) =
    γ0 + γmax * c^mH / (Ks^mH + c^mH)
@inline _hill_gamma_prime(c, γmax, Ks, mH) =
    γmax * mH * Ks^mH * c^(mH - 1) / (Ks^mH + c^mH)^2

"""Q2/Q1/Q1 Brinkman-marker form matching `Article_Cells/src/VarForms.jl`."""
function article_form(p::ArticleParameters)
    velocity = StaticAssembler.FiniteElement(
        StaticAssembler.Q2ReferenceQuadrilateral(), 2)
    pressure = StaticAssembler.FiniteElement(
        StaticAssembler.Q1ReferenceQuadrilateral(), 1)
    marker = StaticAssembler.FiniteElement(
        StaticAssembler.Q1ReferenceQuadrilateral(), 1)
    volume_bilinear = (
        StaticAssembler.BilinearSlot{1,1}(
            identity_op(), identity_op(), StaticAssembler.ConstantCoefficient(p.η)),
        StaticAssembler.BilinearSlot{1,1}(
            grad_op(), grad_op(), StaticAssembler.ConstantCoefficient(p.ξ)),
        StaticAssembler.BilinearSlot{1,2}(
            div_op(), identity_op(), StaticAssembler.ConstantCoefficient(-1.0)),
        StaticAssembler.BilinearSlot{2,1}(
            identity_op(), div_op(), StaticAssembler.ConstantCoefficient(1.0)),
        StaticAssembler.BilinearSlot{3,3}(
            scalar_value_op(), scalar_value_op(),
            StaticAssembler.ConstantCoefficient(inv(p.Δt))),
        StaticAssembler.BilinearSlot{3,3}(
            grad_op(), grad_op(), StaticAssembler.ConstantCoefficient(p.D)),
    )
    active_force = point -> begin
        gx, gy = StaticAssembler.coefficient_gradient(point, 1)
        scale = p.asign * _hill_zeta_prime(point[1], p.σmax, p.Kb, p.nH)
        VectorValue(scale * gx, scale * gy)
    end
    volume_rhs = (
        StaticAssembler.RhsSlot{1}(identity_op(), active_force, 1.0),
        StaticAssembler.RhsSlot{3}(scalar_value_op(), point -> point[2], inv(p.Δt)),
    )
    marker_flux = StaticAssembler.StateFunction() do point
        nx, ny = StaticAssembler.boundary_normal(point)
        p.a * StaticAssembler.facet_value(point, 2) *
        (point[3] * nx + point[4] * ny)
    end
    normal_penalty = StaticAssembler.StateFunction(
        point -> p.β / p.hΣ * StaticAssembler.facet_value(point, 3))
    tangent_penalty = StaticAssembler.StateFunction(
        point -> p.slip * StaticAssembler.facet_value(point, 3))
    boundary_bilinear = (
        StaticAssembler.BilinearSlot{3,3}(
            scalar_value_op(), scalar_value_op(), marker_flux),
        StaticAssembler.BilinearSlot{1,1}(
            normal_op(), normal_op(), normal_penalty),
        StaticAssembler.BilinearSlot{1,1}(
            tangent_op(), tangent_op(), tangent_penalty),
    )
    normal_traction = point -> begin
        κ = StaticAssembler.facet_value(point, 1)
        χΓ = StaticAssembler.facet_value(point, 2)
        χΣ = StaticAssembler.facet_value(point, 3)
        Fn = StaticAssembler.facet_value(point, 4)
        gΣ = StaticAssembler.facet_value(point, 6)
        -χΓ * _hill_gamma(point[1], p.γ0, p.γmax, p.Ks, p.mH) * κ + Fn +
            χΣ * p.β / p.hΣ * gΣ
    end
    tangent_traction = point -> begin
        χΓ = StaticAssembler.facet_value(point, 2)
        Ft = StaticAssembler.facet_value(point, 5)
        tx, ty = StaticAssembler.boundary_tangent(point)
        gx, gy = StaticAssembler.coefficient_gradient(point, 1)
        χΓ * p.mar * _hill_gamma_prime(point[1], p.γmax, p.Ks, p.mH) *
            (gx * tx + gy * ty) + p.fric * Ft
    end
    boundary_rhs = (
        StaticAssembler.RhsSlot{1}(normal_op(), normal_traction, 1.0),
        StaticAssembler.RhsSlot{1}(tangent_op(), tangent_traction, 1.0),
    )
    q1 = StaticAssembler.Q1ReferenceQuadrilateral()
    q2 = StaticAssembler.Q2ReferenceQuadrilateral()
    return StaticAssembler.ElementVariationalForm(
        (velocity, pressure, marker), (velocity, pressure, marker),
        volume_bilinear, volume_rhs, 4;
        coefficient_elements=(q1, q1, q2, q2),
        coefficient_gradients=true,
        boundary_bilinear, boundary_rhs, nfacetfields=6)
end

Base.@kwdef struct KelvinVoigtParameters{T <: AbstractFloat}
    "Far-field viscous parameter ξ"
    ξ::T = T(0.001)
    "Active stress amplitude A"
    A::T = T(0.01)
    "Poisson ratio ν"
    ν::T = T(0.48)
end

@inline density(x, y) = 0.70 + 0.05 * x - 0.03 * y

function _validate_parameters(parameters)
    parameters isa KelvinVoigtParameters{Float64} ||
        throw(ArgumentError("parameters must use Float64"))
    all(isfinite, (parameters.ξ, parameters.A, parameters.ν)) ||
        throw(ArgumentError("parameters must be finite"))
    return parameters
end

@inline function strain_components(x, y)
    return 0.03 + 0.01 * x,
           -0.02 + 0.015 * y,
           0.005 + 0.002 * x - 0.001 * y
end


struct CutGridapReference{M,C,A,O}
    nx::Int
    ny::Int
    parameters::KelvinVoigtParameters{Float64}
    hx::Float64
    hy::Float64
    model::M
    cut_geometry::C
    active_domain::A
    domain::O
end

Base.@kwdef struct StokesParameters{T <: AbstractFloat}
    "Viscosity"
    μ::T = T(1.0)
end

struct CutStokesGridapReference{M,C}
    nx::Int
    ny::Int
    parameters::StokesParameters{Float64}
    model::M
    cut_geometry::C
end

function build_cut_reference(
    nx::Int,
    ny::Int;
    radius::Real=0.42,
    center::Tuple{<:Real,<:Real}=(0.5, 0.5),
    domain::Tuple{<:Real,<:Real,<:Real,<:Real}=(0.0, 1.0, 0.0, 1.0),
    levelset_values=nothing,
    model=nothing,
    parameters=KelvinVoigtParameters{Float64}(),
)
    nx > 0 && ny > 0 || throw(ArgumentError("nx and ny must be positive"))
    radius > 0 || throw(ArgumentError("radius must be positive"))
    _validate_parameters(parameters)
    xmin, xmax, ymin, ymax = Float64.(domain)
    xmax > xmin && ymax > ymin || throw(ArgumentError("domain extents must be positive"))
    model = isnothing(model) ?
        CartesianDiscreteModel(Point(xmin, ymin), Point(xmax, ymax), (nx, ny)) : model
    if isnothing(levelset_values)
        geometry = disk(Float64(radius), x0=Point(Float64(center[1]), Float64(center[2])))
    else
        length(levelset_values) == (nx + 1) * (ny + 1) ||
            throw(ArgumentError("levelset_values must contain one value per grid node"))
        points = vec(collect(get_node_coordinates(model)))
        geometry = GridapEmbedded.LevelSetCutters.DiscreteGeometry(levelset_values, points)
    end
    cut_geometry = cut(model, geometry)
    active_domain = Triangulation(cut_geometry, ACTIVE)
    domain = Triangulation(cut_geometry, PHYSICAL)

    return CutGridapReference(
        nx, ny, parameters, (xmax - xmin) / nx, (ymax - ymin) / ny,
        model, cut_geometry, active_domain, domain,
    )
end

function build_cut_stokes_reference(
    nx::Int,
    ny::Int;
    radius::Real=0.42,
    center::Tuple{<:Real,<:Real}=(0.5, 0.5),
    domain::Tuple{<:Real,<:Real,<:Real,<:Real}=(0.0, 1.0, 0.0, 1.0),
    model=nothing,
    parameters=StokesParameters{Float64}(),
)
    nx > 0 && ny > 0 || throw(ArgumentError("nx and ny must be positive"))
    radius > 0 || throw(ArgumentError("radius must be positive"))
    all(isfinite, (parameters.μ,)) || throw(ArgumentError("parameters must be finite"))
    xmin, xmax, ymin, ymax = Float64.(domain)
    xmax > xmin && ymax > ymin || throw(ArgumentError("domain extents must be positive"))
    model = isnothing(model) ?
        CartesianDiscreteModel(Point(xmin, ymin), Point(xmax, ymax), (nx, ny)) : model
    geometry = disk(Float64(radius), x0=Point(Float64(center[1]), Float64(center[2])))
    cut_geometry = cut(model, geometry)
    return CutStokesGridapReference(
        nx, ny, parameters, model, cut_geometry,
    )
end

function _cut_quadrature_gridap(reference::CutGridapReference, ::Type{T},
                                degree::Int=6) where {T}
    quadrature = get_cell_quadrature(Measure(reference.domain, degree))
    cell_points = collect(quadrature.cell_point)
    cell_weights = collect(quadrature.cell_weight)
    physical_glue = get_glue(reference.domain, Val(2))
    active_glue = get_glue(reference.active_domain, Val(2))
    physical_bg_cells = Int.(collect(physical_glue.tface_to_mface))
    active_bg_cells = Int32.(collect(active_glue.tface_to_mface))
    reference_maps = collect(physical_glue.tface_to_mface_map)
    physical_maps = collect(get_cell_map(reference.domain))

    bg_to_active = zeros(Int32, reference.nx * reference.ny)
    for (active_cell, bg_cell) in enumerate(active_bg_cells)
        bg_to_active[bg_cell] = active_cell
    end

    counts = zeros(Int, length(active_bg_cells))
    for physical_cell in eachindex(physical_bg_cells)
        active_cell = bg_to_active[physical_bg_cells[physical_cell]]
        length(cell_points[physical_cell]) == 4 && continue
        counts[active_cell] += length(cell_points[physical_cell])
    end
    pointers = Vector{Int}(undef, length(counts) + 1)
    pointers[1] = 1
    for cell in eachindex(counts)
        pointers[cell + 1] = pointers[cell] + counts[cell]
    end

    nq = Int(pointers[end] - 1)
    reference_r = Vector{T}(undef, nq)
    reference_s = similar(reference_r)
    weights = similar(reference_r)
    cursor = copy(pointers[1:end-1])

    for physical_cell in eachindex(physical_bg_cells)
        active_cell = bg_to_active[physical_bg_cells[physical_cell]]
        points = cell_points[physical_cell]
        length(points) == 4 && continue
        reference_points = evaluate(reference_maps[physical_cell], points)
        jacobians = evaluate(∇(physical_maps[physical_cell]), points)
        for q in eachindex(points)
            index = cursor[active_cell]
            reference_r[index] = reference_points[q][1]
            reference_s[index] = reference_points[q][2]
            weights[index] = cell_weights[physical_cell][q] * meas(jacobians[q])
            cursor[active_cell] += 1
        end
    end

    data = StaticAssemblerGridapExt._cut_geometry_data(reference.cut_geometry, Float64)
    is_cut = BitVector(data.bg_states[bg_cell] == StaticAssembler.CUT for bg_cell in active_bg_cells)
    return StaticAssembler.CutQuadrature(
        pointers, is_cut, reference_r, reference_s, weights,
    ), active_bg_cells
end

function _gridap_interior_quadrature(degree::Int)
    quadrature = Gridap.ReferenceFEs.Quadrature(Gridap.ReferenceFEs.QUAD, degree)
    return Gridap.ReferenceFEs.get_coordinates(quadrature),
           Gridap.ReferenceFEs.get_weights(quadrature)
end

function _gridap_cut_quadrature(degree::Int)
    quadrature = Gridap.ReferenceFEs.Quadrature(Gridap.ReferenceFEs.TRI, degree)
    points = Gridap.ReferenceFEs.get_coordinates(quadrature)
    weights = Gridap.ReferenceFEs.get_weights(quadrature)
    return [(1 - point[1] - point[2], point[1], point[2]) for point in points],
           2 .* weights
end

function _analytic_coefficients(nx::Int, ny::Int, hx::Float64, hy::Float64,
                                ox::Float64, oy::Float64)
    rho = Vector{Float64}(undef, (nx + 1) * (ny + 1))
    e11 = similar(rho)
    e22 = similar(rho)
    e12 = similar(rho)
    for j in 0:ny, i in 0:nx
        id = i + 1 + j * (nx + 1)
        x, y = ox + i * hx, oy + j * hy
        rho[id] = density(x, y)
        e11[id], e22[id], e12[id] = strain_components(x, y)
    end
    return rho, e11, e22, e12
end

function _errors(matrix, vector, reference_matrix, reference_vector)
    matrix_delta = matrix - reference_matrix
    vector_delta = vector - reference_vector
    relative(delta, reference) = begin
        delta_norm = norm(delta)
        reference_norm = norm(reference)
        isfinite(delta_norm) && isfinite(reference_norm) || return Inf
        iszero(reference_norm) ? (iszero(delta_norm) ? 0.0 : Inf) : delta_norm / reference_norm
    end
    finite_maximum(values) = begin
        value = maximum(abs, values; init=0.0)
        isfinite(value) ? value : Inf
    end
    return (
        matrix_relative = relative(matrix_delta, reference_matrix),
        matrix_maximum = finite_maximum(nonzeros(matrix_delta)),
        vector_relative = relative(vector_delta, reference_vector),
        vector_maximum = finite_maximum(vector_delta),
    )
end

# ---- StaticSystem equivalence harness ---------------------------------------
#
# The harness compares the projected `StaticSystem` operator `B` against a
# Gridap AgFEM reference operator on the same cut geometry with the same
# aggregate roots. Gridap AgFEM DOFs are mapped to master columns in two
# steps: Gridap global DOF -> full-background stable row (element-specific
# local ordering, probed once against Gridap's Lagrangian elements), then
# stable row -> master column via the unit rows of `E` (element-agnostic).

function _record_stable!(stable, standard_dof, stable_dof)
    previous = stable[standard_dof]
    (previous == 0 || previous == stable_dof) ||
        error("local DOF ordering is inconsistent")
    stable[standard_dof] = stable_dof
    return nothing
end

# Q1 vector block: 8 local DOFs, component-major (x-block then y-block),
# corners in engine canonical order.
function _q1_vector_stable_map(cell_dofs, cell_nodes, nstandard)
    stable = zeros(Int, nstandard)
    for cell in eachindex(cell_dofs), local_dof in 1:8
        local_node = (local_dof - 1) % 4 + 1
        node = cell_nodes[local_node, cell]
        _record_stable!(stable, cell_dofs[cell][local_dof],
                        2node - 2 + (local_dof <= 4 ? 1 : 2))
    end
    return stable
end

# Q2 vector block: 18 local DOFs, component-major; scalar node order matches
# the engine canonical order (corners, edge midpoints, center).
function _q2_vector_stable_map(cell_dofs, cell_nodes, nstandard)
    stable = zeros(Int, nstandard)
    for cell in eachindex(cell_dofs), local_dof in 1:18
        local_node = (local_dof - 1) % 9 + 1
        node = cell_nodes[local_node, cell]
        _record_stable!(stable, cell_dofs[cell][local_dof],
                        2node - 2 + (local_dof <= 9 ? 1 : 2))
    end
    return stable
end

# Q1 scalar block (e.g. Stokes pressure): 4 local DOFs in engine canonical
# corner order, shifted by the block's E row offset.
function _q1_scalar_stable_map(cell_dofs, cell_nodes, nstandard, row_offset)
    stable = zeros(Int, nstandard)
    for cell in eachindex(cell_dofs), local_dof in 1:4
        node = cell_nodes[local_dof, cell]
        _record_stable!(stable, cell_dofs[cell][local_dof], row_offset + node)
    end
    return stable
end

# Full-background stable row -> master column via the unit rows of `E`.
# Master rows of E are unit rows by construction; slave and inactive rows
# are not. Element- and block-agnostic.
function stable_to_master(E::SparseMatrixCSC)
    ours = zeros(Int, size(E, 1))
    for row in 1:size(E, 1)
        cols, vals = findnz(E[row, :])
        if length(cols) == 1 && vals[1] == 1.0
            ours[row] = cols[1]
        end
    end
    return ours
end

# AgFEM-DOF -> stable-row map for single Q1 vector spaces whose operators are
# built outside the harness (e.g. custom physics in tests). Composes the Q1
# stable map with the space's master-to-standard DOF map.
function agfem_compact_map(standard_space, agfem_space, active_bg_cells, nx)
    lattice = StaticAssembler._cell_nodes(StaticAssembler.Q1ReferenceQuadrilateral(),
                                          Int32.(active_bg_cells), nx)
    standard = _q1_vector_stable_map(collect(get_cell_dof_ids(standard_space)),
                                     lattice, num_free_dofs(standard_space))
    return standard[Int.(getfield(agfem_space, :mDOF_to_DOF))]
end

# Gridap AgFEM reference operator for Kelvin-Voigt on a cut domain, following
# the GridapEmbedded per-field aggregation recipe: standard spaces on the
# active domain, one AgFEMSpace sharing `roots`, affine operator. Returns the
# matrix/vector plus the AgFEM-DOF -> stable-row map.
function cut_agfem_operator(reference::CutGridapReference, roots, values;
                             order::Int=1)
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2"))
    model, cutgeo = reference.model, reference.cut_geometry
    parameters = reference.parameters
    nx = reference.nx
    vector_fe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    scalar_fe = ReferenceFE(lagrangian, Float64, 1)
    background_scalar = FESpace(model, scalar_fe, conformity=:H1)
    active = Triangulation(cutgeo, ACTIVE)
    physical = Triangulation(cutgeo, PHYSICAL)
    vector_standard = TestFESpace(active, vector_fe, conformity=:H1)
    scalar_standard = TestFESpace(active, scalar_fe, conformity=:H1)
    vector_test = AgFEMSpace(vector_standard, roots)
    vector_trial = TrialFESpace(vector_test)
    scalar_space = AgFEMSpace(scalar_standard, roots)
    rho = interpolate(FEFunction(background_scalar, values[1]), scalar_space)
    e11 = interpolate(FEFunction(background_scalar, values[2]), scalar_space)
    e22 = interpolate(FEFunction(background_scalar, values[3]), scalar_space)
    e12 = interpolate(FEFunction(background_scalar, values[4]), scalar_space)
    strain = Operation((a, b, c) -> SymTensorValue(a, c, b))(e11, e22, e12)
    identity_tensor = SymTensorValue(1.0, 0.0, 1.0)
    D(v) = symmetric_gradient(v)
    elastic(e) = (1.0 - 2parameters.ν) * e +
                 parameters.ν * tr(e) * identity_tensor
    active_stress = parameters.A *
                    (1.0 / (rho - 0.2) - 0.5) * identity_tensor
    measure = Measure(physical, order == 1 ? 6 : 10)
    a(u, v) = integrate(u ⋅ v + parameters.ξ * (D(u) ⊙ D(v)), measure)
    l(v) = integrate(-(elastic(strain) + active_stress) ⊙ D(v), measure)
    operator = AffineFEOperator(a, l, vector_trial, vector_test)
    active_bg_cells = collect(get_glue(active, Val(2)).tface_to_mface)
    standard_dofs = collect(get_cell_dof_ids(vector_standard))
    if order == 1
        lattice = StaticAssembler._cell_nodes(StaticAssembler.Q1ReferenceQuadrilateral(),
                                              Int32.(active_bg_cells), nx)
        standard = _q1_vector_stable_map(standard_dofs, lattice,
                                         num_free_dofs(vector_standard))
    else
        lattice = StaticAssembler._cell_nodes(StaticAssembler.Q2ReferenceQuadrilateral(),
                                              Int32.(active_bg_cells), nx)
        standard = _q2_vector_stable_map(standard_dofs, lattice,
                                         num_free_dofs(vector_standard))
    end
    compact_to_stable = standard[Int.(getfield(vector_test, :mDOF_to_DOF))]
    return (; matrix=get_matrix(operator), vector=get_vector(operator),
            compact_to_stable, standard, vector_standard, vector_test)
end

# Gridap AgFEM reference operator for Stokes on a cut domain: per-field
# aggregation shared across fields, multifield combination afterwards (the
# GridapEmbedded StokesAgFEM recipe). Returns the multifield matrix/vector
# plus the concatenated AgFEM-DOF -> stable-row map (velocity block first,
# matching both Gridap multifield numbering and E block order).
function cut_stokes_agfem_operator(reference::CutStokesGridapReference, roots)
    model, cutgeo = reference.model, reference.cut_geometry
    μ = reference.parameters.μ
    nx, ny = reference.nx, reference.ny
    active = Triangulation(cutgeo, ACTIVE)
    physical = Triangulation(cutgeo, PHYSICAL)
    velocity_fe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
    pressure_fe = ReferenceFE(lagrangian, Float64, 1)
    velocity_standard = TestFESpace(active, velocity_fe, conformity=:H1)
    pressure_standard = TestFESpace(active, pressure_fe, conformity=:H1)
    velocity_test = AgFEMSpace(velocity_standard, roots)
    pressure_test = AgFEMSpace(pressure_standard, roots)
    velocity_trial = TrialFESpace(velocity_test)
    pressure_trial = TrialFESpace(pressure_test)
    measure = Measure(physical, 10)
    bilinear((u, p), (v, q)) = integrate(
        μ * (∇(u) ⊙ ∇(v)) - q * tr(∇(u)) - p * tr(∇(v)), measure)
    linear((v, q)) = integrate(VectorValue(0.0, -1.0) ⋅ v, measure)
    Y = MultiFieldFESpace([velocity_test, pressure_test])
    X = MultiFieldFESpace([velocity_trial, pressure_trial])
    operator = AffineFEOperator(bilinear, linear, X, Y)
    active_bg_cells = collect(get_glue(active, Val(2)).tface_to_mface)
    q2lattice = StaticAssembler._cell_nodes(StaticAssembler.Q2ReferenceQuadrilateral(),
                                            Int32.(active_bg_cells), nx)
    vstandard = _q2_vector_stable_map(
        collect(get_cell_dof_ids(velocity_standard)), q2lattice,
        num_free_dofs(velocity_standard))
    vcompact = vstandard[Int.(getfield(velocity_test, :mDOF_to_DOF))]
    pressure_offset = 2 * ((2nx + 1) * (2ny + 1))
    q1lattice = StaticAssembler._cell_nodes(StaticAssembler.Q1ReferenceQuadrilateral(),
                                            Int32.(active_bg_cells), nx)
    pstandard = _q1_scalar_stable_map(
        collect(get_cell_dof_ids(pressure_standard)), q1lattice,
        num_free_dofs(pressure_standard), pressure_offset)
    pcompact = pstandard[Int.(getfield(pressure_test, :mDOF_to_DOF))]
    compact_to_stable = vcat(vcompact, pcompact)
    matrix, vector = get_matrix(operator), get_vector(operator)
    length(compact_to_stable) == size(matrix, 1) ||
        error("multifield AgFEM DOF count does not match reference matrix")
    return (; matrix, vector, compact_to_stable)
end

@inline _article_cprev(x, y) = 0.8 + 0.2x - 0.1y + 0.05x*y
@inline _article_cstar(x, y) = 0.7 - 0.04x + 0.08y + 0.03x*y
@inline _article_vx(x, y) = 0.15 + 0.2x - 0.1y + 0.07x*y + 0.03x^2
@inline _article_vy(x, y) = -0.12 + 0.06x + 0.18y - 0.04x*y + 0.02y^2

function _article_data(reference::CutGridapReference)
    nx, ny = reference.nx, reference.ny
    coordinates = vec(collect(get_node_coordinates(reference.model)))
    cprev = [_article_cprev(x[1], x[2]) for x in coordinates]
    cstar = [_article_cstar(x[1], x[2]) for x in coordinates]
    origin = first(coordinates)
    hx, hy = reference.hx, reference.hy
    vx = Float64[]
    vy = Float64[]
    for j in 0:2ny, i in 0:2nx
        x = origin[1] + i * hx / 2
        y = origin[2] + j * hy / 2
        push!(vx, _article_vx(x, y))
        push!(vy, _article_vy(x, y))
    end
    data = StaticAssemblerGridapExt._cut_geometry_data(
        reference.cut_geometry, Float64)
    nfacets = length(data.boundary.facet_to_bgcell)
    κ = [0.9 + 0.01i for i in 1:nfacets]
    χΣ = [i % 3 == 0 ? 1.0 : 0.0 for i in 1:nfacets]
    χΓ = [1.0 - value for value in χΣ]
    Fn = [-0.02 + 0.001i for i in 1:nfacets]
    Ft = [0.015 - 0.0005i for i in 1:nfacets]
    gΣ = [0.03 + 0.0007i for i in 1:nfacets]
    return (; coefficients=(cprev, cstar, vx, vy),
            facet_fields=(κ, χΓ, χΣ, Fn, Ft, gΣ))
end

function _article_gridap_operator(reference::CutGridapReference, roots,
                                  data, prm::ArticleParameters)
    model, cutgeo = reference.model, reference.cut_geometry
    nx, ny = reference.nx, reference.ny
    active = Triangulation(cutgeo, ACTIVE)
    physical = Triangulation(cutgeo, PHYSICAL)
    boundary = EmbeddedBoundary(cutgeo)
    velocity_fe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
    scalar_fe = ReferenceFE(lagrangian, Float64, 1)
    velocity_standard = TestFESpace(active, velocity_fe, conformity=:H1)
    pressure_standard = TestFESpace(active, scalar_fe, conformity=:H1)
    marker_standard = TestFESpace(active, scalar_fe, conformity=:H1)
    velocity_test = AgFEMSpace(velocity_standard, roots)
    pressure_test = AgFEMSpace(pressure_standard, roots)
    marker_test = AgFEMSpace(marker_standard, roots)
    velocity_trial = TrialFESpace(velocity_test)
    pressure_trial = TrialFESpace(pressure_test)
    marker_trial = TrialFESpace(marker_test)

    background_scalar = FESpace(model, scalar_fe, conformity=:H1)
    background_velocity = FESpace(model, velocity_fe, conformity=:H1)
    cprev = interpolate(
        FEFunction(background_scalar, data.coefficients[1]), marker_test)
    cstar = interpolate(
        FEFunction(background_scalar, data.coefficients[2]), marker_test)
    vprev = interpolate(
        interpolate(x -> VectorValue(
            _article_vx(x[1], x[2]), _article_vy(x[1], x[2])),
            background_velocity), velocity_test)
    κ, χΓ, χΣ, Fn, Ft, gΣ =
        (CellField(values, boundary) for values in data.facet_fields)
    normal = get_normal_vector(boundary)
    # `normal ⋅ T` evaluates to the counter-clockwise tangent `(-n_y, n_x)`
    # matching `StaticAssembler.boundary_tangent`. Gridap stores `TensorValue`
    # column-major, so the reversed order `T ⋅ normal` evaluates to the
    # opposite (clockwise) tangent and would flip the single-tangent `Ft` term.
    tangent = normal ⋅ TensorValue(0.0, -1.0, 1.0, 0.0)
    dΩ = Measure(physical, 4)
    dΓ = Measure(boundary, 4)
    ζ′ = c -> _hill_zeta_prime(c, prm.σmax, prm.Kb, prm.nH)
    γ = c -> _hill_gamma(c, prm.γ0, prm.γmax, prm.Ks, prm.mH)
    γ′ = c -> _hill_gamma_prime(c, prm.γmax, prm.Ks, prm.mH)

    function bilinear((v, pressure, marker), (w, q, φ))
        bulk = prm.η * (v ⋅ w) + prm.ξ * (∇(v) ⊙ ∇(w)) -
               pressure * tr(∇(w)) + q * tr(∇(v)) +
               marker * φ / prm.Δt + prm.D * (∇(marker) ⋅ ∇(φ))
        surface = χΓ * prm.a * (vprev ⋅ normal) * marker * φ +
                  χΣ * (prm.β / prm.hΣ) * (v ⋅ normal) * (w ⋅ normal) +
                  χΣ * prm.slip * (v ⋅ tangent) * (w ⋅ tangent)
        return ∫(bulk)dΩ + ∫(surface)dΓ
    end
    function linear((w, q, φ))
        bulk = prm.asign * (ζ′ ∘ cprev) * (∇(cprev) ⋅ w) + cstar * φ / prm.Δt
        normal_force = -χΓ * (γ ∘ cprev) * κ + Fn + χΣ * (prm.β / prm.hΣ) * gΣ
        tangent_force = χΓ * prm.mar * (γ′ ∘ cprev) *
                        (∇(cprev) ⋅ tangent) + prm.fric * Ft
        return ∫(bulk)dΩ +
               ∫(normal_force * (w ⋅ normal) +
                 tangent_force * (w ⋅ tangent))dΓ
    end
    Y = MultiFieldFESpace([velocity_test, pressure_test, marker_test])
    X = MultiFieldFESpace([velocity_trial, pressure_trial, marker_trial])
    operator = AffineFEOperator(bilinear, linear, X, Y)

    active_bg_cells = collect(get_glue(active, Val(2)).tface_to_mface)
    q2lattice = StaticAssembler._cell_nodes(
        StaticAssembler.Q2ReferenceQuadrilateral(), Int32.(active_bg_cells), nx)
    vstandard = _q2_vector_stable_map(
        collect(get_cell_dof_ids(velocity_standard)), q2lattice,
        num_free_dofs(velocity_standard))
    vcompact = vstandard[Int.(getfield(velocity_test, :mDOF_to_DOF))]
    q1lattice = StaticAssembler._cell_nodes(
        StaticAssembler.Q1ReferenceQuadrilateral(), Int32.(active_bg_cells), nx)
    pressure_offset = 2 * (2nx + 1) * (2ny + 1)
    pstandard = _q1_scalar_stable_map(
        collect(get_cell_dof_ids(pressure_standard)), q1lattice,
        num_free_dofs(pressure_standard), pressure_offset)
    pcompact = pstandard[Int.(getfield(pressure_test, :mDOF_to_DOF))]
    marker_offset = pressure_offset + (nx + 1) * (ny + 1)
    cstandard = _q1_scalar_stable_map(
        collect(get_cell_dof_ids(marker_standard)), q1lattice,
        num_free_dofs(marker_standard), marker_offset)
    ccompact = cstandard[Int.(getfield(marker_test, :mDOF_to_DOF))]
    compact_to_stable = vcat(vcompact, pcompact, ccompact)
    matrix, vector = get_matrix(operator), get_vector(operator)
    return (; matrix, vector, compact_to_stable)
end

function article_equivalence_errors(nx::Int=4, ny::Int=3;
                                    parameters=ArticleParameters{Float64}(),
                                    radius::Real=0.38,
                                    center=(0.49, 0.51))
    reference = build_cut_reference(nx, ny; radius, center)
    roots = aggregate(AggregateAllCutCells(), reference.cut_geometry)
    data = _article_data(reference)
    oracle = _article_gridap_operator(reference, roots, data, parameters)
    system = StaticSystem(
        reference.model, reference.cut_geometry, roots, nx, ny;
        form=article_form(parameters),
        interior_quadrature=_gridap_interior_quadrature(4),
        cut_quadrature=_gridap_cut_quadrature(4))
    gridap_to_ours = stable_to_master(system.refresh.E)[oracle.compact_to_stable]
    all(!iszero, gridap_to_ours) ||
        error("Article AgFEM DOFs do not cover the master set exactly")
    B, b = StaticAssembler.assemble!(
        system, data.coefficients;
        facet_fields=data.facet_fields, threaded=false)
    serial = _errors(B[gridap_to_ours, gridap_to_ours], b[gridap_to_ours],
                     oracle.matrix, oracle.vector)
    B, b = StaticAssembler.assemble!(
        system, data.coefficients;
        facet_fields=data.facet_fields, threaded=true)
    threaded = _errors(B[gridap_to_ours, gridap_to_ours], b[gridap_to_ours],
                       oracle.matrix, oracle.vector)
    return (; serial, threaded, system, data)
end

# Background-indexed naive cut quadrature from Gridap's own measure, for
# independent comparison against the engine's staged quadrature. Mirrors the
# `StaticSystem` layout (one segment per background cell, empty for cells
# without physical quadrature points).
function naive_quadrature(reference::CutGridapReference)
    quad, active_bg_cells = _cut_quadrature_gridap(reference, Float64)
    ncells = reference.nx * reference.ny
    pointers = Vector{Int}(undef, ncells + 1)
    is_cut = falses(ncells)
    reference_r = Float64[]
    reference_s = Float64[]
    weight = Float64[]
    active_of = Dict{Int,Int}(bg => a for (a, bg) in enumerate(active_bg_cells))
    pointers[1] = 1
    for bg in 1:ncells
        a = get(active_of, bg, 0)
        if a == 0
            pointers[bg + 1] = pointers[bg]
        else
            lo, hi = quad.pointers[a], quad.pointers[a + 1] - 1
            append!(reference_r, quad.reference_r[lo:hi])
            append!(reference_s, quad.reference_s[lo:hi])
            append!(weight, quad.weight[lo:hi])
            is_cut[bg] = quad.is_cut[a]
            pointers[bg + 1] = pointers[bg] + max(hi - lo + 1, 0)
        end
    end
    return (; pointers, is_cut, reference_r, reference_s, weight)
end

function cut_equivalence_errors(nx::Int=8, ny::Int=7;
                                parameters=KelvinVoigtParameters{Float64}(),
                                order::Int=1, kwargs...)
    reference = build_cut_reference(nx, ny; parameters, kwargs...)
    cutgeo, model = reference.cut_geometry, reference.model
    roots = aggregate(AggregateAllCutCells(), cutgeo)
    origin = first(get_node_coordinates(model))
    coefficients = _analytic_coefficients(nx, ny, reference.hx, reference.hy,
                                          Float64(origin[1]), Float64(origin[2]))
    op = cut_agfem_operator(reference, roots, coefficients; order)
    element = order == 1 ? StaticAssembler.Q1ReferenceQuadrilateral() :
                           StaticAssembler.Q2ReferenceQuadrilateral()
    degree = order == 1 ? 6 : 10
    system = StaticSystem(
        model, cutgeo, roots, nx, ny;
        form=kelvin_voigt_form(parameters; element),
        interior_quadrature=_gridap_interior_quadrature(degree))
    E = system.refresh.E
    gridap_to_ours = stable_to_master(E)[op.compact_to_stable]
    all(!iszero, gridap_to_ours) ||
        error("AgFEM DOFs do not cover the master set exactly")
    B, b = StaticAssembler.assemble!(system, coefficients; threaded=false)
    serial = _errors(B[gridap_to_ours, gridap_to_ours], b[gridap_to_ours],
                     op.matrix, op.vector)
    B, b = StaticAssembler.assemble!(system, coefficients; threaded=true)
    threaded = _errors(B[gridap_to_ours, gridap_to_ours], b[gridap_to_ours],
                       op.matrix, op.vector)
    return (; serial, threaded)
end

function cut_stokes_equivalence_errors(nx::Int=4, ny::Int=3;
                                       parameters=StokesParameters{Float64}(),
                                       radius::Real=0.42,
                                       center::Tuple{<:Real,<:Real}=(0.5, 0.5),
                                       domain::Tuple{<:Real,<:Real,<:Real,<:Real}=(0.0, 1.0, 0.0, 1.0))
    reference = build_cut_stokes_reference(nx, ny; parameters, radius, center, domain)
    cutgeo, model = reference.cut_geometry, reference.model
    roots = aggregate(AggregateAllCutCells(), cutgeo)
    op = cut_stokes_agfem_operator(reference, roots)
    system = StaticSystem(
        model, cutgeo, roots, nx, ny;
        form=stokes_form(parameters),
        interior_quadrature=_gridap_interior_quadrature(10))
    E = system.refresh.E
    gridap_to_ours = stable_to_master(E)[op.compact_to_stable]
    all(!iszero, gridap_to_ours) ||
        error("AgFEM DOFs do not cover the master set exactly")
    B, b = StaticAssembler.assemble!(system, (); threaded=false)
    serial = _errors(B[gridap_to_ours, gridap_to_ours], b[gridap_to_ours],
                     op.matrix, op.vector)
    B, b = StaticAssembler.assemble!(system, (); threaded=true)
    threaded = _errors(B[gridap_to_ours, gridap_to_ours], b[gridap_to_ours],
                       op.matrix, op.vector)
    return (; serial, threaded)
end


end # module
