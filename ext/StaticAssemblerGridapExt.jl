# Gridap/GridapEmbedded producer for StaticAssembler descriptors.
#
# This extension is the only place Gridap types and private-field layouts are
# read. It converts Gridap data structures into the backend-agnostic
# Mesh/CutData descriptors the engine consumes, and re-exposes the
# classic Gridap-native constructor signatures as conveniences. Layout
# tolerance is enforced by `fieldnames` inspection rather than exact version
# pins, so the extension tracks the pinned Gridap/GridapEmbedded majors.

module StaticAssemblerGridapExt

using Gridap
using GridapEmbedded
using Gridap.Geometry: get_node_coordinates
using StaticAssembler
using StaticAssembler: Mesh, CutData, BoundaryData

const _T = Float64

# StaticAssembler canonical states must match GridapEmbedded's encoding (IN=-1,
# OUT=1, CUT=0). Fail loudly if a GridapEmbedded major changes the encoding.
@assert IN == StaticAssembler.IN && GridapEmbedded.Interfaces.CUT == StaticAssembler.CUT &&
        OUT == StaticAssembler.OUT

function _cut_geometry_data(cut_geometry, ::Type{T}; include_boundary::Bool=true) where {T}
    # GridapEmbedded has no public zero-copy subcell API. Keep its pinned
    # private layout dependency in this adapter and fail loudly if it drifts.
    @assert fieldnames(typeof(cut_geometry)) == (
        :bgmodel, :ls_to_bgcell_to_inoutcut, :subcells,
        :ls_to_subcell_to_inout, :subfacets, :ls_to_subfacet_to_inout,
        :oid_to_ls, :geo,
    )
    geo = getfield(cut_geometry, :geo)
    bg_states_composed = Vector{Int8}(
        GridapEmbedded.Interfaces.compute_bgcell_to_inoutcut(cut_geometry, geo))
    subcell_states_composed = Vector{Int8}(
        GridapEmbedded.Interfaces.compute_subcell_to_inout(cut_geometry, geo))

    subcells = getfield(cut_geometry, :subcells)
    @assert fieldnames(typeof(subcells)) == (
        :cell_to_points, :cell_to_bgcell, :point_to_coords, :point_to_rcoords)
    cell_to_points = getfield(subcells, :cell_to_points)
    @assert fieldnames(typeof(cell_to_points)) == (:data, :ptrs)
    to_pairs(points) = map(p -> (T(p[1]), T(p[2])), points)
    boundary_data = if include_boundary
        subfacets = getfield(EmbeddedBoundary(cut_geometry), :subfacets)
        @assert fieldnames(typeof(subfacets)) == (
            :facet_to_points, :facet_to_normal, :facet_to_bgcell,
            :point_to_coords, :point_to_rcoords)
        facet_to_points = getfield(subfacets, :facet_to_points)
        @assert fieldnames(typeof(facet_to_points)) == (:data, :ptrs)
        BoundaryData{T}(
            Vector{Int32}(getfield(subfacets, :facet_to_bgcell)),
            Vector{Int32}(getfield(facet_to_points, :data)),
            Vector{Int32}(getfield(facet_to_points, :ptrs)),
            to_pairs(getfield(subfacets, :facet_to_normal)),
            to_pairs(getfield(subfacets, :point_to_coords)),
            to_pairs(getfield(subfacets, :point_to_rcoords)),
        )
    else
        BoundaryData{T}()
    end
    return CutData{T}(
        getfield(cut_geometry, :bgmodel),
        bg_states_composed,
        subcell_states_composed,
        Vector{Int32}(getfield(subcells, :cell_to_bgcell)),
        Vector{Int32}(getfield(cell_to_points, :data)),
        Vector{Int32}(getfield(cell_to_points, :ptrs)),
        to_pairs(getfield(subcells, :point_to_coords)),
        to_pairs(getfield(subcells, :point_to_rcoords)),
        boundary_data,
    )
end

function _validate_model_metrics(coordinates, nx::Int, ny::Int, hx::T, hy::T) where {T}
    lower_left = coordinates[1]
    x_neighbor = coordinates[2]
    y_neighbor = coordinates[nx + 2]
    upper_right = coordinates[end]
    StaticAssembler._metric_matches(T(x_neighbor[1] - lower_left[1]), one(T), hx) &&
        StaticAssembler._metric_matches(T(x_neighbor[2] - lower_left[2]), zero(T), hy) &&
        StaticAssembler._metric_matches(T(y_neighbor[1] - lower_left[1]), zero(T), hx) &&
        StaticAssembler._metric_matches(T(y_neighbor[2] - lower_left[2]), one(T), hy) &&
        StaticAssembler._metric_matches(T(upper_right[1] - lower_left[1]), T(nx), hx) &&
        StaticAssembler._metric_matches(T(upper_right[2] - lower_left[2]), T(ny), hy) ||
        throw(ArgumentError("background model coordinates disagree with hx and hy"))
    return nothing
end

"""
    from_gridap(model, cut_geometry, nx, ny) -> (mesh, cutdata)

Build the background-grid and cut-geometry descriptors from a Gridap model and
GridapEmbedded cut geometry. `nx`/`ny` are the background partition and are
validated against the model coordinates.
"""
function from_gridap(model, cut_geometry, nx::Int, ny::Int)
    T = _T
    nx > 0 && ny > 0 || throw(ArgumentError("nx and ny must be positive"))
    coordinates = get_node_coordinates(model)
    length(coordinates) == (nx + 1) * (ny + 1) ||
        throw(ArgumentError("background model dimensions do not match nx and ny"))
    origin = first(coordinates)
    hx = T(coordinates[2][1] - origin[1])
    hy = T(coordinates[nx + 2][2] - origin[2])
    isfinite(hx) && isfinite(hy) && hx > 0 && hy > 0 ||
        throw(ArgumentError("background grid metrics must be finite and positive"))
    _validate_model_metrics(coordinates, nx, ny, hx, hy)
    mesh = Mesh{T}(model, nx, ny, T(origin[1]), T(origin[2]), hx, hy)
    cutdata = _cut_geometry_data(cut_geometry, T)
    cutdata.mesh_id === model ||
        throw(ArgumentError("cut geometry belongs to a different background mesh"))
    return mesh, cutdata
end

# ---------------------------------------------------------------------------
# Gridap-native convenience constructor: same call signature as the classic
# StaticAssembler API, translated to descriptors at the boundary.
# ---------------------------------------------------------------------------

function StaticAssembler.StaticSystem(
    model, cut_geometry, aggregate_roots, nx::Int, ny::Int;
    form,
    interior_quadrature=nothing,
    cut_quadrature=nothing,
    boundary_quadrature=nothing,
)
    mesh, cutdata = from_gridap(model, cut_geometry, nx, ny)
    return StaticAssembler.StaticSystem(
        mesh, cutdata, aggregate_roots;
        form, interior_quadrature, cut_quadrature, boundary_quadrature)
end

StaticAssembler.refresh_geometry!(plan::StaticAssembler.StaticSystem,
                              cut_geometry, aggregate_roots) = begin
    tables = plan.refresh.element_tables
    include_boundary = !isempty(tables.boundary_bilinear) ||
                       !isempty(tables.boundary_rhs)
    StaticAssembler.refresh_geometry!(
        plan, _cut_geometry_data(cut_geometry, _T; include_boundary),
        aggregate_roots)
end

end # module
