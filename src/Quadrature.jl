# # Cut-cell quadrature staging and transactional commit.

mutable struct CutQuadrature{T}
    pointers::Vector{Int}
    is_cut::BitVector
    reference_r::Vector{T}
    reference_s::Vector{T}
    weight::Vector{T}
end

mutable struct CutQuadratureStaging{T}
    pointers::Vector{Int}
    is_cut::BitVector
    reference_r::Vector{T}
    reference_s::Vector{T}
    weight::Vector{T}
end

mutable struct BoundaryQuadrature{T}
    pointers::Vector{Int}
    facet::Vector{Int32}
    reference_r::Vector{T}
    reference_s::Vector{T}
    weight::Vector{T}
    normal_x::Vector{T}
    normal_y::Vector{T}
    nfacets::Int
end

mutable struct BoundaryQuadratureStaging{T}
    pointers::Vector{Int}
    facet::Vector{Int32}
    reference_r::Vector{T}
    reference_s::Vector{T}
    weight::Vector{T}
    normal_x::Vector{T}
    normal_y::Vector{T}
    nfacets::Int
end

function _validate_and_size_boundary_staging!(
    staging::BoundaryQuadratureStaging{T}, boundary::BoundaryData{T},
    bg_states, nx::Int, ny::Int, hx::T, hy::T,
    rule::_ReferenceQuadrature{1,T},
) where {T}
    ncells = nx * ny
    nfacets = length(boundary.facet_to_bgcell)
    point_ids = boundary.point_ids
    point_pointers = boundary.point_pointers
    physical_points = boundary.physical_points
    reference_points = boundary.reference_points
    normals = boundary.normals
    length(staging.pointers) == ncells + 1 ||
        error("boundary quadrature staging pointer count changed")
    length(bg_states) == ncells ||
        throw(ArgumentError("boundary cell states do not match the background mesh"))
    length(normals) == nfacets ||
        throw(ArgumentError("boundary normal count is inconsistent"))
    length(point_pointers) == nfacets + 1 && point_pointers[1] == 1 &&
        point_pointers[end] == length(point_ids) + 1 ||
        throw(ArgumentError("boundary connectivity pointers are inconsistent"))
    length(reference_points) == length(physical_points) ||
        throw(ArgumentError("boundary reference and physical point counts differ"))

    reference_tolerance = T(256) * eps(T)
    length_tolerance = T(512) * eps(T) * max(hx, hy)
    counts = staging.pointers
    fill!(counts, 0)
    previous_bg_cell = 0
    nrule = length(rule.points)
    for facet in 1:nfacets
        bg_cell = Int(boundary.facet_to_bgcell[facet])
        1 <= bg_cell <= ncells ||
            throw(ArgumentError("boundary facet references an invalid background cell"))
        bg_cell >= previous_bg_cell ||
            throw(ArgumentError("boundary facets are not grouped in background-cell order"))
        bg_states[bg_cell] == CUT ||
            throw(ArgumentError("boundary facet is attached to an uncut background cell"))
        previous_bg_cell = bg_cell

        first_point = Int(point_pointers[facet])
        next_point = Int(point_pointers[facet + 1])
        next_point == first_point + 2 ||
            throw(ArgumentError("direct boundary quadrature requires line segments"))
        first_point >= 1 && next_point - 1 <= length(point_ids) ||
            throw(ArgumentError("boundary connectivity is out of bounds"))
        point1 = Int(point_ids[first_point])
        point2 = Int(point_ids[first_point + 1])
        1 <= point1 <= length(reference_points) &&
            1 <= point2 <= length(reference_points) ||
            throw(ArgumentError("boundary point ID is out of bounds"))

        r1, r2 = reference_points[point1], reference_points[point2]
        x1, x2 = physical_points[point1], physical_points[point2]
        r, s = T(r1[1]), T(r1[2])
        u, v = T(r2[1]), T(r2[2])
        x, y = T(x1[1]), T(x1[2])
        z, w = T(x2[1]), T(x2[2])
        all(isfinite, (r, s, u, v, x, y, z, w)) ||
            throw(ArgumentError("boundary geometry contains non-finite point data"))
        all(q -> -reference_tolerance <= q <= one(T) + reference_tolerance,
            (r, s, u, v)) ||
            throw(ArgumentError("boundary point lies outside its background reference cell"))
        _metric_matches(z - x, u - r, hx) &&
            _metric_matches(w - y, v - s, hy) ||
            throw(ArgumentError("boundary coordinates disagree with background mesh metrics"))

        segment_length = hypot(z - x, w - y)
        segment_length <= length_tolerance && continue
        normal = normals[facet]
        normal_x, normal_y = T(normal[1]), T(normal[2])
        isfinite(normal_x) && isfinite(normal_y) ||
            throw(ArgumentError("boundary geometry contains a non-finite normal"))
        hypot(normal_x, normal_y) > reference_tolerance ||
            throw(ArgumentError("non-degenerate boundary facet has a zero normal"))
        counts[bg_cell + 1] += nrule
    end

    counts[1] = 1
    for cell in 1:ncells
        counts[cell + 1] += counts[cell]
    end
    npoints = counts[end] - 1
    for values in (staging.facet, staging.reference_r, staging.reference_s,
                   staging.weight, staging.normal_x, staging.normal_y)
        length(values) < npoints && resize!(values, npoints)
    end
    staging.nfacets = nfacets
    return npoints
end

function _stage_boundary_quadrature!(
    staging::BoundaryQuadratureStaging{T}, boundary::BoundaryData{T},
    bg_states, nx::Int, ny::Int, hx::T, hy::T,
    rule::_ReferenceQuadrature{1,T},
) where {T}
    npoints = _validate_and_size_boundary_staging!(
        staging, boundary, bg_states, nx, ny, hx, hy, rule)
    length_tolerance = T(512) * eps(T) * max(hx, hy)
    q = 1
    for facet in eachindex(boundary.facet_to_bgcell)
        first_point = Int(boundary.point_pointers[facet])
        point1 = Int(boundary.point_ids[first_point])
        point2 = Int(boundary.point_ids[first_point + 1])
        r1, r2 = boundary.reference_points[point1], boundary.reference_points[point2]
        x1, x2 = boundary.physical_points[point1], boundary.physical_points[point2]
        segment_length = T(hypot(x2[1] - x1[1], x2[2] - x1[2]))
        segment_length <= length_tolerance && continue
        normal = boundary.normals[facet]
        normal_length = T(hypot(normal[1], normal[2]))
        normal_x = T(normal[1]) / normal_length
        normal_y = T(normal[2]) / normal_length
        @inbounds for k in eachindex(rule.points, rule.weights)
            t = rule.points[k][1]
            staging.facet[q] = Int32(facet)
            staging.reference_r[q] = (one(T) - t) * T(r1[1]) + t * T(r2[1])
            staging.reference_s[q] = (one(T) - t) * T(r1[2]) + t * T(r2[2])
            staging.weight[q] = segment_length * rule.weights[k]
            staging.normal_x[q] = normal_x
            staging.normal_y[q] = normal_y
            q += 1
        end
    end
    @assert q == npoints + 1
    return npoints
end

function _commit_boundary_quadrature!(
    destination::BoundaryQuadrature, staging::BoundaryQuadratureStaging,
    npoints::Int,
)
    length(destination.pointers) == length(staging.pointers) ||
        error("boundary quadrature pointer count changed during commit")
    resize!(destination.facet, npoints)
    resize!(destination.reference_r, npoints)
    resize!(destination.reference_s, npoints)
    resize!(destination.weight, npoints)
    resize!(destination.normal_x, npoints)
    resize!(destination.normal_y, npoints)
    copyto!(destination.pointers, staging.pointers)
    copyto!(destination.facet, 1, staging.facet, 1, npoints)
    copyto!(destination.reference_r, 1, staging.reference_r, 1, npoints)
    copyto!(destination.reference_s, 1, staging.reference_s, 1, npoints)
    copyto!(destination.weight, 1, staging.weight, 1, npoints)
    copyto!(destination.normal_x, 1, staging.normal_x, 1, npoints)
    copyto!(destination.normal_y, 1, staging.normal_y, 1, npoints)
    destination.nfacets = staging.nfacets
    return destination
end

function _validate_and_size_staging!(
    staging::CutQuadratureStaging{T}, active_bg_cells, cutdata::CutData{T},
    nx::Int, ny::Int, hx::T, hy::T, rule::_ReferenceQuadrature{3,T},
) where {T}
    nrule = length(rule.points)
    ncells = nx * ny
    bg_states = cutdata.bg_states
    subcell_states = cutdata.subcell_states
    cell_to_bgcell = cutdata.cell_to_bgcell
    point_ids = cutdata.point_ids
    point_pointers = cutdata.point_pointers
    physical_points = cutdata.physical_points
    reference_points = cutdata.reference_points
    nsubcells = length(cell_to_bgcell)

    length(staging.pointers) == length(active_bg_cells) + 1 ||
        error("quadrature staging pointer count changed")
    length(staging.is_cut) == length(active_bg_cells) ||
        error("quadrature staging cell-state count changed")
    length(subcell_states) == nsubcells ||
        throw(ArgumentError("subcell state count is inconsistent"))
    length(point_pointers) == nsubcells + 1 && point_pointers[1] == 1 &&
        point_pointers[end] == length(point_ids) + 1 ||
        throw(ArgumentError("subcell connectivity pointers are inconsistent"))
    length(reference_points) == length(physical_points) ||
        throw(ArgumentError("reference and physical point counts differ"))

    reference_tolerance = T(256) * eps(T)
    for point in eachindex(reference_points)
        rpoint = reference_points[point]
        xpoint = physical_points[point]
        r, s = T(rpoint[1]), T(rpoint[2])
        x, y = T(xpoint[1]), T(xpoint[2])
        isfinite(r) && isfinite(s) && isfinite(x) && isfinite(y) ||
            throw(ArgumentError("cut geometry contains non-finite point data"))
        -reference_tolerance <= r <= one(T) + reference_tolerance &&
            -reference_tolerance <= s <= one(T) + reference_tolerance ||
            throw(ArgumentError("subcell point lies outside its background reference cell"))
    end

    current_bg_cell = 0
    cut_area = zero(T)
    cell_area = hx * hy
    area_tolerance = T(512) * eps(T) * max(one(T), cell_area)
    reference_det_tolerance = T(512) * eps(T)
    physical_det_tolerance = T(512) * eps(T) * cell_area
    for subcell in 1:nsubcells
        bg_cell = Int(cell_to_bgcell[subcell])
        1 <= bg_cell <= ncells ||
            throw(ArgumentError("subcell references an invalid background cell"))
        if bg_cell != current_bg_cell
            bg_cell > current_bg_cell ||
                throw(ArgumentError("subcells are not grouped in background-cell order"))
            if current_bg_cell > 0 && bg_states[current_bg_cell] == CUT
                cut_area <= cell_area + area_tolerance ||
                    throw(ArgumentError("cut-cell area is empty or exceeds its background cell"))
            end
            for skipped_bg_cell in (current_bg_cell + 1):(bg_cell - 1)
                bg_states[skipped_bg_cell] != CUT ||
                    throw(ArgumentError("a cut background cell has no subcells"))
            end
            bg_states[bg_cell] == CUT ||
                throw(ArgumentError("subcell is attached to an uncut background cell"))
            current_bg_cell = bg_cell
            cut_area = zero(T)
        end

        subcell_state = subcell_states[subcell]
        _valid_subcell_state(subcell_state) ||
            throw(ArgumentError("invalid subcell state"))
        bg_state = bg_states[bg_cell]

        first_point = Int(point_pointers[subcell])
        next_point = Int(point_pointers[subcell + 1])
        next_point == first_point + 3 ||
            throw(ArgumentError("direct quadrature requires triangular subcells"))
        first_point >= 1 && next_point - 1 <= length(point_ids) ||
            throw(ArgumentError("subcell point connectivity is out of bounds"))
        point1 = Int(point_ids[first_point])
        point2 = Int(point_ids[first_point + 1])
        point3 = Int(point_ids[first_point + 2])
        1 <= point1 <= length(reference_points) &&
            1 <= point2 <= length(reference_points) &&
            1 <= point3 <= length(reference_points) ||
            throw(ArgumentError("subcell point ID is out of bounds"))

        r1, r2, r3 = reference_points[point1], reference_points[point2], reference_points[point3]
        x1, x2, x3 = physical_points[point1], physical_points[point2], physical_points[point3]
        dr21, ds21 = T(r2[1] - r1[1]), T(r2[2] - r1[2])
        dr31, ds31 = T(r3[1] - r1[1]), T(r3[2] - r1[2])
        dx21, dy21 = T(x2[1] - x1[1]), T(x2[2] - x1[2])
        dx31, dy31 = T(x3[1] - x1[1]), T(x3[2] - x1[2])
        _metric_matches(dx21, dr21, hx) && _metric_matches(dy21, ds21, hy) &&
            _metric_matches(dx31, dr31, hx) && _metric_matches(dy31, ds31, hy) ||
            throw(ArgumentError("subcell coordinates disagree with background mesh metrics"))
        reference_det = dr21 * ds31 - dr31 * ds21
        physical_det = dx21 * dy31 - dx31 * dy21
        reference_det >= -reference_det_tolerance &&
            physical_det >= -physical_det_tolerance ||
            throw(ArgumentError("cut geometry contains an inverted triangle"))
        _metric_matches(physical_det, reference_det, cell_area) ||
            throw(ArgumentError("triangle determinant disagrees with background mesh metrics"))

        # Exact level-set/node intersections legitimately create zero-measure
        # subtriangles in GridapEmbedded. They contribute no quadrature points.
        if bg_state == CUT && subcell_state == IN &&
           physical_det > physical_det_tolerance
            cut_area += physical_det / 2
        end
    end
    if current_bg_cell > 0
        cut_area <= cell_area + area_tolerance ||
            throw(ArgumentError("cut-cell area is empty or exceeds its background cell"))
    end
    for skipped_bg_cell in (current_bg_cell + 1):ncells
        bg_states[skipped_bg_cell] != CUT ||
            throw(ArgumentError("a cut background cell has no subcells"))
    end

    counts = staging.pointers
    fill!(counts, 0)
    for cell in eachindex(active_bg_cells)
        staging.is_cut[cell] = bg_states[active_bg_cells[cell]] == CUT
    end
    active_cell = 1
    for subcell in 1:nsubcells
        bg_cell = Int(cell_to_bgcell[subcell])
        if bg_states[bg_cell] == CUT && subcell_states[subcell] == IN
            first_point = Int(point_pointers[subcell])
            point1 = Int(point_ids[first_point])
            point2 = Int(point_ids[first_point + 1])
            point3 = Int(point_ids[first_point + 2])
            x1, x2, x3 = physical_points[point1], physical_points[point2], physical_points[point3]
            physical_det = T(
                (x2[1] - x1[1]) * (x3[2] - x1[2]) -
                (x3[1] - x1[1]) * (x2[2] - x1[2]))
            physical_det > physical_det_tolerance || continue
            while active_bg_cells[active_cell] < bg_cell
                active_cell += 1
            end
            @assert active_bg_cells[active_cell] == bg_cell
            counts[active_cell + 1] += nrule
        end
    end
    counts[1] = 1
    for cell in eachindex(active_bg_cells)
        counts[cell + 1] += counts[cell]
    end
    npoints = counts[end] - 1
    length(staging.reference_r) < npoints && resize!(staging.reference_r, npoints)
    length(staging.reference_s) < npoints && resize!(staging.reference_s, npoints)
    length(staging.weight) < npoints && resize!(staging.weight, npoints)
    return npoints
end

function _stage_cut_quadrature!(
    staging::CutQuadratureStaging{T}, active_bg_cells, cutdata::CutData{T},
    nx::Int, ny::Int, hx::T, hy::T, rule::_ReferenceQuadrature{3,T},
) where {T}
    npoints = _validate_and_size_staging!(
        staging, active_bg_cells, cutdata, nx, ny, hx, hy, rule)
    rule_points, rule_weights = rule.points, rule.weights
    q = 1
    for subcell in eachindex(cutdata.cell_to_bgcell)
        bg_cell = Int(cutdata.cell_to_bgcell[subcell])
        cutdata.bg_states[bg_cell] == CUT && cutdata.subcell_states[subcell] == IN || continue
        first_point = Int(cutdata.point_pointers[subcell])
        point1 = Int(cutdata.point_ids[first_point])
        point2 = Int(cutdata.point_ids[first_point + 1])
        point3 = Int(cutdata.point_ids[first_point + 2])
        r1, r2, r3 = cutdata.reference_points[point1], cutdata.reference_points[point2], cutdata.reference_points[point3]
        x1, x2, x3 = cutdata.physical_points[point1], cutdata.physical_points[point2], cutdata.physical_points[point3]
        physical_det = T(
            (x2[1] - x1[1]) * (x3[2] - x1[2]) -
            (x3[1] - x1[1]) * (x2[2] - x1[2]))
        physical_det > T(512) * eps(T) * hx * hy || continue
        weight = physical_det / 2

        # Barycentric rule points on the subtriangle; the affine map to the
        # background reference cell is element-independent.
        @inbounds for k in eachindex(rule_points, rule_weights)
            lambda = rule_points[k]
            staging.reference_r[q] = T(lambda[1] * r1[1] + lambda[2] * r2[1] + lambda[3] * r3[1])
            staging.reference_s[q] = T(lambda[1] * r1[2] + lambda[2] * r2[2] + lambda[3] * r3[2])
            staging.weight[q] = weight * T(rule_weights[k])
            q += 1
        end
    end
    @assert q == npoints + 1
    return npoints
end

function _commit_quadrature!(
    destination::CutQuadrature, pointers, is_cut,
    reference_r, reference_s, weight, npoints::Int)
    length(destination.pointers) == length(pointers) ||
        error("quadrature pointer count changed during commit")
    if npoints > length(destination.reference_r)
        sizehint!(destination.reference_r, npoints)
        sizehint!(destination.reference_s, npoints)
        sizehint!(destination.weight, npoints)
    end
    resize!(destination.reference_r, npoints)
    resize!(destination.reference_s, npoints)
    resize!(destination.weight, npoints)
    copyto!(destination.reference_r, 1, reference_r, 1, npoints)
    copyto!(destination.reference_s, 1, reference_s, 1, npoints)
    copyto!(destination.weight, 1, weight, 1, npoints)
    copyto!(destination.pointers, pointers)
    copyto!(destination.is_cut, is_cut)
    return destination
end
