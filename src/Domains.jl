# # Cell-state validation and metric matching.

@inline _valid_bg_state(state) = state == IN || state == CUT || state == OUT
@inline _valid_subcell_state(state) = state == IN || state == OUT

@inline function _metric_matches(physical_delta::T, reference_delta::T, scale::T) where {T}
    expected = scale * reference_delta
    tolerance = T(256) * eps(T) * max(one(T), abs(physical_delta), abs(expected))
    return abs(physical_delta - expected) <= tolerance
end