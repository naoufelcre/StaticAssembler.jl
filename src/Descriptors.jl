# # Backend-agnostic input descriptors.
#
# The engine consumes only these plain data structures. Gridap/GridapEmbedded
# are one convenient producer through the StaticAssemblerGridapExt extension; a
# custom low-level implementation can construct them directly. Descriptors
# are built once per geometry refresh, never inside the assembly hot loop.

# Canonical cut states. GridapEmbedded uses the same encoding (IN=-1, OUT=1,
# CUT=0); the extension asserts compatibility when it reads Gridap data.
const IN = Int8(-1)
const OUT = Int8(1)
const CUT = Int8(0)

"""
    Mesh{T}

Structured background grid: partition, origin, and cell spacing.
`mesh_id` is an opaque identity token used to reject cut geometry from a
different background mesh; producers that do not need identity checks may
use `nothing`.
"""
struct Mesh{T}
    mesh_id::Any
    nx::Int
    ny::Int
    origin_x::T
    origin_y::T
    hx::T
    hy::T
end

"""
    CutData{T}

Cut-cell geometry for a single level set: per-background-cell states
(`IN`/`CUT`/`OUT`), subcell states, triangular subcell connectivity, and
point coordinates as plain `(x, y)` / `(r, s)` pairs.
"""
struct CutData{T}
    mesh_id::Any
    bg_states::Vector{Int8}
    subcell_states::Vector{Int8}
    cell_to_bgcell::Vector{Int32}
    point_ids::Vector{Int32}
    point_pointers::Vector{Int32}
    physical_points::Vector{NTuple{2,T}}
    reference_points::Vector{NTuple{2,T}}
end