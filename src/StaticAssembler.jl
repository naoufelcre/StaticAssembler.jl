module StaticAssembler

# Low-level, data-oriented finite element assembly on cut domains
#
# The engine is backend-agnostic. It consumes plain Mesh/CutData descriptors
# (see Descriptors.jl);
#
# Gridap/GridapEmbedded are a convenient
# producer of those descriptors through the optional StaticAssemblerGridapExt
# extension, and a custom low-level implementation can build them directly.
# PDE-specific physics lives in a user-supplied ElementVariationalForm (see
# Forms.jl); element tables (reference basis evaluated at quadrature nodes +
# constant-part element stiffness matrices) are generated once per geometry
# refresh. This package contains no material model and no callable kernel.
#
# Public API (exported):
#   StaticSystem(...)             reusable master-only projected AgFEM system
#   refresh_geometry!(...)        refresh quadrature, E, and B's sparse pattern
#   assemble!(...)                assemble the projected operator B=EᵀAE and RHS
#
# The projection is implicit in B: E is built from the aggregate roots and
# available through `extension(system)` (a real sparse matrix), the raw per-cell blocks
# are projected by the constraint tables, and B is the real sparse operator
# solved directly (cholesky / Preconditioners.jl). Solutions are extended to
# the full background lattice with `E * u_m` (allocation-free via
# `mul!(u_full, E, u_m)`); `restrict_solution!` extracts master values.
#
# Systems are single-owner and non-reentrant: matrices, right-hand sides, and
# the projected operator reuse internal buffers that the next `refresh_geometry!`
# or `assemble!` overwrites.

using LinearAlgebra
using SparseArrays

include("Descriptors.jl")
include("Elements.jl")
include("Forms.jl")
include("Domains.jl")
include("Quadrature.jl")
include("Spaces.jl")
include("Assembly.jl")
include("StaticSystem.jl")

export StaticSystem, refresh_geometry!, assemble!, extension, restrict_solution!
export Mesh, CutData, BoundaryData
export ElementVariationalForm, LocalForm, BilinearSlot, RhsSlot
export ConstantCoefficient, FieldCoefficient, StateFunction
export coefficient_gradient, facet_value, boundary_normal, boundary_tangent
export FiniteElement
# Note: `FESpace` stays available as `StaticAssembler.FESpace` for backward
# compatibility but is intentionally not exported: Gridap also exports `FESpace`.
export ReferenceFiniteElement, AbstractReferenceElement
export Q1ReferenceQuadrilateral, Q2ReferenceQuadrilateral
export BilinearQuad, BiquadraticQuad
export AbstractDifferentialOperator, AbstractOperator
export ElementTables, LocalKernel
export generate_element_tables, generate_local_kernel

end
