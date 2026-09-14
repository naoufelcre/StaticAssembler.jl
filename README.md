# StaticAssembler.jl

Fast repeated finite-element assembly on evolving 2D Cartesian cut domains.

The projected system is

```text
B = E' * A * E
b_m = E' * b
```

where `A` and `b` are sums of local finite-element contributions and `E` is
the AgFEM extension from master degrees of freedom to the active background
lattice.

Changing the form, finite-element spaces, or background mesh requires a new
`StaticSystem`; changing coefficients uses `assemble!`; changing cut geometry
or aggregate roots uses `refresh_geometry!`.

### DONE

- Affine 2D Cartesian background meshes.
- Scalar and vector Q1/Q2 Lagrange elements, including mixed spaces.
- Volume integration over `IN` cells and triangulated physical parts of `CUT`
  cells.
- Constant and state-dependent bilinear terms and right-hand sides.
- Projected AgFEM assembly, geometry refresh, sparse-slot reuse, and colored
  threaded scatter.
- One CSG geometry tree (single level set or composed unions, intersections,
  differences, and complements via composed inside/outside states), `Float64`,
  and identical test/trial spaces in the projected `StaticSystem` path.
- Scalar coefficient fields represented on Q1 or Q2 background lattices;
  default quadrature rules upgrade to Q2 when any coefficient field uses Q2.
- Embedded line-segment quadrature, including multiple segments per cut cell,
  with degeneracy tolerance scaled to the background spacing.
- Outward-normal and counter-clockwise `(-n_y,n_x)` tangent-aware boundary
  operators. Gridap transcriptions must write the tangent as
  `normal ⋅ TensorValue(0.0, -1.0, 1.0, 0.0)`: `TensorValue` stores components
  column-major, so the reversed order evaluates to the opposite tangent and
  flips single-tangent terms such as `Ft`.
- Piecewise-constant per-facet curvature, masks, tractions, and other data.
- Local boundary bilinear and right-hand-side terms, including penalty,
  Neumann, Robin, and user-defined single-sided trace contractions.

### TODO/FUTURE WORK ??

- Boundary data varying independently at every quadrature point rather than
  once per embedded facet.
- Interior-facet terms such as DG fluxes, CIP stabilization, and ghost
  penalties, because one contribution reads and couples two adjacent cells.
- Fully coupled two-sided material-interface terms with independent unknowns
  and jump/average operators on both sides.
- Multiple level sets or multiple physical phases in one system.
- 3D cut surfaces and unstructured background meshes.


### Remarks

- Contributions must have fixed finite support for a given structural state
  so that a reusable sparse pattern can be built. Dense nonlocal boundary
  integral operators are not handled by this local sparse assembler.
- A topology or aggregation change may change the active degrees of freedom
  and sparse pattern. Such a change necessarily takes the structural refresh
  path rather than the coefficient-only hot path.


StaticAssembler does not own level-set evolution, cut generation, curvature
estimation, aggregate-root selection, PDE time integration, linear solvers, or
preconditioners. Geometry producers may provide normals and curvature, and the
assembler evaluates their effect at cached quadrature points. The quality of
those supplied geometric quantities is not recoverable inside assembly.

Development version.
