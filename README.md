# StaticAssembler.jl

Specialized custom assembler and data structure for aggregated nodal finite element methods on 2D evolving domains.

This is a research toolbox to explore, not a fully fleshed out finite element framework. For the latter, we encourage you to look after the following libraries in the `Julia` programming language:

- `GridapEmbedded.jl`: A reference implementation of the aggregated finite element method (AgFEM).
- `Ferrite.jl`: Low-level FEM framework.

`StaticAssembler.jl` is related to `EvolvingDomains.jl`. Both libraries were designed as part of my PHD thesis.

About **AgFEM** see : **The aggregated unfitted finite element method for elliptic problems**  ![Static Badge](https://img.shields.io/badge/doi%3A-10.1016%2Fj.cma.2018.03.022-red)


If you are interested in this work, please get in touch.
Contact: **naoufel.cresson@inria.fr** 
https://www.ljll.fr/~cresson/

-------------------------------------------------

# Quickstart

Let $\Omega_n$ be a collection of domains embedded in a larger discrete ambiant space $\mathcal{A}$.
The objective of the package was to find a efficient and convenient way to leverage the shared $\mathcal{A}$ to reduces repeated allocations and matrix assembly when solving FEM problems on different elements of $\Omega_n$.

## AgFEM in a nutshell, and our take on it

### Aggregates

Let $\Omega$ be a domain embedded in $\mathcal{A}$. Cells of $\mathcal{A}$ can be classified as `IN` if the cell is fully inside $\Omega$, `CUT` if the cell is crossed by the boundary of $\Omega$ and `OUT` if the cell is completely outside of $\Omega$.

Based on this classification we can define *aggregates*. An aggregate is simply a map between `CUT`-cells and `IN`-cells.

*Note that `StaticAssembler` purposely do not provide built-in methods to generate aggregate. To generate them methods are available in `GridapEmbedded` or `EvolvingDomains`.*

$\mathcal{A}$ provides the cells and associated DOFs. Only `IN` and `CUT` cells contribute to the system, with the latter integrated over their intersection with $\Omega$. Without aggregation, assembling these contributions gives a matrix $A$ and a right-hand side $b$, defined on all $\mathcal{A}$-DOFs. This system may be ill-conditioned due to arbitrarily small intersections of `CUT` cells with $\Omega$.

The ill conditioning problem is adressed by enforcing constraints between the DOFs. Provided the aggregate map, DOFs belonging to cut cells can be reformulated in term of DOFs belonging to their respective root cells. This distinguishes *master* DOFs, which remain independent, from *slave* DOFs, whose values are determined by the master DOFs.

### The extension operator and assembling the FEM matrix

Constraint above define the AgFEM *extension operator* $E$, which maps the independent master DOFs to the DOFs of the active ambient discretization:

$$
u_{\mathrm{bg}} = E u_m.
$$

For master DOFs, $E$ simply preserves their values. For slave DOFs, it evaluates the finite-element basis of the corresponding root cell at the slave DOF locations.

Using this operator, the system is projected onto the master DOFs:

$$
B = E^T A E, \qquad b_m = E^T b.
$$

$B$ and $b_m$ are then the AgFEM matrix.

The benefit of proceeding as such is twofold:
 1. It allows for a static memory layout, we only allocate once all DOFs. This allow for true gain when dealing with large systems involving frequent domain changes. (For instance moving domain problems)

The above presentation stays at the high-level on purpose, for more details (e.g. the full algebraic mess) we refer to the documentation : [docs/details.md](docs/details.md).

## Package specs

In the following, code snippets are illustrative

### Inputs: Geometric specs and Variational forms

Inputs are of two different types:

**Geometric specs** (`src/Descriptors.jl`): structured `Mesh`, `CutData` (cell classification, embedded subcells and their reference quadrature) and optional `BoundaryData` (facets, normals). 

**Variational forms** (`src/Forms.jl`): `ElementVariationalForm` names the test/trial blocks, differential operators, coefficient rules and sources. 

```julia
mesh = Mesh(mesh_id, nx, ny, x0, y0, hx, hy)
form = ElementVariationalForm(Q1, Q1, bilinear, rhs, ncoefficients)
```

### Allocation 

Allocate once a `StaticSystem` which is our all-in one container for the background system. 

```julia
system = StaticSystem(mesh, cutdata, roots; form = form) 
```

### Refreshing the system after incoming inputs

In the case where we get new cutdata/aggregates, we should refresh the StaticSystem and extract from it the extension operator.

```julia
refresh_geometry!(system, cutdata, roots)
E = extension(system)              # sparse n_bg × n_master
```

### Assembly and final system

If only coefficients `(k,)` of your variational form changed you can provide them directly to the assembler whitout refreshing the system:

```julia
B, b_m = assemble!(system, (κ,); threaded = true)


#final AgFEM operator: B is already EᵀAE on master DOFs.
u_m  = B \ b_m                    # solve on master DOFs
u_bg = E * u_m                    # map back to the background lattice
```

For more details and examples consult the documentation.

-------------------------------------------------------------------

## Important limitations

It needs to remain clear to the reader that this is a research package and we do not claim it to be a fully fleshed out FEM framework. In particular notable limitations are: 

1. Rigidity of the inputs: `Float64`, 2D Cartesian grids, identical test and trial elements (possibly mixed Q1/Q2 blocks, scalar or componentwise vector).

2. Changing the form, finite-element spaces, or background mesh requires a new
`StaticSystem`.

3.  Dense nonlocal boundary integral operators are not handled.
