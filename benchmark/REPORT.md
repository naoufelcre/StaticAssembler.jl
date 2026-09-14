# StaticAssembler assembly performance

Measured 2026-09-08 on Apple M2, macOS 15.7.3, Julia 1.11.7,
Gridap 0.19.11, and GridapEmbedded 0.9.9.

## Conclusion

The steady-state assembly path scales approximately linearly with active cells.
For the largest tested circular cut domain (128 x 128 background cells):

| Element | Active cells | Master DOFs | Matrix nnz | Serial median | 4-thread median | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Q1 vector | 9,300 | 18,170 | 336,460 | 5.45 ms | 2.13 ms | 2.56x |
| Q2 vector | 9,300 | 71,826 | 2,397,108 | 45.15 ms | 16.62 ms | 2.72x |

This workspace has no Git history or previous benchmark results, so these
measurements establish a baseline; they cannot prove or disprove a historical
performance regression.

## Results

Times are medians of 30 samples after warmup. Each sample contains 3 or 10
assembly calls. Construction, geometry refresh, JIT compilation, and output
copies are excluded. The workload is the four-coefficient Kelvin-Voigt form on
a radius-0.42 circular cut domain. `threaded` uses four Julia threads.

### Q1 vector element

| Grid | Active / total cells | Cut cells | Master DOFs | Matrix nnz | Serial | 4-thread | Speedup | Serial throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 x 16 | 172 / 256 | 52 | 290 | 6,020 | 0.121 ms | 0.251 ms | 0.48x | 1.43 M active cells/s |
| 32 x 32 | 624 / 1,024 | 108 | 1,138 | 22,540 | 0.417 ms | 0.321 ms | 1.30x | 1.50 M active cells/s |
| 64 x 64 | 2,360 / 4,096 | 212 | 4,506 | 85,604 | 1.478 ms | 0.655 ms | 2.25x | 1.60 M active cells/s |
| 128 x 128 | 9,300 / 16,384 | 428 | 18,170 | 336,460 | 5.449 ms | 2.128 ms | 2.56x | 1.71 M active cells/s |

### Q2 vector element

| Grid | Active / total cells | Cut cells | Master DOFs | Matrix nnz | Serial | 4-thread | Speedup | Serial throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 x 16 | 172 / 256 | 52 | 1,058 | 43,012 | 1.028 ms | 0.529 ms | 1.94x | 167.3 k active cells/s |
| 32 x 32 | 624 / 1,024 | 108 | 4,338 | 160,852 | 3.426 ms | 1.371 ms | 2.50x | 182.1 k active cells/s |
| 64 x 64 | 2,360 / 4,096 | 212 | 17,602 | 611,172 | 12.028 ms | 4.469 ms | 2.69x | 196.2 k active cells/s |
| 128 x 128 | 9,300 / 16,384 | 428 | 71,826 | 2,397,108 | 45.155 ms | 16.622 ms | 2.72x | 206.0 k active cells/s |

## Allocation behavior

| Mode | Bytes per call | Allocations per call |
|---|---:|---:|
| Serial | 64 B | 1 |
| 4-thread | 36-39 KiB | 243-265 |

Serial assembly is effectively allocation-free. Threaded assembly incurs a
fixed scheduler cost from one `Threads.@threads` region per graph color. That
cost explains why Q1 threading loses on the 16 x 16 grid; Q2, with more work
per cell, benefits at every measured size.

## Geometry refresh versus full rebuild

The public Gridap API was also measured for two refresh paths:

- Same geometry and roots: quadrature and element-table values are refreshed,
  while `E`, constraints, sparse pattern, scatter slots, and coloring are reused.
- Changed geometry: alternate radius/center geometries change cell states and
  aggregate roots, forcing the structural refresh path.
- Full rebuild: construct a new `StaticSystem` for the alternating geometry.

Times are medians of 20 samples and include Gridap-to-`CutData` conversion.

### Q1 vector element

| Grid | Same geometry | Changed geometry | Full rebuild | Same/rebuild speedup | Changed/rebuild speedup |
|---|---:|---:|---:|---:|---:|
| 16 x 16 | 0.047 ms | 0.331 ms | 0.369 ms | 7.8x | 1.12x |
| 32 x 32 | 0.093 ms | 0.883 ms | 0.973 ms | 10.5x | 1.10x |
| 64 x 64 | 0.194 ms | 2.609 ms | 2.909 ms | 15.0x | 1.12x |
| 128 x 128 | 0.468 ms | 8.775 ms | 10.382 ms | 22.2x | 1.18x |

### Q2 vector element

| Grid | Same geometry | Changed geometry | Full rebuild | Same/rebuild speedup | Changed/rebuild speedup |
|---|---:|---:|---:|---:|---:|
| 16 x 16 | 0.488 ms | 1.859 ms | 2.009 ms | 4.1x | 1.08x |
| 32 x 32 | 1.026 ms | 5.245 ms | 5.633 ms | 5.5x | 1.07x |
| 64 x 64 | 2.043 ms | 15.436 ms | 17.380 ms | 8.5x | 1.13x |
| 128 x 128 | 4.121 ms | 53.433 ms | 64.990 ms | 15.8x | 1.22x |

At 128 x 128, same-geometry refresh allocates 194 KiB. A changed refresh
allocates 9.1 MiB for Q1 and 53.7 MiB for Q2, versus 46.3 MiB and 279 MiB for
constructing fresh systems.

Profiling showed that quadrature staging/commit and Gridap conversion together
take less than 0.4 ms. The original structural path stored heterogeneous
extension rows in `Vector{Any}`, causing boxed work inside every cell loop, and
replaced constraints, sparse-pattern workspaces, CSC arrays, and scatter slots
on every refresh. Type-stable tuple rows plus in-place cache reuse reduced the
Q1 128 x 128 changed path from 106.7 ms/97 MiB to 8.8 ms/9.1 MiB. No
incremental numbering or geometry-map API was needed.

The practical conclusion is that refresh is cheapest while projection
structure stays fixed, but warmed structural refresh now also reuses its large
workspaces. A real 324 x 324 Q1 simulation with changing projection structure
took 83-85 ms and about 50 MiB per refresh, down from 0.84-1.10 s and
923-932 MiB.

Matrices returned by `assemble!` are borrowed cache objects. A structural
refresh may resize their backing storage, so callers must reacquire `B` from
the next assembly call rather than retaining it across refreshes.

## Scaling and variability

- From 32 x 32 to 128 x 128, active cells grow 14.9x. Serial median time grows
  13.1x for Q1 and 13.2x for Q2, consistent with linear complexity.
- Q1 serial cost falls from 0.70 to 0.59 microseconds per active cell.
- Q2 serial cost falls from 6.0 to 4.9 microseconds per active cell as fixed
  overhead is amortized.
- Four threads do not approach a 4x speedup. Graph-color barriers, fixed
  scheduling allocations, sparse scatter traffic, and memory bandwidth limit
  scaling.

## Regression assessment

No regression verdict is possible yet because there is no earlier commit or
stored baseline in this directory. These numbers should be treated as the
first baseline. A practical initial gate on this machine is:

- Track serial Q1 and Q2 at 128 x 128 as the stable regression signals.
- Flag a change only when the median worsens by more than 15% in at least two
  fresh processes.
- Track threaded results separately with a wider threshold because their
  run-to-run variance is materially higher.
- Keep allocation gates at 64 B for serial and 48 KiB for four-thread assembly.

## Reproduction

Run from the package root:

```sh
JULIA_NUM_THREADS=4 julia --project=. --startup-file=no benchmark/assembly.jl
JULIA_NUM_THREADS=4 julia --project=. --startup-file=no benchmark/refresh.jl
JULIA_NUM_THREADS=4 julia --project=. --startup-file=no benchmark/article.jl
JULIA_NUM_THREADS=4 julia --project=. --startup-file=no benchmark/profile_refresh.jl 128 1
```

The script prints CSV rows for Q1/Q2, serial/threaded, all grid sizes, timing,
and allocations. Results are machine-specific and should not be compared
across different Julia versions, power modes, or hardware as a regression
test.

## Scope

This report measures repeated `assemble!` calls, public geometry refresh, and
system reconstruction. It does not measure generation of the Gridap cut
geometry or aggregate roots, Gridap's reference assembler, linear solves,
memory footprint of persistent caches, or scaling beyond four threads. Those
are different operations and need separate benchmarks if they become
performance-critical.
