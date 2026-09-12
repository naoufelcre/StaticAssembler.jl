using StaticAssembler
include(joinpath(@__DIR__, "..", "oracle", "StaticAssemblerOracle.jl"))
using .StaticAssemblerOracle
using GridapEmbedded
using SparseArrays
using Statistics

function build_case(n, order)
    reference = build_cut_reference(n, n)
    roots = aggregate(AggregateAllCutCells(), reference.cut_geometry)
    element = order == 1 ? StaticAssembler.Q1ReferenceQuadrilateral() :
                           StaticAssembler.Q2ReferenceQuadrilateral()
    system = StaticSystem(reference.model, reference.cut_geometry, roots, n, n;
        form=kelvin_voigt_form(reference.parameters; element))
    coefficients = StaticAssemblerOracle._analytic_coefficients(
        n, n, reference.hx, reference.hy, 0.0, 0.0)
    return system, coefficients
end

function measure_call(f; samples=30, evaluations=5)
    f()
    times = Vector{Float64}(undef, samples)
    GC.gc()
    GC.enable(false)
    try
        for sample in 1:samples
            start = time_ns()
            for _ in 1:evaluations
                f()
            end
            times[sample] = (time_ns() - start) / evaluations
        end
    finally
        GC.enable(true)
    end
    sort!(times)
    return (; minimum=times[1], median=median(times), p90=quantile(times, 0.9))
end

function main()
    println("julia=$(VERSION) threads=$(Threads.nthreads()) cpu=$(Sys.CPU_NAME)")
    println("order,n,cells,active,cut,dofs,nnz,mode,min_ms,median_ms,p90_ms,alloc_bytes,allocs")
    for order in (1, 2), n in (16, 32, 64, 128)
        system, coefficients = build_case(n, order)
        active = count(!=(StaticAssembler.OUT), system.refresh.cell_states)
        cut = count(==(StaticAssembler.CUT), system.refresh.cell_states)
        for threaded in (false, true)
            call = () -> assemble!(system, coefficients; threaded)
            call()
            timing = measure_call(call; samples=30, evaluations=n <= 32 ? 10 : 3)
            allocation = @timed call()
            mode = threaded ? "threaded" : "serial"
            println(join((order, n, n^2, active, cut, size(system.assemble.B, 1),
                nnz(system.assemble.B), mode,
                timing.minimum / 1e6, timing.median / 1e6, timing.p90 / 1e6,
                allocation.bytes, allocation.gcstats.malloc + allocation.gcstats.realloc +
                allocation.gcstats.poolalloc + allocation.gcstats.bigalloc), ','))
        end
    end
end

main()
