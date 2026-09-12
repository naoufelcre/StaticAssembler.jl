using StaticAssembler
include(joinpath(@__DIR__, "..", "oracle", "StaticAssemblerOracle.jl"))
using .StaticAssemblerOracle
using GridapEmbedded
using Statistics

function measure_call(f; samples=20)
    f()
    times = Vector{Float64}(undef, samples)
    for sample in eachindex(times)
        GC.gc()
        times[sample] = @elapsed f()
    end
    sort!(times)
    return median(times) * 1e3
end

function main()
    println("julia=$(VERSION) threads=$(Threads.nthreads()) cpu=$(Sys.CPU_NAME)")
    println("order,n,operation,median_ms,alloc_bytes")
    for order in (1, 2), n in (16, 32, 64, 128)
        base = build_cut_reference(n, n; radius=0.42, center=(0.50, 0.50))
        moved = build_cut_reference(n, n; radius=0.39, center=(0.53, 0.48),
                                    model=base.model)
        roots = aggregate(AggregateAllCutCells(), base.cut_geometry)
        moved_roots = aggregate(AggregateAllCutCells(), moved.cut_geometry)
        element = order == 1 ? StaticAssembler.Q1ReferenceQuadrilateral() :
                               StaticAssembler.Q2ReferenceQuadrilateral()
        form = kelvin_voigt_form(base.parameters; element)
        system = StaticSystem(base.model, base.cut_geometry, roots, n, n; form)

        same = () -> refresh_geometry!(system, base.cut_geometry, roots)
        same_time = measure_call(same)
        same_bytes = @allocated same()
        println(join((order, n, "same_geometry", same_time, same_bytes), ','))

        moved_first = Ref(false)
        changed = () -> begin
            moved_first[] = !moved_first[]
            if moved_first[]
                refresh_geometry!(system, moved.cut_geometry, moved_roots)
            else
                refresh_geometry!(system, base.cut_geometry, roots)
            end
        end
        changed_time = measure_call(changed)
        changed_bytes = @allocated changed()
        println(join((order, n, "changed_geometry", changed_time, changed_bytes), ','))

        rebuild_moved = Ref(false)
        rebuild = () -> begin
            rebuild_moved[] = !rebuild_moved[]
            rebuild_moved[] ?
                StaticSystem(base.model, moved.cut_geometry, moved_roots, n, n; form) :
                StaticSystem(base.model, base.cut_geometry, roots, n, n; form)
        end
        rebuild_time = measure_call(rebuild)
        rebuild_bytes = @allocated rebuild()
        println(join((order, n, "full_rebuild", rebuild_time, rebuild_bytes), ','))
    end
end

main()
