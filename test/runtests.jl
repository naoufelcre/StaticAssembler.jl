# Shared imports, fixtures, and reference calculations are loaded once before selected families.
include("TestSupport.jl")

const FAMILIES = ("integration", "local", "projection", "assembly", "lifecycle")
const SELECTED = Set(ARGS)
unknown = setdiff(SELECTED, FAMILIES)
isempty(unknown) || throw(ArgumentError(
    "unknown test family $(join(sort!(collect(unknown)), ", ")); choose $(join(FAMILIES, ", "))"))

function run_testset(body, name, family)
    isempty(SELECTED) || family in SELECTED || return nothing
    @testset "$name" begin
        body()
    end
    return nothing
end

for (family, file) in zip(FAMILIES, (
        "TestIntegration.jl", "TestLocalDiscretization.jl",
        "TestExtensionProjection.jl", "TestAssembly.jl", "TestLifecycle.jl"))
    isempty(SELECTED) || family in SELECTED || continue
    @testset "$family" begin
        include(file)
    end
end


if !isempty(SA4_RECORDS)
    println("\n===== Cell-projection differences (observed / tolerance) =====")
    for record in SA4_RECORDS
        println(rpad(record.label, 56),
                " left=", record.left_error, " / ", record.left_tol,
                "  matrix=", record.matrix_error, " / ", record.matrix_tol,
                "  vector=", record.vector_error, " / ", record.vector_tol)
    end
end
if !isempty(SA4_ALLOCATIONS)
    println("\n===== Warmed projection allocations (bytes) =====")
    for allocation in SA4_ALLOCATIONS
        println(rpad(allocation.label, 16),
                " production=", allocation.production_bytes,
                " scalar_reference=", allocation.reference_bytes,
                " julia_threads=", allocation.julia_threads)
    end
end
