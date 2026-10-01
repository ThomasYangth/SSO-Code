# SlowopDMRG test suite (CPU only). Run from julia/SlowopDMRG:
#     julia --project=. test/runtests.jl        (or Pkg.test())
# ~26 min on 4 CPU cores (job 14826936). The rolling-resume regression is a separate shell
# test (test/rolling_resume_smoke.sh) because it drives the ladder script.
using Test

# The end-to-end tests go through run_dmrg_for_size, which saves MPS files and
# tracking CSVs; keep them out of the production data directory.
if !haskey(ENV, "SLOWOP_DATA_DIR")
    ENV["SLOWOP_DATA_DIR"] = joinpath(get(ENV, "SSO_OUTPUT",
        normpath(joinpath(@__DIR__, "..", "..", "..", "output"))), "dmrg_test")
end

@testset "SlowopDMRG" begin
    include("lobpcg_smoke.jl")          # LOBPCG helpers + L=7 end-to-end smoke
    include("oracle_small_L.jl")        # L=6 sweeper vs exact frontier
    include("lobpcg_stepboundary.jl")   # CholQR regression at ε-step boundaries
end
