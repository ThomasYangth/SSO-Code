# Environment of the production TI-ansatz runs (Princeton Della), sourced by the
# launchers in this directory. Julia 1.10.2 with the pinned julia/TIAnsatz
# Manifest (CUDA.jl 6.3.1, CUDA runtime 12.9 artifact via LocalPreferences.toml).
# Source from the repository root.
module load anaconda3/2024.2
module load julia/1.10.2
module load cudatoolkit/12.9 2>/dev/null || true   # GPU nodes only; runtime comes from the artifact
conda activate julia
unset LD_LIBRARY_PATH
export JULIA_PROJECT="$PWD/julia/TIAnsatz"

# log to slurmlogs/<date>/<name>.out (stderr merged)
ti_log() {
    local dir="${SSO_LOGS:-slurmlogs}/$(date +%Y-%m-%d)"
    mkdir -p "$dir"
    exec > "$dir/$1.out" 2>&1
}
