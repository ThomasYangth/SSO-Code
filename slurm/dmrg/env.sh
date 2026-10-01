# Shared environment for the SlowopDMRG launchers (source, do not execute).
#
# Julia 1.10.2 as in the production runs. JULIA_DEPOT_PATH stacks a private
# overlay depot (precompile caches; per-job when SLURM_JOB_ID is set so
# concurrent CUDA-extension precompiles do not race) over the shared depot that
# already holds the pinned packages. CUDA.jl is NOT a hard dependency of
# SlowopDMRG (weakdep -> SlowopDMRGCUDAExt): on GPU runs `using CUDA` resolves
# through the stacked global environment @v1.10 of the shared depot, which
# carries CUDA.jl 5.8.3 (the version used for every plotted GPU run).
#
# Override SLOWOP_SHARED_DEPOT / SLOWOP_OVERLAY_DEPOT (default $SSO_OUTPUT/julia_depot)
# / SSO_OUTPUT as needed. First use in a fresh overlay precompiles (~5 min).
# NOTE: the julia/1.10.2 module sets OMP_NUM_THREADS=invalid, so set the BLAS
# thread count AFTER loading it (SLOWOP_THREADS, default 1 as in production).
module load julia/1.10.2
export OMP_NUM_THREADS=${SLOWOP_THREADS:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PROJ="$REPO/julia/SlowopDMRG"
SLOWOP_SHARED_DEPOT=${SLOWOP_SHARED_DEPOT:-/scratch/gpfs/DABANIN/ty1475/julia_depot/main}
SLOWOP_OVERLAY_DEPOT=${SLOWOP_OVERLAY_DEPOT:-${SSO_OUTPUT:-$REPO/output}/julia_depot}
export JULIA_DEPOT_PATH="${SLOWOP_OVERLAY_DEPOT}${SLURM_JOB_ID:+/job_${SLURM_JOB_ID}}:${SLOWOP_OVERLAY_DEPOT}:${SLOWOP_SHARED_DEPOT}"
export JULIA_PKG_OFFLINE=${JULIA_PKG_OFFLINE:-true}
