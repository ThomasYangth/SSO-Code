#!/bin/bash
#SBATCH --job-name=Filt_Lorentz
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/filter_lorentz_L12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# Lorentzian filter m^2/(omega^2+m^2) (KrylovChain convention) for figure_B:
#   kicked Ising L = 12: Sum Z_x, Sum X_x (K = 0), Z_0 (all momenta), 50 widths
#     -> KickedIsing_g0.9J1.0h0.809_FloquetHardCut_Data/lorentzcut_L12_*_l0.0.npz
#   hx = 0.9, hz = 0.03 current YZcur, L = 12, K = 0, o4, 60 widths, all filters
#   (incl. lorentz2 = squared Lorentzian, for comparison only)
#     -> Ising_J1.0X0.9Z0.03_StaticFilter_Data/filt_L12_K0_YZcur_l0.0o4.npz
M="--model=KickedIsing --g=0.9 --J=1.0 --h=0.809 --L=12 --usegpu=True"
python $D/filter_floquet.py $M --K=0 --init_op=Zsum --filters=lorentz --n_width=50 --override=True
python $D/filter_floquet.py $M --K=0 --init_op=Xsum --filters=lorentz --n_width=50 --override=True
python $D/filter_floquet.py $M --K=none --init_op=Z0 --filters=lorentz --n_width=50 --override=True
python $D/filter_static.py --model=Ising --J=1.0 --hx=0.9 --hz=0.03 --L=12 --K=0 \
    --init_op=YZcur --pows_max=3 --n_width=60 --filters=avg,hard,gauss,lorentz,lorentz2 --usegpu=True --override=True
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
