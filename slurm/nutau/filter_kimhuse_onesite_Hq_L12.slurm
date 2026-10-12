#!/bin/bash
#SBATCH --job-name=KH_onesite
#SBATCH --output=slurmlogs/tmp/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=32G
# Submit from the repository root with SSO_OUTPUT pointing at scratch:
#     mkdir -p slurmlogs/tmp && SSO_OUTPUT=/path/to/scratch sbatch slurm/nutau/filter_kimhuse_onesite_Hq_L12.slurm
: "${SSO_OUTPUT:?export SSO_OUTPUT (raw-output root) before sbatch}"
module purge
module load anaconda3/2024.2
conda activate cuspin
D=scripts/compute/nutau
# figure_G (a), Kim-Huse (J=1, hx=0.905, hz=0.809), L = 12, Lorentzian filter:
#  1. optimal single-site Pauli direction n (sum_x n.sigma_x, o4) maximising nu(tau = inf);
#  2. that sum_x n.sigma_x (o4) under the Lorentzian (file tag OPT1S);
#  3. the longest-wavelength energy-density mode H_q, q = 2 pi / L (K = 1, nothing projected).
#   -> Ising_J1.0X0.905Z0.809_StaticFilter_Data/{onesite_opt_L12_o4, filt_L12_K0_OPT1S_l0.0o4,
#      filt_L12_K1_Hdens_l0.0o0}.npz
M="--model=Ising --J=1.0 --hx=0.905 --hz=0.809 --L=12"
python $D/optimal_onesite.py $M --pows_max=3 --usegpu=True
MIX=$(python -c "import numpy as np, os; n=np.load(os.path.join(os.environ['SSO_OUTPUT'], 'Ising_J1.0X0.905Z0.809_StaticFilter_Data/onesite_opt_L12_o4.npz'))['sum_n']; print(f'Xsum:{n[0]:.10f},Ysum:{n[1]:.10f},Zsum:{n[2]:.10f}')")
echo "optimal mixture: $MIX"
python $D/filter_static.py $M --K=0 --pows_max=3 --mix=$MIX --mix_tag=OPT1S --n_width=60 \
    --filters=lorentz,gauss,hard,avg --usegpu=True --override=True
python $D/filter_static.py $M --K=1 --init_op=Hdens --n_width=60 \
    --filters=lorentz,gauss,hard,avg --usegpu=True --override=True
today=$(date +%Y-%m-%d)
mkdir -p slurmlogs/$today
mv slurmlogs/tmp/slurm-${SLURM_JOB_ID}.out slurmlogs/$today/slurm-${SLURM_JOB_ID}.out
