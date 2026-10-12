"""Static TI-ansatz tables (Kim-Huse, q = 0, o4) for figure_G (c).

    python scripts/extract/ti_kimhuse_o4.py --src=<directory of nutau_rtp_M*_o4.dat>

Copies nutau_rtp_M{6..12}_o4.dat verbatim into data/kimhuse_three_panel/ti_rtp_o4/.
These come from OGH's nutau/lobpcg/ham_mf_driver_rtp.jl (ORDER = 4): the identity is
absent from the TI basis and H, H^2, H^3 are removed as "restricted true powers" (the
bulk strings of H^k with support <= M + 1, Gram-Schmidt orthonormalised), header
`... q=0 M=.. o=4 deflated=restricted-true-powers solver=mf-L1L1-gpures-rtp`.  The ED
counterpart is the K = 0, o4 grid (I..H^3 removed exactly); at small tau the TI curves lie
2-5 % above ED (L = 12 vs M = 12), attributed to the truncated powers.
--src is required (on Della the tables are archived in the TI workstream's OGH/data).
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sso.cli import find_value, parse_argv
from sso.config import data_dir

args = parse_argv()
src = find_value(args, "src", str, None)
if src is None:
    raise SystemExit("--src=<directory of nutau_rtp_M{6..12}_o4.dat> is required")
out = os.path.join(data_dir("kimhuse_three_panel"), "ti_rtp_o4")
os.makedirs(out, exist_ok=True)
for M in range(6, 13):
    name = f"nutau_rtp_M{M}_o4.dat"
    shutil.copy(os.path.join(src, name), os.path.join(out, name))
print("wrote", out)
