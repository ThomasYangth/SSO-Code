"""Build the slim data for the DMRG-attempt figures (data/attempt_dmrg/).

Writes two CSV files:

``dmrg_points.csv``
    One row per converged DMRG result of the ε-ladder campaign
    (julia/SlowopDMRG, ``scheduling=:eps_constrained, local_solver=:lobpcg``):
    ``eps = ⟨C_H²⟩`` and ``nu = ⟨M⟩`` of the saved operator MPS, τ = 1/√eps
    (static convention).  Columns::

        L, chi, step, eps_target, eps, nu, theta_sq_ema, run, job, source,
        saved, frozen_tail, plotted, wall_s, note

    ``saved=1``: read from an HDF5 checkpoint (``done_stepNN.h5`` or a
    one-step output); ``saved=0``: the run completed and printed its result,
    but the MPS was not written (the two χ-refinement points) — read from the
    slurm log.  ``frozen_tail=1``: a ladder step that did not move off the
    previous one (|Δeps|/eps < 1e-3 — the χ wall: θ² at its 1e14 ceiling);
    these are kept for the record but not plotted (``plotted=0``).

``ed_front_L12.csv``
    The exact L=12 ν(⟨C_H²⟩) frontier (OBC full-space ED soft solver, now
    ``sso.nutau`` full-space OBC soft solver): for each θ the leading
    eigenpair (``As[0]``, ``Bs[0]``) = (⟨C_H²⟩, ⟨M⟩) of the archived
    ``L12_theta{θ}l0.0k2o4.npz``; the 17 files with θ ≥ 0.3981 (the range
    plotted in the original figures).

Dropped on purpose (see docs/dmrg_attempt.md): the χ=256 and χ=512 L=12
"step 10" points (jobs 13367089, 13325835 were killed by the time limit
mid-step, the original plots used their last-sweep log values), the anneal=1
step-10 point spliced into the original χ=128 L=12 curve (13325137_1), and the
χ=128 seed states copied into the χ=256 continuation directories.

Usage (login node is fine: reads scalars only, lazily)::

    python scripts/extract/dmrg_points.py \
        --archive=<SavedMPS dir containing hk_scan_cmp/> \
        --logs=<original slurmlogs dir> \
        --ed=<Ising_J1.0X0.905Z0.809OBC_DSsoft_Data dir>

``--archive`` defaults to ``sso.config.output_dir("dmrg")`` (where a re-run
of the slurm/dmrg launchers writes the same ``hk_scan_cmp/...`` layout).
HDF5 scalars are read with h5py when available, else with ``h5dump``.
"""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sso import config  # noqa: E402

# --------------------------------------------------------------------------
# Provenance tables (from sacct and the launchers of the original campaign)
# --------------------------------------------------------------------------

# job id -> (start, end, launcher). A done_step file is attributed to the job
# whose [start, end] window contains its write timestamp.
JOBS = {
    "13102819":   ("2026-08-28T12:50:55", "2026-08-29T00:56:13", "ed_L12_single.slurm (CPU 16c)"),
    "13177208":   ("2026-08-29T23:55:53", "2026-08-30T12:01:18", "ed_L12_single.slurm (CPU 16c)"),
    "13199108":   ("2026-08-30T12:03:19", "2026-08-30T23:20:55", "ed_L12_single.slurm (CPU 16c)"),
    "13306156":   ("2026-09-01T12:10:32", "2026-09-01T13:45:05", "ed_chi_refine_mig.slurm (MIG)"),
    "13312300_1": ("2026-09-01T14:48:15", "2026-09-01T19:17:03", "ed_step9_anneal_mig.slurm task 1 (MIG)"),
    "13325137_0": ("2026-09-01T21:15:51", "2026-09-02T03:37:03", "ed_gpu128_anneal.slurm task 0 (MIG)"),
    "13325835":   ("2026-09-01T21:36:23", "2026-09-02T20:41:42", "ed_gpu512_step9.slurm (A100 40GB)"),
    "13342673_0": ("2026-09-02T09:58:37", "2026-09-03T01:57:29", "ed_bigL_chi128.slurm task 0 (MIG)"),
    "13342673_1": ("2026-09-02T09:58:37", "2026-09-03T10:03:52", "ed_bigL_chi128.slurm task 1 (MIG)"),
    "13342673_2": ("2026-09-02T09:58:37", "2026-09-03T10:03:52", "ed_bigL_chi128.slurm task 2 (MIG)"),
    "13367090":   ("2026-09-02T21:37:01", "2026-09-03T00:07:35", "ed_step8_chi512_gpu40.slurm (A100 40GB)"),
    "13384642":   ("2026-09-03T09:27:13", "2026-09-03T17:58:16", "ed_L18_chi256.slurm (MIG)"),
    "13399489_1": ("2026-09-03T16:01:03", "2026-09-03T23:31:54", "ed_bigL_chi128.slurm resubmit task 1 (MIG)"),
    "13399489_2": ("2026-09-03T16:01:03", "2026-09-04T11:16:21", "ed_bigL_chi128.slurm resubmit task 2 (MIG)"),
    "13399746":   ("2026-09-03T16:06:11", "2026-09-04T16:11:27", "ed_cont_chi256.slurm L=24 (MIG)"),
    "13399747":   ("2026-09-03T16:06:11", "2026-09-04T16:11:27", "ed_cont_chi256.slurm L=30 (MIG)"),
    "13461754":   ("2026-09-04T21:02:48", "2026-09-05T06:55:18", "ed_cont_chi256.slurm L=24 resubmit (MIG)"),
    "13461755":   ("2026-09-04T21:02:48", "2026-09-07T13:02:43", "ed_cont_chi256.slurm L=30 resubmit (MIG)"),
}

# Ladder series: (L, chi, checkpoint dir under hk_scan_cmp/, candidate jobs,
# seed step copied in from a chi=128 run (dropped) or None, note).
LADDERS = [
    (12, 128, "L12_lobpcg_sinmc_repgpu128_a0", ["13325137_0"], None,
     "anneal=0 (quench) ladder, MAX_STEPS=10; the single run used for the whole L=12 chi=128 curve"),
    (18, 128, "L18_lobpcg_sinmc_repbigL18", ["13342673_0"], None, "anneal=1"),
    (24, 128, "L24_lobpcg_sinmc_repbigL24", ["13342673_1", "13399489_1"], None, "anneal=1"),
    (30, 128, "L30_lobpcg_sinmc_repbigL30", ["13342673_2", "13399489_2"], None, "anneal=1"),
    (18, 256, "L18_lobpcg_sinmc_repbigL18_chi256", ["13384642"], 8,
     "anneal=1; continued from the L=18 chi=128 step-8 state"),
    (24, 256, "L24_lobpcg_sinmc_repbigL24_chi256", ["13399746", "13461754"], 7,
     "anneal=1; continued from the L=24 chi=128 step-7 state"),
    (30, 256, "L30_lobpcg_sinmc_repbigL30_chi256", ["13399747", "13461755"], 5,
     "anneal=1; continued from the L=30 chi=128 step-5 state"),
    (12, 512, "L12_lobpcg_sinmc_repgpu512", ["13325835"], 8,
     "anneal=1; continued from the CPU chi=128 ladder (repchi128) step-8 state; "
     "step 10 was killed by the time limit and is not included"),
]

FROZEN_REL = 1e-3


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def h5_scalars(path, keys):
    """Read scalar datasets ``keys`` from an HDF5 file (h5py, else h5dump)."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            out = {}
            for k in keys:
                if k in f:
                    v = f[k][()]
                    out[k] = v.decode() if isinstance(v, bytes) else v
            return out
    except ImportError:
        pass
    if shutil.which("h5dump") is None:
        raise RuntimeError("need h5py or h5dump to read %s" % path)
    out = {}
    for k in keys:
        r = subprocess.run(["h5dump", "-m", "%.17g", "-d", "/" + k, path],
                           capture_output=True, text=True)
        m = re.search(r"\(0\): (.+)", r.stdout)
        if r.returncode == 0 and m:
            s = m.group(1).strip()
            out[k] = s.strip('"') if s.startswith('"') else float(s)
    return out


def find_log(logs, job):
    hits = glob.glob(os.path.join(logs, "*", "slurm-%s.out" % job))
    return hits[0] if hits else None


def step_walls(logs, jobs):
    """{step: wall seconds} from the ``[step N] done. ... wall=Xs`` log lines."""
    walls = {}
    for job in jobs:
        log = find_log(logs, job)
        if log is None:
            continue
        for line in open(log, errors="replace"):
            m = re.match(r"\[step (\d+)\] done\..*wall=([0-9.]+)s", line)
            if m:
                walls[int(m.group(1))] = float(m.group(2))
    return walls


def attribute(ts, candidates):
    t = datetime.fromisoformat(ts[:19])
    for job in candidates:
        start, end, _ = JOBS[job]
        if datetime.fromisoformat(start) <= t <= datetime.fromisoformat(end):
            return job
    raise ValueError("timestamp %s not inside any of %s" % (ts, candidates))


def log_line(logs, job, pattern):
    log = find_log(logs, job)
    if log is None:
        raise FileNotFoundError("log of job %s not found under %s" % (job, logs))
    for line in open(log, errors="replace"):
        m = re.search(pattern, line)
        if m:
            return m, log
    raise ValueError("pattern %r not in %s" % (pattern, log))


# --------------------------------------------------------------------------

def ladder_rows(archive, logs):
    rows = []
    keys = ["step", "eps_target", "eps_achieved", "nu", "theta_sq_ema", "variant", "timestamp"]
    for L, chi, sub, jobs, seed, note in LADDERS:
        d = os.path.join(archive, "hk_scan_cmp", sub)
        walls = step_walls(logs, jobs)
        prev_eps = None
        for path in sorted(glob.glob(os.path.join(d, "done_step*.h5"))):
            v = h5_scalars(path, keys)
            step = int(v["step"])
            if seed is not None and step == seed:
                assert not v["variant"].endswith("_chi%d" % chi), path
                continue                                   # chi=128 seed copy
            job = attribute(v["timestamp"], jobs)
            eps, nu = float(v["eps_achieved"]), float(v["nu"])
            frozen = prev_eps is not None and abs(eps - prev_eps) / prev_eps < FROZEN_REL
            prev_eps = eps
            rows.append(dict(
                L=L, chi=chi, step=step, eps_target=float(v["eps_target"]), eps=eps, nu=nu,
                theta_sq_ema=float(v["theta_sq_ema"]), run=v["variant"], job=job,
                source="hk_scan_cmp/%s/%s" % (sub, os.path.basename(path)), saved=1,
                frozen_tail=int(frozen), plotted=int(not frozen), wall_s=walls.get(step, ""),
                note="%s; launcher %s" % (note, JOBS[job][2])))
    return rows


def single_rows(archive, logs):
    """L=12 chi=256 / chi=512 points from one-step runs seeded by repchi128 step 8."""
    rows = []
    # chi=256 step 9: annealed one-step continuation, saved.
    path = os.path.join(archive, "hk_scan_cmp", "L12_step9_anneal", "step9_anneal1_chi256.h5")
    v = h5_scalars(path, ["eps_target", "eps_achieved", "nu", "anneal_sweeps", "timestamp"])
    m, _ = log_line(logs, "13312300_1", r"STEP9 anneal=1 chi=256 .* wall=([0-9.]+)s")
    rows.append(dict(
        L=12, chi=256, step=9, eps_target=float(v["eps_target"]), eps=float(v["eps_achieved"]),
        nu=float(v["nu"]), theta_sq_ema="", run="step9_anneal1_chi256", job="13312300_1",
        source="hk_scan_cmp/L12_step9_anneal/step9_anneal1_chi256.h5", saved=1, frozen_tail=0,
        plotted=1, wall_s=float(m.group(1)),
        note="one step eps8->eps9 at chi=256, 6 sweeps, anneal=1, from repchi128 step 8; "
             "launcher %s" % JOBS["13312300_1"][2]))
    # chi=256 / chi=512 step 8: chi refinements at the step-8 target; MPS not saved.
    for chi, job in ((256, "13306156"), (512, "13367090")):
        pat = (r"REFINE step=8 eps_tgt=([0-9.e-]+) chi=%d .*\| chi%d: eps=([0-9.e-]+) "
               r"nu=([0-9.e-]+) .*wall=([0-9.]+)s" % (chi, chi))
        m, log = log_line(logs, job, pat)
        rows.append(dict(
            L=12, chi=chi, step=8, eps_target=float(m.group(1)), eps=float(m.group(2)),
            nu=float(m.group(3)), theta_sq_ema="", run="chi_refine_step8_chi%d" % chi, job=job,
            source="slurmlogs/%s (REFINE line)" % os.path.relpath(log, logs), saved=0,
            frozen_tail=0, plotted=1, wall_s=float(m.group(4)),
            note="chi refinement at the step-8 target, %d sweeps, no anneal, from repchi128 step 8; "
                 "run completed but save_mps=false, value from the log (6 significant digits); "
                 "launcher %s" % (4 if chi == 256 else 6, JOBS[job][2])))
    return rows


def ed_front(ed_dir):
    rows = []
    for path in glob.glob(os.path.join(ed_dir, "L12_theta*l0.0k2o4.npz")):
        theta = float(re.search(r"theta([0-9.]+)l", os.path.basename(path)).group(1))
        if theta < 0.398:
            continue
        with np.load(path) as z:                      # lazy: only As, Bs are read
            rows.append(dict(theta=theta, eps=float(z["As"][0]), nu=float(z["Bs"][0]),
                             source=os.path.basename(path)))
    rows.sort(key=lambda r: r["eps"])
    assert len(rows) == 17, len(rows)
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(r[k]) if isinstance(r[k], float) else r[k]) for k in fields})
    print("wrote %s (%d rows)" % (path, len(rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", default=config.output_dir("dmrg"))
    ap.add_argument("--logs", required=True)
    ap.add_argument("--ed", required=True)
    a = ap.parse_args()

    out = config.data_dir("attempt_dmrg")
    os.makedirs(out, exist_ok=True)
    rows = ladder_rows(a.archive, a.logs) + single_rows(a.archive, a.logs)
    rows.sort(key=lambda r: (r["L"], r["chi"], r["step"]))
    write_csv(os.path.join(out, "dmrg_points.csv"), rows,
              ["L", "chi", "step", "eps_target", "eps", "nu", "theta_sq_ema", "run", "job",
               "source", "saved", "frozen_tail", "plotted", "wall_s", "note"])
    write_csv(os.path.join(out, "ed_front_L12.csv"), ed_front(a.ed), ["theta", "eps", "nu", "source"])


if __name__ == "__main__":
    main()
