"""Resume-safe seed-sweep driver for the roman_features/ ablation matrix.

Runs every (ablation_id, feature_set) job of configs/ablation_matrix.json for
the locked 10 SEEDS of src/registry.py — the same seeds as the original
suite ([1234, 100, 0, 2024, 42, 3, 123, 13, 64, 2026]). Seed 2026 was already
executed in the first batch and is skipped automatically.

- A job is skipped iff its folds_summary.json exists (exit code 0 alone is
  never trusted — the file is written only after all folds finish).
- Subprocesses get the single-threaded BLAS env of the original runner
  (run_ablation.py also sets it internally).
- Job stdout/stderr go to results/logs/job_<aid>_seed<seed>.log (overwritten
  on retry); nothing is written outside roman_features/.

Usage (from repo root):
    .venv/bin/python roman_features/code/run_seed_sweep.py --workers 7
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROMAN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROMAN.parent / "src"))
sys.path.insert(0, str(_ROMAN / "code"))

from registry import SEEDS  # noqa: E402

RESULTS = _ROMAN / "results"
LOGS = RESULTS / "logs"
MATRIX = json.loads((_ROMAN / "configs" / "ablation_matrix.json").read_text())
PY = sys.executable


def job_done(aid, seed):
    return (RESULTS / aid / f"seed{seed}" / "folds_summary.json").exists()


def run_one(job):
    aid, feature_set, seed = job
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"job_{aid}_seed{seed}.log"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    cmd = [PY, str(_ROMAN / "code" / "run_ablation.py"),
           "--ablation-id", aid, "--feature-set", feature_set,
           "--seed", str(seed), "--fold", "all"]
    with open(log_path, "w") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=_ROMAN.parent, env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    ok = proc.returncode == 0 and job_done(aid, seed)
    return aid, seed, ok, log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="subset of the locked seeds (default: all 10)")
    args = ap.parse_args()

    seeds = args.seeds if args.seeds is not None else list(SEEDS)
    jobs = [(r["ablation_id"], r["feature_set"], s)
            for r in MATRIX["runs"] if r.get("run", True) for s in seeds]
    pending = [j for j in jobs if not job_done(j[0], j[2])]
    print(f"{len(jobs)} total jobs, {len(jobs) - len(pending)} already done, "
          f"{len(pending)} pending, workers={args.workers}", flush=True)

    n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in pending]
        for i, fut in enumerate(as_completed(futs), 1):
            aid, seed, ok, log_path = fut.result()
            if not ok:
                n_fail += 1
            print(f"[{i}/{len(pending)}] {aid} seed{seed}: "
                  f"{'OK' if ok else 'FAILED'} (log {log_path.name})",
                  flush=True)
    print(f"done: {len(pending) - n_fail} ok, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
