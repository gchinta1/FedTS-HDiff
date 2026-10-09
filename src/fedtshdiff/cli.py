"""``fedts`` command line.

    fedts prepare-data --dataset ptbxl --raw-root raw --data-root data [--download]
    fedts run --config my.yaml --set method=fedavg --set seed=2 --out results/adhoc/x
    fedts campaign --root results/revision --data-root data --gpus auto --slots-per-gpu 4 [--only main,ablation]
                   [--dry-run] [--no-tuning] [--continue-on-error] [--seeds 1,2,3,4,5]
    fedts preflight --root results/preflight --data-root data --gpus auto --slots-per-gpu 4
    fedts status --root results/revision [--watch 10]
    fedts analyse --root results/revision [--data-root data]
    fedts describe-data --data-root data --out results/revision/analysis/tables
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from .utils import atomic_write_json, read_json, runtime_record


def _cmd_prepare(a: argparse.Namespace) -> int:
    from .data.prepare import prepare

    opts = json.loads(a.options) if a.options else None
    for ds in a.dataset:
        path = prepare(ds, raw_root=a.raw_root, data_root=a.data_root, download=a.download, options=opts)
        meta = read_json(path / "meta.json")
        print(f"{ds}: {meta['n_clients']} clients, {sum(meta['client_windows'])} windows -> {path}")
    return 0


def _cmd_run(a: argparse.Namespace) -> int:
    from .config import load_config
    from .fl.runner import run_experiment

    cfg = load_config(a.config, a.set)
    res = run_experiment(cfg, a.out)
    print(json.dumps({k: v for k, v in res.get("final", {}).items() if not isinstance(v, list)}, indent=2))
    return 0 if res["status"] in ("complete", "diverged") else 1


def _specs(a: argparse.Namespace):
    from .orchestrate.suites import DATASETS, GROUPS, SEEDS, campaign

    groups = a.only.split(",") if a.only else list(GROUPS)
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise SystemExit(f"unknown groups {sorted(unknown)}; choose from {GROUPS}")
    seeds = tuple(int(s) for s in a.seeds.split(",")) if a.seeds else SEEDS
    datasets = tuple(a.datasets.split(",")) if a.datasets else DATASETS
    return campaign(groups, tuned=not a.no_tuning, seeds=seeds, datasets=datasets)


def _print_plan(plan, limit: int = 15) -> None:
    print(f"{'group':<12} {'done':>6} {'ready':>6} {'waiting':>8}")
    for g, c in plan.counts().items():
        print(f"{g:<12} {c['done']:>6} {c['ready']:>6} {c['waiting']:>8}")
    ready = plan.by_state("ready")
    print(f"\n{len(plan.tasks)} tasks; {len(ready)} ready. Next ready (priority order):")
    for t in ready[:limit]:
        print(f"  {t.task_id}{'  [exclusive]' if t.spec.exclusive else ''}")
    waits = sorted({w for t in plan.by_state("waiting") for w in t.waiting_on})
    if waits:
        print(f"Waiting on: {', '.join(waits[:12])}{' ...' if len(waits) > 12 else ''}")


def _cmd_campaign(a: argparse.Namespace) -> int:
    from .orchestrate.planner import build_plan, write_manifest
    from .orchestrate.scheduler import Scheduler, parse_slots

    root = Path(a.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    specs = _specs(a)
    write_manifest(root, specs, {"data_root": str(Path(a.data_root).resolve()), "created": time.time(),
                                 "tuned": not a.no_tuning})
    plan = build_plan(root, specs, str(Path(a.data_root).resolve()))
    _print_plan(plan)
    if a.dry_run:
        return 0
    atomic_write_json(root / "env.json", runtime_record())
    slots = parse_slots(a.gpus, a.slots_per_gpu)
    sched = Scheduler(root, specs, str(Path(a.data_root).resolve()), slots, continue_on_error=a.continue_on_error,
                      poll=a.poll)
    state = sched.run()
    print(f"campaign {state}")
    if state == "complete" and not a.no_analyse:
        from .analysis.report import analyse

        print(f"report: {analyse(root, a.data_root) / 'report.md'}")
    return 0 if state in ("complete",) else 1


def _cmd_preflight(a: argparse.Namespace) -> int:
    """Environment checks + two-round smoke of every variant on the real prepared data."""
    import torch

    from .data.store import verify_prepared
    from .orchestrate.planner import build_plan, write_manifest
    from .orchestrate.scheduler import Scheduler, parse_slots
    from .orchestrate.suites import DATASETS, smoke_campaign

    root = Path(a.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report: dict = {"runtime": runtime_record(), "checks": {}}
    datasets = a.datasets.split(",") if a.datasets else list(DATASETS)
    for ds in datasets:
        try:
            report["checks"][f"data:{ds}"] = verify_prepared(a.data_root, ds)
        except Exception as exc:  # noqa: BLE001
            report["checks"][f"data:{ds}"] = f"FAIL: {exc}"
    free = shutil.disk_usage(root).free / 2**30
    report["checks"]["disk_free_gib"] = free
    report["checks"]["cuda"] = torch.cuda.is_available()
    slots = parse_slots(a.gpus, a.slots_per_gpu)
    report["slots"] = [s.slot_id for s in slots]
    bad = [k for k, v in report["checks"].items() if isinstance(v, str) and v.startswith("FAIL")]
    if free < 20:
        bad.append("disk_free_gib < 20")
    if bad:
        report["status"] = "failed"
        atomic_write_json(root / "preflight.json", report)
        print("preflight failed:", bad)
        return 1
    specs = smoke_campaign([d for d in datasets])
    write_manifest(root, specs, {"data_root": str(Path(a.data_root).resolve()), "smoke": True})
    _print_plan(build_plan(root, specs, str(Path(a.data_root).resolve())))
    state = Scheduler(root, specs, str(Path(a.data_root).resolve()), slots, continue_on_error=True,
                      poll=2.0).run()
    status = read_json(root / "status.json")
    report["status"] = "passed" if state == "complete" and not status["failed"] else "failed"
    report["smoke"] = {"state": state, "tasks": status["total"], "done": status["done"], "failed": status["failed"]}
    atomic_write_json(root / "preflight.json", report)
    print(f"preflight {report['status']}: {status['done']}/{status['total']} smoke runs sealed")
    return 0 if report["status"] == "passed" else 1


def _fmt_s(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _cmd_status(a: argparse.Namespace) -> int:
    path = Path(a.root) / "status.json"
    while True:
        if not path.exists():
            print("no status.json yet")
        else:
            st = read_json(path)
            age = time.time() - st["updated"]
            print(f"state={st['state']}  done {st['done']}/{st['total']}  running {len(st['running'])}  "
                  f"failed {len(st['failed'])}  ETA {_fmt_s(st.get('eta_seconds'))}  (updated {age:.0f}s ago)")
            for g, c in st["counts"].items():
                print(f"  {g:<12} done {c['done']:>4}  ready {c['ready']:>4}  waiting {c['waiting']:>4}")
            for r in sorted(st["running"], key=lambda r: r["slots"][0]):
                prog = f"{r['round']}/{r['rounds']} {r['phase']}" if r.get("round") is not None else "starting"
                print(f"  [{','.join(r['slots'])}] {r['task_id']}  {prog}  {_fmt_s(r['elapsed'])}")
            for tid, f in st["failed"].items():
                print(f"  FAILED {tid}: exit {f['exit_code']}  log {f['log']}")
        if not a.watch:
            return 0
        time.sleep(a.watch)
        print()


def _cmd_analyse(a: argparse.Namespace) -> int:
    from .analysis.report import analyse

    out = analyse(a.root, a.data_root)
    print(f"report: {out / 'report.md'}")
    return 0


def _cmd_describe(a: argparse.Namespace) -> int:
    from .analysis.datasets import dataset_table

    t = dataset_table(a.data_root, out=Path(a.out) if a.out else None)
    print(t.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fedts", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare-data")
    s.add_argument("--dataset", nargs="+", required=True)
    s.add_argument("--raw-root", default="raw")
    s.add_argument("--data-root", default="data")
    s.add_argument("--download", action="store_true")
    s.add_argument("--options", help="JSON dict of builder keyword arguments")
    s.set_defaults(fn=_cmd_prepare)

    s = sub.add_parser("run")
    s.add_argument("--config")
    s.add_argument("--set", action="append", default=[])
    s.add_argument("--out", required=True)
    s.set_defaults(fn=_cmd_run)

    for name, fn in (("campaign", _cmd_campaign), ("preflight", _cmd_preflight)):
        s = sub.add_parser(name)
        s.add_argument("--root", required=True)
        s.add_argument("--data-root", default="data")
        s.add_argument("--gpus", default="auto")
        s.add_argument("--slots-per-gpu", type=int, default=4)
        s.add_argument("--datasets", help="comma-separated subset of mitbih,ptbxl,electricity,metrla")
        if name == "campaign":
            s.add_argument("--only", help="comma-separated groups")
            s.add_argument("--seeds")
            s.add_argument("--no-tuning", action="store_true", help="use the paper's baseline values")
            s.add_argument("--dry-run", action="store_true")
            s.add_argument("--continue-on-error", action="store_true")
            s.add_argument("--no-analyse", action="store_true")
            s.add_argument("--poll", type=float, default=5.0)
        s.set_defaults(fn=fn)

    s = sub.add_parser("status")
    s.add_argument("--root", required=True)
    s.add_argument("--watch", type=float, default=0)
    s.set_defaults(fn=_cmd_status)

    s = sub.add_parser("analyse")
    s.add_argument("--root", required=True)
    s.add_argument("--data-root")
    s.set_defaults(fn=_cmd_analyse)

    s = sub.add_parser("describe-data")
    s.add_argument("--data-root", default="data")
    s.add_argument("--out")
    s.set_defaults(fn=_cmd_describe)

    a = p.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
