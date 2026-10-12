"""Local GPU-slot scheduler (same execution model as the FedStop/PEARL campaigns).

* The scheduling unit is one run; each run is one subprocess that sees exactly one GPU
  (``CUDA_VISIBLE_DEVICES`` is the physical id, translated through any inherited mask).
* ``--slots-per-gpu`` workers share a device; *exclusive* tasks (cost measurements) wait
  for a device with no running worker and then occupy all of its slots.
* The plan is rebuilt from disk every poll, so a restart of the same command resumes:
  sealed runs with a matching fingerprint are reused, anything else is rerun from scratch.
* A driver lock prevents two schedulers on one root; workers recorded in ``status.json``
  by a killed driver are terminated on restart (identified by pid + start time + task file).
* After a worker failure no new task starts unless ``continue_on_error`` (divergence is an
  outcome, not a failure).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from ..fl.runner import is_sealed
from ..utils import atomic_write_json, read_json
from .planner import Plan, PlannedTask, build_plan
from .suites import TaskSpec


@dataclass(frozen=True)
class Slot:
    slot_id: str
    device: str  # CUDA_VISIBLE_DEVICES value ("" = CPU)


def inherited_visible_devices(env: Mapping[str, str] | None = None) -> list[str] | None:
    raw = (os.environ if env is None else env).get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    out = []
    for tok in (t.strip() for t in raw.split(",")):
        if not tok:
            continue
        if tok.lstrip("-").isdigit() and int(tok) < 0:
            break
        out.append(tok)
    return out


def parse_slots(gpus: str, slots_per_gpu: int, env: Mapping[str, str] | None = None,
                device_count: Callable[[], int] | None = None) -> list[Slot]:
    """``cpu`` | ``auto`` | ``1,2,3`` (logical ids, i.e. indices into an inherited mask)."""
    if slots_per_gpu < 1:
        raise ValueError("--slots-per-gpu must be >= 1")
    spec = gpus.strip().lower()
    if spec == "cpu":
        return [Slot(f"cpu.{j}", "") for j in range(slots_per_gpu)]
    visible = inherited_visible_devices(env)
    if spec == "auto":
        if device_count is None:
            import torch

            device_count = torch.cuda.device_count
        n = device_count()
        if visible is not None:
            n = min(n, len(visible))
        if n < 1:
            raise ValueError("--gpus auto: no CUDA device visible (use --gpus cpu)")
        logical = list(range(n))
    else:
        logical = [int(t) for t in spec.split(",") if t.strip()]
        if not logical or any(i < 0 for i in logical):
            raise ValueError(f"invalid --gpus {gpus!r}")
        if visible is not None and any(i >= len(visible) for i in logical):
            raise ValueError(f"--gpus {gpus}: the inherited CUDA_VISIBLE_DEVICES has {len(visible)} devices")
    slots, used = [], Counter()
    for i in logical:
        dev = visible[i] if visible is not None else str(i)
        for _ in range(slots_per_gpu):
            slots.append(Slot(f"gpu{dev}.{used[dev]}", dev))
            used[dev] += 1
    return slots


def worker_env(slot: Slot, base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env["CUDA_VISIBLE_DEVICES"] = slot.device
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["FEDTS_SLOT"] = slot.slot_id
    if slot.device:
        env["FEDTS_REQUIRE_CUDA"] = "1"  # the worker must fail rather than fall back to CPU
    else:
        env.pop("FEDTS_REQUIRE_CUDA", None)
    # workers import exactly the package the driver runs (also when it is not pip-installed)
    pkg_parent = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join([pkg_parent] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env


@dataclass
class Running:
    task: PlannedTask
    slots: list[Slot]
    proc: subprocess.Popen
    started: float
    log: Path
    task_file: Path


def _process_start(pid: int) -> str | None:
    try:
        import psutil

        return str(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001
        return None


def _kill_tree(proc_or_pid: subprocess.Popen | int, sig: int = signal.SIGTERM) -> None:
    pid = proc_or_pid.pid if isinstance(proc_or_pid, subprocess.Popen) else proc_or_pid
    try:
        if hasattr(os, "killpg"):
            os.killpg(pid, sig)
        elif isinstance(proc_or_pid, subprocess.Popen):
            proc_or_pid.terminate()
        else:
            import psutil

            psutil.Process(pid).terminate()
    except (ProcessLookupError, PermissionError, OSError):
        pass
    except Exception:  # noqa: BLE001 - psutil.NoSuchProcess etc.
        pass


@contextmanager
def driver_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "driver.lock"
    if lock.exists():
        try:
            info = read_json(lock)
            pid = int(info["pid"])
            if _process_start(pid) is not None and _process_start(pid) == info.get("start"):
                raise RuntimeError(f"another driver (pid {pid}) holds {lock}")
        except (ValueError, KeyError, OSError):
            pass
    atomic_write_json(lock, {"pid": os.getpid(), "start": _process_start(os.getpid()), "time": time.time()})
    try:
        yield
    finally:
        try:
            if read_json(lock).get("pid") == os.getpid():
                lock.unlink()
        except (OSError, ValueError):
            pass


def terminate_orphans(root: Path) -> list[int]:
    """Kill workers that a dead driver recorded as running in status.json."""
    status = root / "status.json"
    killed = []
    if not status.exists():
        return killed
    try:
        running = read_json(status).get("running", [])
    except (OSError, ValueError):
        return killed
    for entry in running:
        pid, start, tf = entry.get("pid"), entry.get("pid_start"), entry.get("task_file")
        if not pid or _process_start(pid) is None or _process_start(pid) != start:
            continue
        try:
            import psutil

            cmd = " ".join(psutil.Process(pid).cmdline())
        except Exception:  # noqa: BLE001
            continue
        if tf and tf in cmd:
            _kill_tree(pid)
            killed.append(pid)
    return killed


class Scheduler:
    def __init__(self, root: Path, specs: list[TaskSpec], data_root: str, slots: list[Slot], *,
                 continue_on_error: bool = False, poll: float = 5.0, python: str | None = None,
                 log: Callable[[str], None] | None = None):
        self.root, self.specs, self.slots = Path(root).resolve(), specs, slots
        self.data_root = str(Path(data_root).resolve())
        self.continue_on_error, self.poll = continue_on_error, poll
        self.python = python or sys.executable
        self.running: dict[str, Running] = {}
        self.failed: dict[str, dict[str, Any]] = {}
        self.stop = False
        self.t0 = time.time()
        self.completed_seconds: list[tuple[float, float]] = []  # (seconds, cost_hint)
        self.draining: str | None = None  # device reserved for a waiting exclusive task
        self._log = log or (lambda m: print(m, flush=True))

    # ------------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        self._log(line)
        with open(self.root / "driver.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def free_slots(self) -> list[Slot]:
        busy = {s.slot_id for r in self.running.values() for s in r.slots}
        return [s for s in self.slots if s.slot_id not in busy]

    def _slots_for(self, task: PlannedTask) -> list[Slot] | None:
        free = self.free_slots()
        if not free:
            return None
        load = Counter(s.device for r in self.running.values() for s in r.slots)
        if not task.spec.exclusive:
            free = [s for s in free if s.device != self.draining]
            if not free:
                return None
            free.sort(key=lambda s: (load[s.device], s.slot_id))  # least-loaded device first
            return [free[0]]
        devices = list(dict.fromkeys(s.device for s in self.slots))
        for dev in devices:
            dev_slots = [s for s in self.slots if s.device == dev]
            if all(s in free for s in dev_slots):
                if self.draining == dev:
                    self.draining = None
                return dev_slots
        if self.draining is None:  # reserve the least-loaded device until it drains
            self.draining = min(devices, key=lambda d: (load[d], d))
            self.log(f"draining device {self.draining} for exclusive task {task.task_id}")
        return None

    def launch(self, task: PlannedTask, slots: list[Slot]) -> None:
        tdir = self.root / "tasks"
        tdir.mkdir(parents=True, exist_ok=True)
        task_file = tdir / f"{task.task_id}.json"
        atomic_write_json(task_file, {"task_id": task.task_id, "config": task.config.to_dict(),
                                      "out_dir": str(task.out_dir), "fingerprint": task.fingerprint})
        task.out_dir.mkdir(parents=True, exist_ok=True)
        log = task.out_dir / "worker.log"
        with open(log, "ab") as fh:
            fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {task.task_id} on "
                     f"{','.join(s.slot_id for s in slots)}\n".encode())
            proc = subprocess.Popen([self.python, "-u", "-m", "fedtshdiff.orchestrate.worker", str(task_file)],
                                    stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env=worker_env(slots[0]), cwd=str(self.root),
                                    start_new_session=hasattr(os, "setsid"))
        self.running[task.task_id] = Running(task, slots, proc, time.time(), log, task_file)
        self.log(f"start {task.task_id} on {','.join(s.slot_id for s in slots)} (pid {proc.pid})")

    def reap(self) -> None:
        for tid, r in list(self.running.items()):
            code = r.proc.poll()
            if code is None:
                continue
            del self.running[tid]
            secs = time.time() - r.started
            done = r.task.out_dir / "done.json"
            if code == 0 and done.exists() and is_sealed(r.task.out_dir, r.task.fingerprint):
                status = read_json(done).get("status")
                self.completed_seconds.append((secs, r.task.cost_hint))
                self.log(f"{status} {tid} in {secs / 60:.1f} min")
            elif code == 0:
                self.failed[tid] = {"exit_code": code, "log": str(r.log), "time": time.time(),
                                    "reason": "finished without a matching seal (code or config changed?)"}
                self.log(f"FAILED {tid}: finished but its seal does not match the planned fingerprint")
            else:
                self.failed[tid] = {"exit_code": code, "log": str(r.log), "time": time.time()}
                self.log(f"FAILED {tid} (exit {code}); see {r.log}")

    def write_status(self, plan: Plan, state: str) -> None:
        now = time.time()
        running = []
        for tid, r in self.running.items():
            prog = {}
            p = r.task.out_dir / "progress.json"
            if p.exists():
                try:
                    prog = read_json(p)
                except (OSError, ValueError):
                    prog = {}
            running.append({"task_id": tid, "slots": [s.slot_id for s in r.slots], "pid": r.proc.pid,
                            "pid_start": _process_start(r.proc.pid), "task_file": str(r.task_file),
                            "elapsed": now - r.started, "round": prog.get("round"), "rounds": prog.get("rounds"),
                            "phase": prog.get("phase")})
        rate = (sum(s for s, _ in self.completed_seconds) / max(sum(c for _, c in self.completed_seconds), 1e-9)
                if self.completed_seconds else None)
        remaining = sum(t.cost_hint for t in plan.tasks if t.state != "done" and t.task_id not in self.running)
        eta = rate * remaining / max(len(self.slots), 1) if rate else None
        atomic_write_json(self.root / "status.json", {
            "state": state, "updated": now, "started": self.t0, "host": os.uname().nodename if hasattr(os, "uname") else "",
            "counts": plan.counts(), "total": len(plan.tasks),
            "done": sum(1 for t in plan.tasks if t.state == "done"),
            "running": running, "failed": self.failed, "slots": [s.slot_id for s in self.slots],
            "eta_seconds": eta,
        })

    # ------------------------------------------------------------------------------
    def run(self) -> str:
        with driver_lock(self.root):
            killed = terminate_orphans(self.root)
            if killed:
                self.log(f"terminated orphaned workers {killed}")
            prev = {}
            for sig in (signal.SIGINT, signal.SIGTERM):
                prev[sig] = signal.signal(sig, self._on_signal)
            try:
                return self._loop()
            finally:
                for sig, h in prev.items():
                    signal.signal(sig, h)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        if self.stop:
            for r in self.running.values():
                _kill_tree(r.proc, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
        self.stop = True

    def _loop(self) -> str:
        self.log(f"scheduler started: {len(self.specs)} tasks, {len(self.slots)} slots")
        while True:
            self.reap()
            plan = build_plan(self.root, self.specs, self.data_root)
            if self.stop:
                self.log("stopping: terminating workers")
                for r in self.running.values():
                    _kill_tree(r.proc)
                deadline = time.time() + 30
                while self.running and time.time() < deadline:
                    time.sleep(0.5)
                    self.reap()
                self.write_status(plan, "stopped")
                return "stopped"
            blocked = bool(self.failed) and not self.continue_on_error
            if not any(t.spec.exclusive and t.task_id not in self.failed and t.task_id not in self.running
                       for t in plan.by_state("ready")):
                self.draining = None
            if not blocked:
                for task in plan.by_state("ready"):
                    if task.task_id in self.running or task.task_id in self.failed:
                        continue
                    slots = self._slots_for(task)
                    if slots is None:
                        if task.spec.exclusive:
                            continue  # let non-exclusive tasks use remaining slots
                        if not self.free_slots():
                            break
                        continue
                    self.launch(task, slots)
            pending = [t for t in plan.tasks if t.state != "done" and t.task_id not in self.failed]
            launchable = [t for t in plan.by_state("ready") if t.task_id not in self.failed]
            if not self.running and (not pending or blocked or not launchable):
                state = "complete" if not pending and not self.failed else (
                    "failed" if self.failed else "blocked")
                if state == "blocked":
                    waits = {w for t in plan.by_state("waiting") for w in t.waiting_on}
                    self.log(f"nothing runnable; waiting on {sorted(waits)[:10]}")
                self.write_status(plan, state)
                self.log(f"scheduler finished: {state}")
                return state
            self.write_status(plan, "running")
            time.sleep(self.poll)
