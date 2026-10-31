"""Per-job experiment report: settings, data, models, environment and the result files they belong to.

    with ExperimentReport("train_mse_all", config=config_from_globals(globals())) as rep:
        rep.add_data("states", path)
        rep.add_model("cnn", model)
        rep.log_metric("mse", mse, step=ratio)
        rep.add_output(csv_path)

Writes $LOKA_ROOT/reports/<name>_<jobid>/report.{json,md} and appends every output to
$LOKA_ROOT/reports/index.jsonl, so a result file can be traced back to the report that made it.
"""
import hashlib
import inspect
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

INDEX_NAME = "index.jsonl"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".gif"}
_HASH_CACHE: dict = {}  # (path, size, mtime) -> (sha256, n_lines)


def _jsonable(x: Any) -> Any:
    """Turn config values (paths, functions, lambdas, tuples, ...) into stable JSON."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if callable(x):
        name = getattr(x, "__name__", type(x).__name__)
        if name != "<lambda>":
            return name
        try:
            return inspect.getsource(x).strip()
        except (OSError, TypeError):
            return name
    return repr(x)


def config_from_globals(g: dict) -> dict:
    """The UPPERCASE constants of a script (its CONFIG block), as JSON-friendly values."""
    return {k: _jsonable(v) for k, v in g.items()
            if k.isupper() and not k.startswith("_")
            and not (inspect.ismodule(v) or inspect.isclass(v) or callable(v))}  # skips aliases like `F`


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def file_info(path, hash_file: bool = True) -> dict:
    """Path, size, mtime and (optionally) sha256 + line count of a file."""
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    st = p.stat()
    info = {"path": str(p.resolve()), "exists": True, "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
    if hash_file:
        key = (info["path"], st.st_size, st.st_mtime_ns)
        if key not in _HASH_CACHE:
            h, lines = hashlib.sha256(), 0
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
                    lines += chunk.count(b"\n")
            _HASH_CACHE[key] = (h.hexdigest(), lines)
        info["sha256"], info["n_lines"] = _HASH_CACHE[key]
    return info


def _run(cmd: list, cwd=None) -> Optional[str]:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_info(directory) -> Optional[dict]:
    head = _run(["git", "rev-parse", "HEAD"], directory)
    if head is None:
        return None
    diff = _run(["git", "diff", "HEAD"], directory) or ""
    return {"dir": str(directory), "commit": head,
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], directory),
            "dirty": bool(_run(["git", "status", "--porcelain"], directory)),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else None}


def _torch_info() -> Optional[dict]:
    try:
        import torch
    except ImportError:
        return None
    info = {"version": torch.__version__, "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "initial_seed": torch.initial_seed()}
    if info["cuda_available"]:
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def _slurm_info() -> Optional[dict]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        return None
    info = {"env": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}}
    log_dir = os.environ.get("LOG_DIR")
    if log_dir and os.environ.get("SLURM_JOB_NAME"):  # matches --output=%x_%j.out in the sbatch files
        info["log_file"] = str(Path(log_dir) / f"{os.environ['SLURM_JOB_NAME']}_{job_id}.out")
    out = _run(["scontrol", "show", "job", job_id, "-o"])  # time limit and dependency aren't in the env
    if out:
        fields = dict(f.split("=", 1) for f in out.split() if "=" in f)
        info["scontrol"] = {k: fields[k] for k in ("TimeLimit", "Dependency", "StdOut", "Command", "Partition", "TRES")
                            if k in fields}
    return info


def _packages() -> dict:
    return {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}


class ExperimentReport:
    """Context manager that writes a report even when the job fails or is killed by Slurm."""

    def __init__(self, name: str, config: Optional[dict] = None, root=None, script=None):
        self.name = name
        self.config = _jsonable(config or {})
        self.config_hash = config_hash(self.config)
        self.root = Path(root or os.environ.get("LOKA_ROOT", ".")) / "reports"
        self.job_id = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("local-%Y%m%d-%H%M%S")
        self.dir = self.root / f"{name}_{self.job_id}"
        if script is None and sys.argv and sys.argv[0].endswith(".py"):
            script = sys.argv[0]
        self.script = Path(script) if script and Path(script).is_file() else None
        self.data: dict = {}
        self.models: dict = {}
        self.metrics: list = []
        self.outputs: list = []
        self.skipped: list = []
        self.status = "running"
        self.error: Optional[str] = None
        self._start = time.time()
        self._end: Optional[float] = None
        self._old_sigterm = None

    # ---- recording ----
    def add_data(self, name: str, path) -> dict:
        self.data[name] = file_info(path)
        self._write()
        return self.data[name]

    def add_model(self, name: str, model, **init_kwargs) -> dict:
        params = list(model.parameters()) if hasattr(model, "parameters") else []
        self.models[name] = {"class": f"{type(model).__module__}.{type(model).__qualname__}",
                             "repr": repr(model), "init_kwargs": _jsonable(init_kwargs),
                             "n_params": sum(p.numel() for p in params),
                             "n_trainable": sum(p.numel() for p in params if p.requires_grad)}
        self._write()
        return self.models[name]

    def log_metric(self, key: str, value: float, step=None) -> None:
        self.metrics.append({"key": key, "value": float(value), "step": _jsonable(step)})

    def add_output(self, path, **extra) -> dict:
        """Record a result file and register it in the index under this report."""
        info = {**file_info(path), **_jsonable(extra)}
        if info["exists"] and Path(path).suffix == ".csv":
            info["rows"] = info["n_lines"] - 1  # minus header
        if info["exists"] and Path(path).suffix.lower() in IMAGE_SUFFIXES:
            self.dir.mkdir(parents=True, exist_ok=True)  # plots are copied in so report.md renders anywhere
            shutil.copy2(path, self.dir / Path(path).name)
            info["copy"] = Path(path).name
        self.outputs.append(info)
        if info["exists"]:
            self.root.mkdir(parents=True, exist_ok=True)
            record = {"path": info["path"], "sha256": info["sha256"], "report": str(self.dir),
                      "name": self.name, "job_id": self.job_id, "config_hash": self.config_hash,
                      "time": datetime.now().isoformat(timespec="seconds")}
            with open(self.root / INDEX_NAME, "a") as f:
                f.write(json.dumps(record) + "\n")
        self._write()
        return info

    def provenance(self, path) -> Optional[dict]:
        """The latest index record for a result file, or None if no report made it."""
        index, target, found = self.root / INDEX_NAME, str(Path(path).resolve()), None
        if index.is_file():
            for line in index.read_text().splitlines():
                rec = json.loads(line)
                if rec["path"] == target:
                    found = rec
        return found

    def is_stale(self, path) -> bool:
        """True if the file was made by a report with different settings."""
        rec = self.provenance(path)
        return rec is not None and rec["config_hash"] != self.config_hash

    def add_skipped(self, path) -> dict:
        """Record an existing output that was not recomputed, and warn if it may not match these settings."""
        rec = self.provenance(path)
        info = file_info(path)
        modified = rec is not None and info.get("sha256") != rec["sha256"]
        stale = rec is not None and rec["config_hash"] != self.config_hash
        if rec is None:
            print(f"Warning: {path} exists but no report records how it was made", flush=True)
        elif stale or modified:
            why = "different settings" if stale else "changed since it was written"
            print(f"Warning: {path} was made with {why} (report {rec['report']})", flush=True)
        self.skipped.append({**info, "produced_by": rec, "stale": stale, "modified": modified})
        self._write()
        return self.skipped[-1]

    # ---- context manager ----
    def __enter__(self) -> "ExperimentReport":
        self.dir.mkdir(parents=True, exist_ok=True)
        self.environment = {
            "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
            "argv": sys.argv, "cwd": os.getcwd(), "pid": os.getpid(),
            "loka_root": os.environ.get("LOKA_ROOT"),
            "tqdm_env": {k: v for k, v in os.environ.items() if k.startswith("TQDM_")},
            "git": {"loka": _git_info(Path(__file__).parent),
                    **({"script": _git_info(self.script.parent)} if self.script else {})},
            "torch": _torch_info(), "packages": _packages(),
        }
        self.slurm = _slurm_info()
        self.script_info = None
        if self.script:
            self.script_info = file_info(self.script)
            shutil.copy2(self.script, self.dir / self.script.name)
        if threading.current_thread() is threading.main_thread():  # Slurm sends SIGTERM at the time limit
            self._old_sigterm = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        self._write()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._end = time.time()
        if exc_type is None:
            self.status = "ok"
        else:
            self.status = "interrupted" if issubclass(exc_type, (KeyboardInterrupt, SystemExit)) else "failed"
            self.error = "".join(traceback.format_exception(exc_type, exc, tb))
        if self._old_sigterm is not None:
            signal.signal(signal.SIGTERM, self._old_sigterm)
        self._write()
        print(f"report ({self.status}): {self.dir}", flush=True)
        return False

    # ---- writing ----
    def to_dict(self) -> dict:
        end = self._end or time.time()
        return {"name": self.name, "job_id": self.job_id, "status": self.status, "error": self.error,
                "start": datetime.fromtimestamp(self._start).isoformat(timespec="seconds"),
                "end": datetime.fromtimestamp(self._end).isoformat(timespec="seconds") if self._end else None,
                "duration_s": round(end - self._start, 1),
                "config_hash": self.config_hash, "config": self.config,
                "data": self.data, "models": self.models, "metrics": self.metrics,
                "outputs": self.outputs, "skipped": self.skipped,
                "script": getattr(self, "script_info", None),
                "environment": getattr(self, "environment", None), "slurm": getattr(self, "slurm", None)}

    def _write(self) -> None:
        if not self.dir.is_dir():
            return
        report = self.to_dict()
        (self.dir / "report.json").write_text(json.dumps(report, indent=2))
        (self.dir / "report.md").write_text(_markdown(report))


def _table(rows: list, header: list) -> list:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def _markdown(r: dict) -> str:
    env, slurm = r["environment"] or {}, r["slurm"] or {}
    lines = [f"# {r['name']} ({r['status']})", "",
             f"- job: `{r['job_id']}`   host: `{env.get('host')}`",
             f"- {r['start']} → {r['end']}  ({r['duration_s']} s)",
             f"- config hash: `{r['config_hash']}`"]
    if slurm.get("log_file"):
        lines.append(f"- slurm log: `{slurm['log_file']}`")
    if r["error"]:
        lines += ["", "## Error", "```", r["error"].rstrip(), "```"]

    lines += ["", "## Settings", ""] + _table([(k, f"`{v}`") for k, v in r["config"].items()], ["setting", "value"])

    if r["data"]:
        lines += ["## Data", ""] + _table(
            [(k, f"`{d['path']}`", d.get("n_lines"), d.get("size"), f"`{str(d.get('sha256'))[:12]}`")
             for k, d in r["data"].items()], ["name", "path", "lines", "bytes", "sha256"])
    if r["models"]:
        lines += ["## Models", ""] + _table(
            [(k, f"`{m['class']}`", m["n_params"], f"`{m['init_kwargs']}`") for k, m in r["models"].items()],
            ["name", "class", "params", "init args"])
        lines += [f"<details><summary>{k}</summary>\n\n```\n{m['repr']}\n```\n</details>\n" for k, m in r["models"].items()]
    if r["metrics"]:
        lines += ["## Metrics", ""] + _table([(m["key"], m["step"], f"{m['value']:.6g}") for m in r["metrics"]],
                                              ["metric", "step", "value"])
    if r["outputs"]:
        lines += ["## Results (this report produced these files)", ""] + _table(
            [(f"`{o['path']}`", o.get("rows", o.get("size")), f"`{str(o.get('sha256'))[:12]}`") for o in r["outputs"]],
            ["file", "rows/bytes", "sha256"])
    for o in r["outputs"]:  # plots are shown inline from the copy in the report folder
        if o.get("copy"):
            lines += [f"### {o['copy']}", "", f"![{o['copy']}]({o['copy']})", ""]
    if r["skipped"]:
        lines += ["## Skipped (already existed, not produced by this report)", ""] + _table(
            [(f"`{s['path']}`", (s["produced_by"] or {}).get("report", "unknown"),
              "stale" if s["stale"] else "changed" if s["modified"] else "ok") for s in r["skipped"]],
            ["file", "made by report", "state"])

    git = env.get("git") or {}
    lines += ["## Environment", ""]
    for k, g in git.items():
        if g:
            lines.append(f"- git ({k}): `{g['commit'][:10]}` on `{g['branch']}`" +
                         (f", **dirty** (diff `{g['diff_sha256']}`)" if g["dirty"] else ""))
    t = env.get("torch") or {}
    if t:
        lines.append(f"- torch {t['version']}, CUDA {t['cuda']}, GPUs: {t.get('gpus', 'none')}")
    pk = env.get("packages") or {}
    lines.append(f"- python {env.get('python')}; " + ", ".join(f"{p} {pk[p]}" for p in
                 ("numpy", "pandas", "scikit-learn", "loka") if p in pk))
    if r["script"]:
        lines.append(f"- script: `{r['script']['path']}` (sha256 `{r['script']['sha256'][:12]}`)")
    if slurm:
        keys = ("SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE", "SLURM_SUBMIT_DIR")
        lines += ["", "## Slurm", ""] + [f"- {k.removeprefix('SLURM_')}: `{slurm['env'][k]}`" for k in keys
                                         if k in slurm["env"]]
        lines += [f"- {k}: `{v}`" for k, v in (slurm.get("scontrol") or {}).items()]
    return "\n".join(lines) + "\n"
