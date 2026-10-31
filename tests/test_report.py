import json
import os as _os
from pathlib import Path

import pytest

from loka.report import ExperimentReport, config_from_globals, config_hash

EPOCHS = 3
LR = 0.1
ROOT = Path("/x")
METHODS = [len, lambda v: v + 1]
OS = _os  # an uppercase module alias is not a setting
HELPER = len


@pytest.fixture(autouse=True)
def no_slurm(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("SLURM_"):
            monkeypatch.delenv(k)


def make(tmp_path, **config):
    return ExperimentReport("job", config=config, root=tmp_path, script=None)


def load(rep):
    return json.loads((rep.dir / "report.json").read_text())


def test_config_from_globals_is_stable_json():
    cfg = config_from_globals(globals())
    assert cfg["EPOCHS"] == 3 and cfg["ROOT"] == "/x"
    assert cfg["METHODS"][0] == "len" and "lambda" in cfg["METHODS"][1]
    assert "pytest" not in cfg and "OS" not in cfg and "HELPER" not in cfg
    assert config_hash(cfg) == config_hash(config_from_globals(globals()))


def test_report_written_on_success(tmp_path):
    data = tmp_path / "states.txt"
    data.write_text("a\nb\nc\n")
    out = tmp_path / "res.csv"
    out.write_text("ratio,mse\n0.1,1.0\n0.2,0.5\n")
    with make(tmp_path, EPOCHS=3) as rep:
        rep.add_data("states", data)
        rep.log_metric("mse", 0.5, step=0.2)
        rep.add_output(out)
    r = load(rep)
    assert r["status"] == "ok" and r["config"] == {"EPOCHS": 3}
    assert r["data"]["states"]["n_lines"] == 3
    assert r["outputs"][0]["rows"] == 2
    assert r["metrics"] == [{"key": "mse", "value": 0.5, "step": 0.2}]
    assert (rep.dir / "report.md").read_text().startswith("# job (ok)")


def test_report_written_on_failure(tmp_path):
    with pytest.raises(ValueError):
        with make(tmp_path) as rep:
            raise ValueError("boom")
    r = load(rep)
    assert r["status"] == "failed" and "boom" in r["error"]


def test_exit_is_recorded_as_interrupted(tmp_path):
    with pytest.raises(SystemExit):
        with make(tmp_path) as rep:
            raise SystemExit(143)
    assert load(rep)["status"] == "interrupted"


def test_model_info(tmp_path):
    torch = pytest.importorskip("torch")
    with make(tmp_path) as rep:
        rep.add_model("lin", torch.nn.Linear(3, 2), n_extra=1)
    m = load(rep)["models"]["lin"]
    assert m["n_params"] == 8 and m["init_kwargs"] == {"n_extra": 1}


def test_index_and_staleness(tmp_path):
    out = tmp_path / "res.csv"
    out.write_text("ratio,mse\n0.1,1.0\n")
    with make(tmp_path, EPOCHS=3) as first:
        first.add_output(out)
    with make(tmp_path, EPOCHS=3) as same:
        assert same.provenance(out)["report"] == str(first.dir)
        assert not same.is_stale(out)
    with make(tmp_path, EPOCHS=50) as other:
        assert other.is_stale(out)
        skipped = other.add_skipped(out)
    assert skipped["stale"] and skipped["produced_by"]["job_id"] == first.job_id


def test_skipped_without_provenance_warns(tmp_path, capsys):
    out = tmp_path / "old.csv"
    out.write_text("ratio,mse\n")
    with make(tmp_path) as rep:
        rep.add_skipped(out)
    assert "no report records" in capsys.readouterr().out
    assert load(rep)["skipped"][0]["produced_by"] is None


def test_slurm_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_NAME", "train")
    monkeypatch.setenv("LOG_DIR", "/logs")
    with make(tmp_path) as rep:
        pass
    assert rep.dir.name == "job_123"
    assert load(rep)["slurm"]["log_file"] == "/logs/train_123.out"


def test_plot_outputs_are_embedded_in_markdown(tmp_path):
    png = tmp_path / "plots" / "curve.png"
    png.parent.mkdir()
    png.write_bytes(b"\x89PNG")
    with make(tmp_path) as rep:
        rep.add_output(png)
        rep.add_output(tmp_path / "plots" / "missing.png")
    md = (rep.dir / "report.md").read_text()
    assert "![curve.png](curve.png)" in md and "![missing.png]" not in md
    assert (rep.dir / "curve.png").read_bytes() == png.read_bytes()
