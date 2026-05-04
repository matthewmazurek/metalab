from __future__ import annotations

import argparse
import builtins
import json

import pytest

from metalab import cli
from metalab.cli_config import load_run_config
from metalab.executor.thread import ThreadExecutor


def test_load_experiment_static_target_without_config(tmp_path):
    target = tmp_path / "experiment_file.py"
    target.write_text(
        """
class Exp:
    experiment_id = "static:1"

experiment = Exp()
""",
        encoding="utf-8",
    )

    exp = cli._load_experiment(str(target))

    assert exp.experiment_id == "static:1"


def test_config_aware_module_target_gets_app_config_only(tmp_path, monkeypatch):
    module_path = tmp_path / "my_experiment.py"
    module_path.write_text(
        """
seen_config = None

class Exp:
    experiment_id = "configured:1"

def build(config):
    global seen_config
    seen_config = config
    return Exp()
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "smoke",
                "data": {"input": "data.h5ad"},
                "metalab": {"store": "runs/smoke"},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_run_config(str(config_path))

    exp = cli._load_experiment("my_experiment:build", loaded.app_config)

    import my_experiment

    assert exp.experiment_id == "configured:1"
    assert my_experiment.seen_config == {
        "experiment_name": "smoke",
        "data": {"input": "data.h5ad"},
    }


def test_toml_config_and_top_level_interpolation(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
experiment_name = "smoke"

[metalab]
store = "experiments/{experiment_name}"
executor = "local"
workers = 4
resume = false
""",
        encoding="utf-8",
    )

    loaded = load_run_config(str(config_path))

    assert loaded.app_config == {"experiment_name": "smoke"}
    assert loaded.run_config.store == "experiments/smoke"
    assert loaded.run_config.executor == "local"
    assert loaded.run_config.workers == 4
    assert loaded.run_config.resume is False


def test_yaml_config_loads_when_pyyaml_is_installed(tmp_path):
    pytest.importorskip("yaml")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment_name: smoke
metalab:
  store: experiments/{experiment_name}
""",
        encoding="utf-8",
    )

    loaded = load_run_config(str(config_path))

    assert loaded.run_config.store == "experiments/smoke"


def test_yaml_config_without_pyyaml_raises_install_hint(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("metalab:\n  store: runs\n", encoding="utf-8")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match=r"metalab\[config\]"):
        load_run_config(str(config_path))


def test_unknown_metalab_key_fails(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"metalab": {"store": "runs", "surprise": True}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown `metalab` config key"):
        load_run_config(str(config_path))


def test_missing_interpolation_key_fails(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"metalab": {"store": "runs/{missing}"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Missing interpolation value"):
        load_run_config(str(config_path))


def test_non_scalar_interpolation_value_fails(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"experiment_name": {"nested": "bad"}, "metalab": {"store": "runs/{experiment_name}"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"top-level scalar"):
        load_run_config(str(config_path))


def test_handle_run_uses_config_store_and_cli_overrides(tmp_path, monkeypatch):
    module_path = tmp_path / "expmod.py"
    module_path.write_text(
        """
class Exp:
    experiment_id = "cli:1"

def build(config):
    return Exp()
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "from-config",
                "metalab": {
                    "store": "runs/{experiment_name}",
                    "workers": 2,
                    "resume": True,
                },
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class FakeHandle:
        can_reconnect = False
        job_id = None

        def result(self):
            return [object(), object()]

    def fake_run(exp, *, store, executor, resume):
        captured.update(exp=exp, store=store, executor=executor, resume=resume)
        return FakeHandle()

    monkeypatch.setattr("metalab.run", fake_run)
    args = argparse.Namespace(
        target="expmod:build",
        config=str(config_path),
        store=str(tmp_path / "override-runs"),
        executor=None,
        workers=4,
        resume=False,
    )

    assert cli._handle_run(args) == 0

    assert captured["store"] == str(tmp_path / "override-runs")
    assert captured["resume"] is False
    assert isinstance(captured["executor"], ThreadExecutor)
    assert captured["executor"]._max_workers == 4


def test_handle_run_passes_slurm_executor_config(tmp_path, monkeypatch):
    module_path = tmp_path / "expmod_slurm.py"
    module_path.write_text(
        """
class Exp:
    experiment_id = "cli-slurm:1"

experiment = Exp()
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "metalab": {
                    "store": "runs",
                    "executor": "slurm",
                    "executor_config": {"partition": "gpu", "time": "12:00:00"},
                }
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class FakeExecutor:
        pass

    class FakeHandle:
        can_reconnect = True
        job_id = "12345"

    def fake_executor_from_config(executor_type, config):
        captured["executor_type"] = executor_type
        captured["executor_config"] = config
        return FakeExecutor()

    def fake_run(exp, *, store, executor, resume):
        captured.update(store=store, executor=executor, resume=resume)
        return FakeHandle()

    monkeypatch.setattr("metalab.executor.config.executor_from_config", fake_executor_from_config)
    monkeypatch.setattr("metalab.run", fake_run)
    args = argparse.Namespace(
        target="expmod_slurm",
        config=str(config_path),
        store=None,
        executor=None,
        workers=None,
        resume=None,
    )

    assert cli._handle_run(args) == 0

    assert captured["executor_type"] == "slurm"
    assert captured["executor_config"] == {"partition": "gpu", "time": "12:00:00"}
    assert captured["store"] == "runs"
    assert captured["resume"] is True
    assert isinstance(captured["executor"], FakeExecutor)
