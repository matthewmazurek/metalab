import json

import pytest

from atlas.export import collect_captured_data_json, runs_to_dataframe
from atlas.models import ProvenanceInfo, RecordFields, RunResponse, RunStatus


def _run(run_id: str, experiment_id: str = "exp:1") -> RunResponse:
    record = RecordFields(
        run_id=run_id,
        experiment_id=experiment_id,
        status=RunStatus.SUCCESS,
        context_fingerprint="ctx",
        params_fingerprint="params",
        seed_fingerprint="seed",
        started_at=__import__("datetime").datetime(2026, 1, 1, 0, 0, 0),
        finished_at=__import__("datetime").datetime(2026, 1, 1, 0, 0, 1),
        duration_ms=1000,
        provenance=ProvenanceInfo(),
    )
    return RunResponse(record=record)


class _DummyStore:
    def __init__(self, by_run_id: dict[str, dict[str, dict]]) -> None:
        self._by_run_id = by_run_id

    def list_results(self, run_id: str) -> list[str]:
        return list(self._by_run_id.get(run_id, {}).keys())

    def get_result(self, run_id: str, name: str):
        return self._by_run_id.get(run_id, {}).get(name)


def test_collect_captured_data_json_builds_per_run_json():
    store = _DummyStore(
        {
            "r1": {
                "matrix": {
                    "data": [[1, 2], [3, 4]],
                    "dtype": "int64",
                    "shape": [2, 2],
                    "metadata": {"kind": "demo"},
                }
            },
            "r2": {},
        }
    )

    out = collect_captured_data_json(store, ["r1", "r2"])
    assert out["r2"] is None

    parsed = json.loads(out["r1"])
    assert "matrix" in parsed
    assert parsed["matrix"]["shape"] == [2, 2]


def test_runs_to_dataframe_includes_captured_data_column_when_enabled():
    pandas = pytest.importorskip("pandas")
    _ = pandas  # silence unused lint in environments where pandas exists

    runs = [_run("r1"), _run("r2")]
    df = runs_to_dataframe(
        runs,
        include_params=False,
        include_metrics=False,
        include_derived=False,
        include_record=True,
        include_data=True,
        captured_data_json={
            "r1": json.dumps({"x": {"data": [1, 2, 3], "dtype": None, "shape": [3], "metadata": {}}}),
            "r2": None,
        },
    )

    assert "captured_data" in df.columns
    assert df.loc[df["run_id"] == "r1", "captured_data"].iloc[0] is not None
    # Pandas will represent missing values as NaN
    assert pandas.isna(df.loc[df["run_id"] == "r2", "captured_data"].iloc[0])


def test_runs_to_dataframe_omits_captured_data_column_when_disabled():
    pandas = pytest.importorskip("pandas")
    _ = pandas

    runs = [_run("r1")]
    df = runs_to_dataframe(
        runs,
        include_params=False,
        include_metrics=False,
        include_derived=False,
        include_record=True,
        include_data=False,
    )

    assert "captured_data" not in df.columns

