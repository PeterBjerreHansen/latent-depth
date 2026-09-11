import json

import pytest

from sweep_data_size import _validate_resume_state


def test_resume_requires_sweep_snapshot_for_existing_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    snapshot_path = tmp_path / "sweep_config.json"
    metrics_path.write_text(json.dumps({"train_size": 32}) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no sweep_config.json"):
        _validate_resume_state(
            metrics_path=metrics_path,
            snapshot_path=snapshot_path,
            snapshot={"train_sizes": [32]},
            resume=True,
        )
