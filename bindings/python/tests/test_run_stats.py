# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

import uairt


def test_run_stats_measure_the_backend_run():
    x = np.array([[1, 2, 3, 4]], np.float32)
    with uairt.Engine("reference", delay_ms=5) as engine, engine.load_model("unused") as model:
        stats = model.run_stats()
        assert (stats.runs, stats.total_ns, stats.min_ns, stats.max_ns) == (0, 0, 0, 0) and stats.mean_ns == 0.0
        for _ in range(3):
            model.run(x)
        with pytest.raises(uairt.InvalidArgument):
            model.run(x[:, :2])  # rejected before the backend: not counted
        stats = model.run_stats()
        assert stats.runs == 3
        assert 5e6 <= stats.min_ns <= stats.last_ns <= stats.max_ns < 2e9
        assert stats.total_ns >= 3 * stats.min_ns and 5.0 <= stats.mean_ms < 2000.0
        model.reset_run_stats()
        assert model.run_stats().runs == 0
    with uairt.Engine("reference") as engine, engine.load_model("unused") as model:
        model.close()
        with pytest.raises(uairt.InvalidArgument):
            model.run_stats()
