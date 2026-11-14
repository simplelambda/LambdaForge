"""Small scientific namespaces do not accidentally load training/optional backends."""

import os
import subprocess
import sys


def test_topology_stats_and_validation_do_not_load_torch_sklearn_or_pot():
    code = """
import sys
from lambdaforge.topology import SuperlevelMergeTree, StableRegionSelector
from lambdaforge.stats import EmpiricalNullCalibrator, bootstrap_quantile
from lambdaforge.distances import validate_dissimilarity, FGWBarycenter, FGWMatrixBuilder
assert not any(name in sys.modules for name in ('torch', 'sklearn', 'ot'))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
