"""Run analysis."""

from __future__ import annotations

from mlip_testing import analysis
from mlip_testing.analysis.utils import get_config


def main():
    """Run analysis."""
    config = get_config(
        "/Users/elliottkasoar/Documents/PSDI/mlip-testing/mlip_testing/analysis/config.yml"
    )
    test = "equation_of_state"

    module = getattr(analysis, config[test]["module"])
    mlips = config[test]["mlips"]

    metric_results = {}
    for metric in config[test]["score"]["metrics"]:
        inputs = config[test]["score"]["metrics"][metric]["inputs"]
        get_inputs = getattr(
            module, config[test]["score"]["metrics"][metric][inputs]["calc"]
        )

        inputs = get_inputs(mlips)

        get_metric = getattr(module, config[test]["score"]["metrics"][metric]["calc"])
        metric_results[metric] = get_metric(mlips, inputs)

    get_score = getattr(module, config[test]["score"]["calc"])
    score = get_score(mlips, metric_results)

    print(score)
    print(metric_results)


main()
