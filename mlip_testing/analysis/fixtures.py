from __future__ import annotations

import functools

import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(plot_type="matplotlib"):
    def plot_scatter_decorator(func):
        @functools.wraps(func)
        def plot_scatter_wrapper(*args, **kwargs):
            results = func(*args, **kwargs)

            ref = results["ref"]
            max_data = np.max(ref)
            min_data = np.min(ref)
            for mlip, value in results.items():
                if mlip == "ref":
                    continue
                max_data = max(max_data, np.max(value))
                min_data = min(min_data, np.min(value))
                plt.scatter(value, ref, label=mlip)
            xlims = plt.xlim()
            ylims = plt.ylim()

            min_data = min(xlims + ylims)
            max_data = max(xlims + ylims)

            x = np.linspace(min_data, max_data, num=100)
            y = x
            plt.plot(x, y, label=None, linestyle="--")

            plt.gca().set_aspect("equal", adjustable="box")
            plt.savefig("fixture.svg")
            plt.show()

            return results

        return plot_scatter_wrapper

    return plot_scatter_decorator
