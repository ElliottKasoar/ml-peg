"""Fixtures for MLIP results analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def plot_scatter(plot_type: Literal["matplotlib", "plotly"] = "matplotlib") -> Callable:
    """
    Plot scatter plot of MLIP results against reference data.

    Parameters
    ----------
    plot_type
        Type of plot to create and save.

    Returns
    -------
    Callable
        Decorator to wrap function.
    """

    def plot_scatter_decorator(func: Callable) -> Callable:
        """
        Decorate function to plot scatter.

        Parameters
        ----------
        func
            Function being wrapped.

        Returns
        -------
        Callable
            Wrapped function.
        """

        @functools.wraps(func)
        def plot_scatter_wrapper(*args, **kwargs) -> dict[str, Any]:
            """
            Wrap function to plot scatter.

            Parameters
            ----------
            *args
                Arguments to pass to the function being wrapped.
            **kwargs
                Key word arguments to pass to the function being wrapped.

            Returns
            -------
            dict
                Results dictionary.
            """
            results = func(*args, **kwargs)

            match plot_type:
                case "matplotlib":
                    fig, ax = plt.subplots()
                    ref = results["ref"]
                    for mlip, value in results.items():
                        if mlip == "ref":
                            continue
                        ax.scatter(value, ref, label=mlip)

                    lims = [
                        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                    ]

                    ax.plot(lims, lims, "k-", alpha=0.75)
                    ax.set_aspect("equal")
                    ax.set_xlim(lims)
                    ax.set_ylim(lims)

                    fig.savefig("scatter.svg")
                    # fig.show()

                case "plotly":
                    fig = go.Figure()

                    ref = results["ref"]

                    for mlip, value in results.items():
                        if mlip == "ref":
                            continue
                        fig.add_trace(
                            go.Scatter(
                                x=value,
                                y=ref,
                                name=mlip,
                                mode="markers",
                            )
                        )

                    full_fig = fig.full_figure_for_development()
                    x_range = full_fig.layout.xaxis.range
                    y_range = full_fig.layout.yaxis.range

                    lims = [
                        np.min([x_range, y_range]),  # min of both axes
                        np.max([x_range, y_range]),  # max of both axes
                    ]

                    fig.add_trace(
                        go.Scatter(
                            x=lims,
                            y=lims,
                            mode="lines",
                            showlegend=False,
                        )
                    )

                    fig.update_traces()
                    fig.write_json("scatter.json")

                case _:
                    raise NotImplementedError("Plot type not available.")

            return results

        return plot_scatter_wrapper

    return plot_scatter_decorator


def plot_bar(
    plot_type: Literal["plotly"] = "plotly", labels: Sequence | None = None
) -> Callable:
    """
    Plot bar chart of MLIP results against reference data.

    Parameters
    ----------
    plot_type
        Type of plot to create and save.
    labels
        Labels for bars.

    Returns
    -------
    Callable
        Decorator to wrap function.
    """
    print(labels)

    def plot_bar_decorator(func: Callable) -> Callable:
        """
        Decorate function to plot bar chart.

        Parameters
        ----------
        func
            Function being wrapped.

        Returns
        -------
        Callable
            Wrapped function.
        """

        @functools.wraps(func)
        def plot_bar_wrapper(*args, **kwargs) -> dict[str, Any]:
            """
            Wrap function to plot bar chart.

            Parameters
            ----------
            *args
                Arguments to pass to the function being wrapped.
            **kwargs
                Key word arguments to pass to the function being wrapped.

            Returns
            -------
            dict
                Results dictionary.
            """
            results = func(*args, **kwargs)
            ref = results["ref"]

            match plot_type:
                case "plotly":
                    fig = go.Figure()

                    fig.add_trace(
                        go.Bar(
                            x=labels,
                            y=ref,
                            name="Reference",
                        )
                    )

                    for mlip, value in results.items():
                        if mlip == "ref":
                            continue
                        fig.add_trace(
                            go.Bar(
                                x=labels,
                                y=value,
                                name=mlip,
                            )
                        )

                    fig.update_traces()
                    fig.write_json("bar.json")

                case _:
                    raise NotImplementedError("Plot type not available.")

            return results

        return plot_bar_wrapper

    return plot_bar_decorator
