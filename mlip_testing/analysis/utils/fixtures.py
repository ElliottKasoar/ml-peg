"""Fixtures for MLIP results analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
from typing import Any

import numpy as np
import plotly.graph_objects as go


def plot_scatter(plot_combined: bool = True) -> Callable:
    """
    Plot scatter plot of MLIP results against reference data.

    Parameters
    ----------
    plot_combined
        Whether to plot all MLIPs on same graph.

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
            ref = results["ref"]

            traces = []

            for mlip, value in results.items():
                if mlip == "ref":
                    continue
                traces.append(
                    go.Scatter(
                        x=value,
                        y=ref,
                        name=mlip,
                        mode="markers",
                    )
                )

            if not plot_combined:
                for trace in traces:
                    fig = go.Figure()
                    fig.add_trace(trace)

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
                    fig.write_json(f"scatter_{trace.name}.json")
            else:
                fig = go.Figure()

                for trace in traces:
                    fig.add_trace(trace)

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
                fig.write_json("scatter_combined.json")

            return results

        return plot_scatter_wrapper

    return plot_scatter_decorator


def plot_bar(labels: Sequence | None = None) -> Callable:
    """
    Plot bar chart of MLIP results against reference data.

    Parameters
    ----------
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

            return results

        return plot_bar_wrapper

    return plot_bar_decorator


def build_table(labels: Sequence | None = None) -> Callable:
    """
    Build table MLIP results.

    Parameters
    ----------
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

            return results

        return plot_bar_wrapper

    return plot_bar_decorator
