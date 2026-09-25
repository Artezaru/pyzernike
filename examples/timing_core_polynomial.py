# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# ==========================================================================================
# Time computation of all the Zernike polynomials up to a given order with:
#   - _core_polynomials (explicit sum) with precompute=False
#   - _core_polynomials (explicit sum) with precompute=True
#   - _zernike_recursive_up_to_order (recurrence relation)
# ------------------------------------------------------------------------------------------
# The time can change according to the used machine and its current load.
# ==========================================================================================

import csv
import os
import time
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy

try:
    import pyzernike
except:
    import os 
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import pyzernike

from pyzernike.core.core_polynomial import _core_polynomials
from pyzernike.core.zernike_recursive import _zernike_recursive_up_to_order


N_MEAN = 5
ORDERS = [3, 5, 7, 9]
NUM_POINTS = [factor * 10**power for power in [3, 4, 5, 6] for factor in [1, 2, 5]] + [3000 * 4000]
DERIVATIVES = {
    "value": ([0], [0]),
    "value + gradient": ([0, 1, 0], [0, 0, 1]),
}
MAX_OUTPUT_BYTES = 4 * 1024**3  # Configurations whose output exceeds this size are skipped (time = NaN)
RESULTS_DIRECTORY = os.path.join(os.path.dirname(__file__), "results")


def up_to_order(order: int, rho_derivatives: List[int], theta_derivatives: List[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Build the lists (n, m, rho_derivative, theta_derivative) of all the polynomials up to `order` in the OSA/ANSI ordering,
    for each derivative pair, with the layout (D, P) of `_zernike_recursive_up_to_order`."""
    n = [n_i for n_i in range(order + 1) for _ in range(n_i + 1)]
    m = [m_i for n_i in range(order + 1) for m_i in range(-n_i, n_i + 1, 2)]
    n_polynomials = len(n)
    return (
        n * len(rho_derivatives),
        m * len(rho_derivatives),
        [dr for dr in rho_derivatives for _ in range(n_polynomials)],
        [dt for dt in theta_derivatives for _ in range(n_polynomials)],
    )


def explicit(precompute: bool) -> Callable:
    def compute(rho, theta, order, rho_derivatives, theta_derivatives):
        n, m, dr, dt = up_to_order(order, rho_derivatives, theta_derivatives)
        return _core_polynomials(rho, theta, n, m, dr, dt, False, precompute)
    return compute


def recursive(rho, theta, order, rho_derivatives, theta_derivatives):
    return _zernike_recursive_up_to_order(rho, theta, order, rho_derivatives, theta_derivatives)


METHODS: Dict[str, Callable] = {
    "explicit, precompute=False": explicit(False),
    "explicit, precompute=True": explicit(True),
    "recursive": recursive,
}


def time_method(method: Callable, order: int, num_points: int, rho_derivatives: List[int], theta_derivatives: List[int], n_mean: int = N_MEAN) -> float:
    """Mean time (in seconds) of one call of `method` for all the polynomials up to `order` on `num_points` random points."""
    n_polynomials = (order + 1) * (order + 2) // 2
    if len(rho_derivatives) * n_polynomials * num_points * 8 > MAX_OUTPUT_BYTES:
        return numpy.nan

    rng = numpy.random.default_rng(0)
    rho = rng.uniform(0, 1, num_points)
    theta = rng.uniform(0, 2 * numpy.pi, num_points)

    start_time = time.perf_counter()
    for _ in range(n_mean):
        method(rho, theta, order, rho_derivatives, theta_derivatives)
    return (time.perf_counter() - start_time) / n_mean


def run() -> Dict[Tuple[str, str, int], List[float]]:
    """Time all the methods, derivative sets and orders for all the numbers of points."""
    times = {}
    for method_name, method in METHODS.items():
        for derivatives_name, (rho_derivatives, theta_derivatives) in DERIVATIVES.items():
            for order in ORDERS:
                times[(method_name, derivatives_name, order)] = [
                    time_method(method, order, num_points, rho_derivatives, theta_derivatives)
                    for num_points in NUM_POINTS
                ]
                print(f"{method_name} - {derivatives_name} - order {order} done")
    return times


def plot(times: Dict[Tuple[str, str, int], List[float]], file: str) -> None:
    """One subplot per order: one color per method, one line style per derivative set (log-log scale)."""
    colors = {"explicit, precompute=False": "tab:blue", "explicit, precompute=True": "tab:orange", "recursive": "tab:green"}
    linestyles = {"value": "-", "value + gradient": "--"}

    columns = 2
    rows = (len(ORDERS) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 4.5 * rows), sharex=True, sharey=True, squeeze=False)

    for ax, order in zip(axes.flat, ORDERS):
        for method_name in METHODS:
            for derivatives_name in DERIVATIVES:
                ax.plot(
                    NUM_POINTS,
                    times[(method_name, derivatives_name, order)],
                    color=colors[method_name],
                    linestyle=linestyles[derivatives_name],
                    marker="o",
                    markersize=3,
                    label=f"{method_name} - {derivatives_name}",
                )
        ax.set_title(f"Order {order} ({(order + 1) * (order + 2) // 2} polynomials)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes.flat[len(ORDERS):]:
        ax.set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Number of points")
    for ax in axes[:, 0]:
        ax.set_ylabel("Time (s)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Computation time of all the Zernike polynomials up to a given order")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(file, dpi=300)


def save_csv(times: Dict[Tuple[str, str, int], List[float]], file: str) -> None:
    """One row per configuration: method, derivatives, order, number of points, time."""
    with open(file, mode="w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Method", "Derivatives", "Order", "Number of points", "Time (s)"])
        for (method_name, derivatives_name, order), values in times.items():
            for num_points, value in zip(NUM_POINTS, values):
                writer.writerow([method_name, derivatives_name, order, num_points, value])


if __name__ == "__main__":
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    times = run()
    save_csv(times, os.path.join(RESULTS_DIRECTORY, "timing_core_polynomial.csv"))
    plot(times, os.path.join(RESULTS_DIRECTORY, "timing_core_polynomial.png"))

    plt.show()