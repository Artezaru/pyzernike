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
# Time computation evolution of the function core_polynomial
# ------------------------------------------------------------------------------------------
# The time can be change according to the used machine and its current load.
# ==========================================================================================

from pyzernike import xy_zernike_polynomial_up_to_order
import time
import numpy
import matplotlib.pyplot as plt
import os
import csv


def time_xy_zernike_polynomial_up_to_order(order, num_points, dx=None, dy=None, precompute=True, Nmean=10):
    # Create a grid of rho and theta values
    x = numpy.random.uniform(-1, 1, int(numpy.round(numpy.sqrt(num_points))))
    y = numpy.random.uniform(-1, 1, int(numpy.round(numpy.sqrt(num_points))))
    X, Y = numpy.meshgrid(x, y)

    # Record the start time
    start_time = time.perf_counter()

    for _ in range(Nmean):
        # Call the core_create_precomputing_sets function with the generated values
        _ = xy_zernike_polynomial_up_to_order(
            x=X.flatten(),
            y=Y.flatten(),
            order=order,
            x_derivative=dx,
            y_derivative=dy,
            precompute=precompute,
        )

    # Record the end time
    end_time = time.perf_counter()

    # Calculate the elapsed time
    elapsed_time = (end_time - start_time) / Nmean

    return elapsed_time



if __name__ == "__main__":

    Nmean = 5  # Number of repetitions for averaging the time
    orders = [3, 5, 7, 9]
    dx_unique = [0]
    dy_unique = [0]
    dx_first_derivative = [0, 1, 0]
    dy_first_derivative = [0, 0, 1]

    # Create a grid of rho and theta values
    num_points = []
    for power in [3,4,5,6]:
        num_points.extend([10**power, 2*(10**power), 5*(10**power)])
    # num_points.append(3000*4000)

    times_unique = []
    times_first = []
    # times_second = []
    for n_points in num_points:
        print(f"Number of points: {n_points} - starting computations...")
        time_unique = [
            time_xy_zernike_polynomial_up_to_order(order, n_points, dx=dx_unique, dy=dy_unique, Nmean=Nmean)
            for order in orders
        ]
        print("\t unique done")
        time_first = [
            time_xy_zernike_polynomial_up_to_order(order, n_points, dx=dx_first_derivative, dy=dy_first_derivative, Nmean=Nmean)
            for order in orders
        ]
        print("\t first done")

        times_unique.append(time_unique)
        times_first.append(time_first)


    # Plot the results (time versus number of points - log-log scale) 
    #
    # - 1 order = one color
    # - 3 curves per color (unique, first derivative, second derivative) with different line styles
    plt.figure(figsize=(10, 6))
    plt.title
    colors=['blue','red','green','orange','black']
    markers = ['o', 's', '^']
    linestyles = ['-', '--', ':']
    labels = ['Unique', 'First Derivative', 'Second Derivative']
    for i, order in enumerate(orders):
        for j, times in enumerate([times_unique, times_first]):#, times_second]):
            plt.plot(num_points, [t[i] for t in times], label=f"Order {order} - {labels[j]}", color=colors[i], marker=markers[j], linestyle=linestyles[j])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Points (log scale)")
    plt.ylabel("Time (seconds, log scale)")
    plt.title("Timing of 'xy_zernike_polynomial_up_to_order' function for all Zernike polynomials up to a given order")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the results in an image file
    file_fig = os.path.join(os.path.dirname(__file__), "results", "timing_xy_zernike_polynomial_up_to_order.png")
    os.makedirs(os.path.dirname(file_fig), exist_ok=True)
    os.remove(file_fig) if os.path.exists(file_fig) else None
    plt.savefig(file_fig, dpi=300)

    # Save the results in an text csv file
    file_txt = os.path.join(os.path.dirname(__file__), "results", "timing_xy_zernike_polynomial_up_to_order.csv")
    os.makedirs(os.path.dirname(file_txt), exist_ok=True)
    os.remove(file_txt) if os.path.exists(file_txt) else None
    with open(file_txt, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Number of Points'] + [f'Order {order} - {label}' for order in orders for label in labels[:2]]
        writer.writerow(header)
        for i, n_points in enumerate(num_points):
            row = [n_points] + [times_unique[i][j] for j in range(len(orders))] + [times_first[i][j] for j in range(len(orders))]
            writer.writerow(row)

    # Show the plotb
    plt.show()