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
# Time computation evolution of the function core_create_precomputing_sets
# ------------------------------------------------------------------------------------------
# The time can be change according to the used machine and its current load.
# ==========================================================================================

from pyzernike.core import core_create_precomputing_sets
import time
import matplotlib.pyplot as plt
import os

# Define a set of Zernike polynomials and their derivatives to test the timing
def up_to_order(order, dr=None, dt=None):
    # Set the default derivative orders if not provided
    if dr is None:
        dr = [0]
    if dt is None:
        dt = [0]
    assert len(dr) == len(dt), "dr and dt must have the same length"

    # Initialize lists to hold the n, m, dr, and dt values
    n_values = []
    m_values = []
    dr_values = []
    dt_values = []

    # Loop through each order from 0 to the specified order
    for n in range(order + 1):
        for m in range(-n, n + 1, 2):  # m takes values from -n to n in steps of 2
            n_values.extend([n] * len(dr))  # Repeat n for each combination of dr/dt
            m_values.extend([m] * len(dr))  # Repeat m for each combination of dr/dt
            dr_values.extend(dr)  # Add all dr values
            dt_values.extend(dt)  # Add all dt values

    return n_values, m_values, dr_values, dt_values



def time_core_create_precomputing_sets(order, dr=None, dt=None, flag_radial=False, Nmean=10):
    # Generate the n, m, dr, and dt values for the specified order
    n, m, rho_derivative, theta_derivative = up_to_order(order, dr=dr, dt=dt)

    # Record the start time
    start_time = time.perf_counter()

    for _ in range(Nmean):
        # Call the core_create_precomputing_sets function with the generated values
        _ = core_create_precomputing_sets(
            n=n,
            m=m,
            rho_derivative=rho_derivative,
            theta_derivative=theta_derivative,
            flag_radial=flag_radial,
        )

    # Record the end time
    end_time = time.perf_counter()

    # Calculate the elapsed time
    elapsed_time = (end_time - start_time) / Nmean

    return elapsed_time



if __name__ == "__main__":
    Nmean = 20  # Number of repetitions for averaging the time
    orders = [i for i in range(21)]
    dr_unique = [0]
    dt_unique = [0]
    dr_first_derivative = [0, 1, 0]
    dt_first_derivative = [0, 0, 1]
    dr_second_derivative = [0, 1, 0, 1, 2, 0]
    dt_second_derivative = [0, 0, 1, 1, 0, 2]

    times_radial_unique = [
        time_core_create_precomputing_sets(order, dr=dr_unique, dt=dt_unique, flag_radial=True, Nmean=Nmean)
        for order in orders
    ]
    times_full_unique = [
        time_core_create_precomputing_sets(order, dr=dr_unique, dt=dt_unique, flag_radial=False, Nmean=Nmean)
        for order in orders
    ]
    times_radial_first = [
        time_core_create_precomputing_sets(order, dr=dr_first_derivative, dt=dt_first_derivative, flag_radial=True, Nmean=Nmean)
        for order in orders
    ]
    times_full_first = [
        time_core_create_precomputing_sets(order, dr=dr_first_derivative, dt=dt_first_derivative, flag_radial=False, Nmean=Nmean)
        for order in orders
    ]
    times_radial_second = [
        time_core_create_precomputing_sets(order, dr=dr_second_derivative, dt=dt_second_derivative, flag_radial=True, Nmean=Nmean)
        for order in orders
    ]
    times_full_second = [
        time_core_create_precomputing_sets(order, dr=dr_second_derivative, dt=dt_second_derivative, flag_radial=False, Nmean=Nmean)
        for order in orders
    ]

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(orders, times_radial_unique, label="Radial Unique", marker='o')
    plt.plot(orders, times_full_unique, label="Full Unique", marker='o')
    plt.plot(orders, times_radial_first, label="Radial First Derivative", marker='o')
    plt.plot(orders, times_full_first, label="Full First Derivative", marker='o')
    plt.plot(orders, times_radial_second, label="Radial Second Derivative", marker='o')
    plt.plot(orders, times_full_second, label="Full Second Derivative", marker='o')
    plt.yscale('log')
    plt.xlabel("Maximum Order")
    plt.ylabel("Time (seconds, log scale)")
    plt.title("Timing of 'core_create_precomputing_sets' function for all Zernike polynomials up to a given order")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the results in an image file
    file_fig = os.path.join(os.path.dirname(__file__), "results", "timing_core_create_precomputing_sets.png")
    os.makedirs(os.path.dirname(file_fig), exist_ok=True)
    os.remove(file_fig) if os.path.exists(file_fig) else None
    plt.savefig(file_fig, dpi=300)

    # Show the plot
    plt.show()