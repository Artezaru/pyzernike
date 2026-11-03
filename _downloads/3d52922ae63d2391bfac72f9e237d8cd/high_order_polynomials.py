"""
Accuracy and speed at high orders
======================================================================================================

This example compares the two methods available to compute all the Zernike polynomials up to a high
order: the explicit sum and the recurrence relation of Shakibaei and Paramesran.

"""

# %%
# Explicit sum and recurrence relation
# -----------------------------------------------------------------
#
# With ``use_recursive=False``, the radial polynomials are computed with the explicit sum of terms with
# alternating signs, whose coefficients grow quickly with the order: the cancellation between these terms
# limits the accuracy in double precision. With ``use_recursive=True``, a three-term recurrence relation is
# used, which stays close to the machine precision at any order and costs fewer operations.

import time

import numpy
import matplotlib.pyplot as plt
import pyzernike

order = 40
rho, theta = numpy.meshgrid(numpy.linspace(0, 1, 300), numpy.linspace(0, 2 * numpy.pi, 300))

timings = {}
results = {}
for use_recursive in (False, True):
    start = time.perf_counter()
    results[use_recursive] = pyzernike.zernike_polynomial_up_to_order(rho, theta, order=order, use_recursive=use_recursive, to_list=False)[0]
    timings[use_recursive] = time.perf_counter() - start

print(f"{pyzernike.zernike_polynomials_count(order)} polynomials on {rho.size} points")
print(f"Explicit sum:        {timings[False]:.2f} s")
print(f"Recurrence relation: {timings[True]:.2f} s")

# %%
# Error of the explicit sum
# -----------------------------------------------------------------
#
# The recurrence relation is taken as the reference. The maximum absolute error of the explicit sum is
# shown for each radial order :math:`n` (maximum over the azimuthal frequencies :math:`m`): it stays
# negligible up to :math:`n \approx 20` and then grows exponentially.

n, m = pyzernike.zernike_index_to_order(list(range(pyzernike.zernike_polynomials_count(order))))
errors = numpy.max(numpy.abs(results[False] - results[True]), axis=(1, 2))
max_error_per_order = [errors[numpy.asarray(n) == n_j].max() for n_j in range(order + 1)]

plt.figure(figsize=(6, 4))
plt.semilogy(range(order + 1), numpy.maximum(max_error_per_order, 1e-17), "o-")
plt.axhline(1e-10, color="gray", linestyle="--", label=r"$10^{-10}$")
plt.xlabel("Radial order $n$")
plt.ylabel("Maximum absolute error")
plt.title("Explicit sum compared to the recurrence relation")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# .. tip::
#
#     For orders above 20, use ``use_recursive=True`` in :func:`pyzernike.zernike_polynomial_up_to_order`,
#     :func:`pyzernike.xy_zernike_polynomial_up_to_order` and :func:`pyzernike.zernike_fit`.