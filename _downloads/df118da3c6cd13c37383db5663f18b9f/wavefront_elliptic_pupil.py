"""
Wavefront analysis on an elliptic pupil
======================================================================================================

This example measures the aberrations of a wavefront on an off-axis (elliptic) pupil: the wavefront is
fitted with the Zernike polynomials, its RMS is obtained from the coefficients, and its slopes are
computed exactly from the fitted coefficients.

"""

# %%
# Synthetic wavefront on an elliptic pupil
# -----------------------------------------------------------------
#
# For an off-axis field, the effective pupil of a circular aperture is an ellipse, described by its
# semi-axes :math:`R_x, R_y` and its rotation angle :math:`\alpha`. The wavefront is a sum of defocus,
# astigmatism and coma (coefficients in the orthonormal basis, ``normalize=True``), evaluated with
# :func:`pyzernike.xy_zernike_evaluate` and measured with noise on a regular grid.

import numpy
import matplotlib.pyplot as plt
import pyzernike

rng = numpy.random.default_rng(0)
pupil = dict(Rx=1.0, Ry=0.7, alpha=numpy.deg2rad(30.0))
order = 4

true_coefficients = numpy.zeros(pyzernike.zernike_polynomials_count(order))
true_coefficients[pyzernike.zernike_order_to_index(n=[2, 2, 3], m=[0, 2, 1])] = [0.30, -0.15, 0.10]  # waves

x, y = numpy.meshgrid(numpy.linspace(-1, 1, 201), numpy.linspace(-1, 1, 201))
wavefront = pyzernike.xy_zernike_evaluate(true_coefficients, x, y, normalize=True, **pupil)  # NaN outside the pupil
wavefront += rng.normal(scale=0.01, size=wavefront.shape)

# %%
# Fit of the Zernike coefficients
# -----------------------------------------------------------------
#
# :func:`pyzernike.xy_zernike_fit` solves the linear least squares problem on the valid points (the NaN outside
# the pupil are ignored). As the mapping onto the ellipse is affine, the polynomials stay orthonormal on the
# pupil, and :func:`pyzernike.zernike_rms` gives the RMS of the wavefront from the coefficients.

coefficients, residuals = pyzernike.xy_zernike_fit(x, y, wavefront, order=order, normalize=True, return_residuals=True, **pupil)

n, m = pyzernike.zernike_index_to_order(list(range(len(coefficients))))
for n_j, m_j, c_j in zip(n, m, coefficients):
    if abs(c_j) > 0.02:
        print(f"Z_{n_j}^{m_j:+d}: {c_j:+.4f} waves")

print(f"RMS from the coefficients: {pyzernike.zernike_rms(coefficients, normalize=True):.4f} waves")
print(f"RMS of the measured data:  {numpy.nanstd(wavefront):.4f} waves")
print(f"RMS of the residuals:      {numpy.nanstd(residuals):.4f} waves")

# %%
# Slopes of the fitted wavefront
# -----------------------------------------------------------------
#
# The slopes :math:`\partial W / \partial x` and :math:`\partial W / \partial y` (measured for example by a
# Shack-Hartmann sensor) are computed exactly from the fitted coefficients with
# :func:`pyzernike.xy_zernike_evaluate`.

slope_x, slope_y = pyzernike.xy_zernike_evaluate(coefficients, x, y, x_derivative=[1, 0], y_derivative=[0, 1], normalize=True, **pupil)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, image, title in zip(axes, (wavefront, slope_x, slope_y), ("Wavefront", "Slope along x", "Slope along y")):
    mappable = axis.imshow(image, extent=(-1, 1, -1, 1), origin="lower", cmap="RdBu_r")
    axis.set_title(title)
    fig.colorbar(mappable, ax=axis, shrink=0.8)
plt.tight_layout()
plt.show()