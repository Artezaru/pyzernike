"""
Alignment of an annular pupil
======================================================================================================

This example estimates the position and the size of an annular pupil (a telescope with a central
obscuration) from a measured wavefront, with the derivatives of the Zernike polynomials with respect to
the parameters of the pupil.

"""

# %%
# Measured wavefront
# -----------------------------------------------------------------
#
# The aberrations of the optics are known (spherical aberration and coma), but the pupil is decentered and
# slightly scaled on the detector. The measured wavefront is:
#
# .. math::
#
#     W(x, y; \vec{p}) = \sum_j c_j Z_j(x, y; \vec{p}) \quad \text{with} \quad \vec{p} = (x_0, y_0, R_x, R_y)
#
# where the polynomials are defined on the elliptic annulus of parameters :math:`\vec{p}` and :math:`h = 0.3`.

import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pyzernike

rng = numpy.random.default_rng(0)
h = 0.3
order = 4
c = numpy.zeros(pyzernike.zernike_polynomials_count(order))
c[pyzernike.zernike_order_to_index(n=[4, 3, 3], m=[0, 1, -1])] = [0.5, 0.2, -0.1]


def wavefront(x, y, p):
    return pyzernike.xy_zernike_evaluate(c, x, y, x0=p[0], y0=p[1], Rx=p[2], Ry=p[3], h=h)


x, y = numpy.meshgrid(numpy.linspace(-1.2, 1.2, 151), numpy.linspace(-1.2, 1.2, 151))
true_p = numpy.array([0.08, -0.05, 1.04, 0.97])
measured = wavefront(x, y, true_p) + rng.normal(scale=0.005, size=x.shape)  # NaN outside the pupil

# %%
# Gauss-Newton estimation of the parameters
# -----------------------------------------------------------------
#
# The parameters minimize :math:`\frac{1}{2} \sum_k (W(x_k, y_k; \vec{p}) - W_k)^2`.
# :func:`pyzernike.xy_zernike_least_squares` returns the residual function :math:`r(\vec{p})` and its exact Jacobian
# :math:`J(\vec{p})`, which can be given to any least squares solver (here a Gauss-Newton algorithm in a few lines).
# The coefficients are known: they can not be estimated together with the position and the size of the pupil,
# whose changes would be compensated by changes of the coefficients.

residual, jacobian = pyzernike.xy_zernike_least_squares(
    x, y, measured, order, parameters=("x0", "y0", "Rx", "Ry"), coefficients=c, h=h,
)

p = numpy.array([0.0, 0.0, 1.0, 1.0])  # Initial guess: centered unit pupil
for iteration in range(6):
    r = residual(p)
    step = numpy.linalg.lstsq(jacobian(p), -r, rcond=None)[0]
    p = p + step
    print(f"Iteration {iteration}: RMS residual = {numpy.sqrt(numpy.mean(r ** 2)):.5f}, |step| = {numpy.linalg.norm(step):.1e}")

print("True parameters:     ", true_p)
print("Estimated parameters:", numpy.round(p, 4))

# %%
# Measured and estimated pupils
# -----------------------------------------------------------------
#
# The outer and inner boundaries of the initial guess (dashed) and of the estimated pupil (solid) are drawn on
# the measured wavefront. After the estimation, the residuals contain only the noise of the measurement.

initial_p = numpy.array([0.0, 0.0, 1.0, 1.0])
residual_map = wavefront(x, y, p) - measured

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
images = (measured, residual_map)
titles = ("Measured wavefront", "Residuals after the estimation")
for axis, image, title in zip(axes, images, titles):
    mappable = axis.imshow(image, extent=(-1.2, 1.2, -1.2, 1.2), origin="lower", cmap="RdBu_r")
    fig.colorbar(mappable, ax=axis, shrink=0.85)
    axis.set_title(title)
    for parameters_set, style, label in ((initial_p, "--", "Initial guess"), (p, "-", "Estimated pupil")):
        for ratio in (1.0, h):
            axis.add_patch(Ellipse(
                (parameters_set[0], parameters_set[1]),
                width=2 * ratio * parameters_set[2],
                height=2 * ratio * parameters_set[3],
                fill=False,
                linestyle=style,
                color="black",
                label=label if ratio == 1.0 else None,
            ))
axes[0].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()