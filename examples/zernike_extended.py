from pyzernike import xy_zernike_polynomial
import numpy
import matplotlib.pyplot as plt


# Create a grid of points in Cartesian coordinates
x = numpy.linspace(-1, 1, 100)
y = numpy.linspace(-1, 1, 100)
x, y = numpy.meshgrid(x, y)

# Define the ellipse parameters
Rx = 1.0
Ry = 1.0
h = 0.1
alpha = numpy.pi/4
theta1 = - numpy.pi / 2
theta2 = numpy.pi / 2

# Compute the extended Zernike polynomial for a specific order and degree
n = 4
m = 2

default_valus = xy_zernike_polynomial(x, y, n, m)
zernike_values = xy_zernike_polynomial(x, y, n, m, Rx=Rx, Ry=Ry, h=h, alpha=alpha, theta1=theta1, theta2=theta2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.title(f'Zernike Polynomial Z_{n}^{m} (default)')
plt.contourf(x, y, default_valus, levels=50, cmap='viridis')
plt.colorbar(label='Zernike Value')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.subplot(1, 2, 2)
plt.title(f'Extended Zernike Polynomial eZ_{n}^{m} (Rx={Rx}, Ry={Ry}, h={h}, alpha={alpha})')
plt.contourf(x, y, zernike_values, levels=50, cmap='viridis')
plt.colorbar(label='Extended Zernike Value')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()
plt.show() 



