Welcome to pyzernike's documentation!
=========================================================================================

Description of the package
--------------------------

Zernike polynomials computation and visualization.

The Zernike polynomials are defined as follows:

.. math::

    Z_{n}^{m}(\rho, \theta) = R_{n}^{m}(\rho) \cos(m \theta) \quad \text{if} \quad m \geq 0

.. math::
    
    Z_{n}^{m}(\rho, \theta) = R_{n}^{-m}(\rho) \sin(-m \theta) \quad \text{if} \quad m < 0

with :

.. math::
   
   R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

where :math:`n` is the radial order, :math:`m` is the azimuthal frequency, :math:`\rho` is the normalized radial coordinate (:math:`0 \leq \rho \leq 1`) and :math:`\theta` is the azimuthal angle.

Contents
--------

The documentation is divided into the following sections:

- **Installation**: This section describes how to install the package.
- **API Reference**: This section contains the documentation of the package's API.
- **Usage**: This section contains the documentation of how to use the package.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   ./installation
   ./mathematical_description
   ./api
   ./usage


A terminal commmand is created to plot the first zernike polynomials. The command is called `pyzernike` and can be used as follows:

.. code-block:: bash

    pyzernike

.. image:: ../../pyzernike/resources/zernike_display.png
    :align: center

Author
------

The package ``pyzernike`` was created by the following authors:

- Artezaru <artezaru.github@proton.me>

You can access the package and the documentation with the following URL:

- **Git Plateform**: https://github.com/Artezaru/pyzernike.git
- **Online Documentation**: https://Artezaru.github.io/pyzernike

License
-------

Please refer to the [LICENSE] file for the license of the package.
