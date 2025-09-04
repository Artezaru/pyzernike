# pyzernike

## Description

`pyzernike` is a Python package to compute Zernike polynomials and their derivatives. 
See the documentation below for more informations.

Example of usage :

```python
import numpy
from pyzernike import zernike_polynomial

rho = numpy.linspace(0, 1, 100)
theta = numpy.linspace(0, 2*numpy.pi, 100)
result = zernike_polynomial(rho, theta, n=[2], m=[0])
polynomial = result[0]  # result is a list, we take the first element
```

Also compute the symbolic `sympy` expression and display the polynomials as follow :

![Zernike](./pyzernike/resources/zernike_display.png)

## Authors

- Artezaru <artezaru.github@proton.me>

- **Git Plateform**: https://github.com/Artezaru/pyzernike.git
- **Online Documentation**: https://Artezaru.github.io/pyzernike

## Installation

Install with pip

```
pip install pyzernike-polynomials
```

```
pip install git+https://github.com/Artezaru/pyzernike.git
```

Clone with git

```
git clone https://github.com/Artezaru/pyzernike.git
```

## License

Copyright 2025 Artezaru

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
