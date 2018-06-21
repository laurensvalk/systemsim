systemsim: a python simulator for dynamical systems and networks
================================================================

A general purpose Python simulator for networks of systems with the dynamics :math:`\dot{\boldsymbol{x}}_i=\boldsymbol{f}_i(\boldsymbol{x}_i,\boldsymbol{u}_i,t)` and the output :math:`\boldsymbol{y}_i=\boldsymbol{h}_i(\boldsymbol{x}_i,t)`.

This package is compatible with Python 3 and requires ``numpy`` for matrix and vector operations, ``sympy`` for symbolic derivations, and ``scipy`` for ODE integration.

Though not required, it is recommended to use this simulator in a ``jupyter`` notebook. Any library can be used to generate graphs, but the provided examples rely on ``plotly`` to generate interactive plots.

Examples and usage instructions are given in this repository_.

This package was developed to generate simulation results in the MSc thesis entitled *Distributed Control of Underactuated and Heterogeneous Mechanical Systems* which can be viewed here_.

.. _repository: https://github.com/laurensvalk/systemsim-examples
.. _here: https://repository.tudelft.nl/
