#!/usr/bin/env python3
"""Debug Script."""

from sympy import symbols, Matrix
from numpy import array
from systemsim.symbolic import Equation

# Ordered list of parameter names
parameter_symbols = alpha, beta = symbols(['alpha', 'beta'])

# Coordinate variables
phi, x = q = Matrix(symbols(['phi', 'x']))
phi_dot, x_dot = q_dot = Matrix(symbols(['\dot{phi}', '\dot{x}']))

# A basic model with some equations
model = {
    'M': Equation(alpha*phi**2+beta*x**2, [q], parameter_symbols),
    'C': Equation(alpha*phi**2+beta*x_dot**2, [q, q_dot], parameter_symbols)
}

# All of the above is generally run just once.
# Below is what we would run many times in different simulations.
pars = {'beta': 2, 'alpha': 1}

# Substitute numeric values
for (name, equation) in model.items():
    equation.insert_parameters(pars)

# Test evaluations
q0 = array([0, 1])
qdot0 = array([0, 1])

print(model['M'].lambdified(q0, [1, 2]))
print(model['M'].function(q0))
print(model['C'].function(q0, qdot0))
