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

# Matrix example
M = Matrix([[x, phi], [alpha, beta]])

# Vector example
G = Matrix([phi, x])

# Scalar example
s = alpha*phi**2+beta*x_dot**2

# A basic model with some equations
model = {
    'M': Equation(M, [q], parameter_symbols),
    's': Equation(s, [q, q_dot], parameter_symbols),
    'G': Equation(G, [q], parameter_symbols),
    'G_2DVector': Equation(G, [q], parameter_symbols, reshape_vectors=False)
}

# All of the above is generally run just once. Now we can run the lambdified functions as follows

# Test evaluations
q0 = array([0, 1])
qdot0 = array([0, 1])
testpars = [1, 2]

print('M')
print(model['M'].lambdified(q0, testpars))
print('G')
print(model['G'].lambdified(q0, testpars))
print('s')
print(model['s'].lambdified(q0, qdot0, testpars))

# It is often more convenient to substitute the parameters in a verbose way:
pars = {'beta': 2, 'alpha': 1}

# Substitute numeric values
for (name, equation) in model.items():
    equation.insert_parameters(pars)

# Now we can call the functions without the parameters
print('M')
print(model['M'].function(q0))

print('G')
print(model['G'].function(q0))

print('s')
print(model['s'].function(q0, qdot0))

# Vector that we explicitly kept as 2d array
print('G_2DVector')
print(model['G_2DVector'].function(q0))
