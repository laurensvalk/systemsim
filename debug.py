#!/usr/bin/env python3
"""Debug Script."""
import numpy as np
from systemsim.linear import LTI, Integrator
from systemsim.network import Interconnection

A = np.array([[-1]])
B = C = np.array([[1]])
one = np.array([1])
zero = np.array([0])
systems = {
    'plant': LTI(A, B, C, x_initial=-1*one),
    'controller': Integrator(zero)
}
system_connections = {
    ('controller', 'plant'): 1,
    ('plant', 'controller'): -1
}

input_connections = {
    'reference': ['controller'],
    'disturbance': ['plant']
}
exogenous_input_functions = {
    'reference': lambda time: 2*one,
    'disturbance': lambda time: one
}
exogenous_output_functions = {
    'error': lambda output, through: output['plant'] - through['reference']
}
network = Interconnection(
    systems, system_connections, input_connections,
    exogenous_input_functions, exogenous_output_functions
)
time_steps = np.arange(start=0, stop=15, step=0.01)
network.simulate(time_steps)
for (system_name, system) in systems.items():
    print(system_name)
    print(system.plot_state_trajectory())
