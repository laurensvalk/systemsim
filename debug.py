#!/usr/bin/env python3
"""Debug Script."""

from numpy import array, arange
from systemsim.linear import LTI, Integrator
from systemsim.network import Interconnection

A = array([[-1]])
B = C = array([[1]])
one = array([1])
zero = array([0])
plant = LTI(A, B, C, x_initial=-1*one)
controller = Integrator(zero)
systems = [plant, controller]
connections = {
    (controller, plant): 1,
    (plant, controller): -1
}


input_connections = [
    ([controller], lambda time: 2*one), # I.e. the reference
    ([plant], lambda time: one) # I.e. a disturbance
]
network = Interconnection(systems, connections, input_connections)
time_steps = arange(start=0, stop=15, step=0.01)
network.simulate(time_steps)
for system in systems:
    print(system.output_trajectory)
