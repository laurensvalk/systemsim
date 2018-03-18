#!/usr/bin/env python3
"""Debug Script."""

from numpy import array, arange
from systemsim.linear import LTI, Integrator
from systemsim.network import Collection, Interconnection

A = array([[-1]])
B = C = array([[1]])
one = array([1])
zero = array([0])
plant = LTI(A, B, C, x_initial=zero)
controller = Integrator(zero)
systems = [plant, controller]
connections = {
    (controller, plant): -1,
    (plant, controller): 1
}
input_signals = [
    ([plant], lambda time: one*4)
]
output_signals = [
    {plant: 4}
]
time = arange(start=0, step=0.1, stop=10)
collection = Collection(systems, input_signals, output_signals)

closedloop1 = Interconnection(systems,
                              connections,
                              input_signals,
                              output_signals)

closedloop1.simulate(time)
print(closedloop1.output_trajectory)