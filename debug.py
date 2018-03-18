#!/usr/bin/env python3
"""Debug Script."""

from numpy import array
from systemsim.linear import LTI, Integrator
from systemsim.network import Collection

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
    ([plant], lambda time: one)
]
output_signals = [
    {plant: 3}
]
closedloop1 = Collection(systems,
                         input_signals,
                         output_signals)
