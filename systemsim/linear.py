"""Provides class for linear systems."""
import numpy as np
from .core import System


class LTI(System):
    """LTI System."""

    def __init__(
            self,
            A,
            B,
            C,
            K=None,
            x_initial=None,
            exogenous_input_function=None):
        """Initialize."""
        # Store matrices
        self.A, self.B, self.C = A, B, C

        # Get system size from matrices
        n_states = self.A.shape[self.ROW]
        n_inputs = self.B.shape[self.COL]
        n_outputs = self.C.shape[self.ROW]

        # Set k matrix to zero if not supplied as argument
        self.K = K if K is not None else np.zeros((n_inputs, n_states))

        # Throw an error if matrix x_dimensions are not compatible:
        assert A.shape[self.ROW] == A.shape[self.COL]
        assert A.shape[self.ROW] == B.shape[self.ROW] == C.shape[self.COL]
        assert self.K.shape[self.ROW] == n_inputs
        assert self.K.shape[self.COL] == n_states

        # Initialize System object
        System.__init__(self, n_states, n_inputs, n_outputs,
                        x_initial, exogenous_input_function)

    def output(self, state, time):
        """Generate LTI output without feedthrough."""
        return self.C@state

    def state_feedback(self, state, time=None):
        """Output LTI state feedback."""
        return -self.K@state

    def equations_of_motion(self, state, total_input, time=None):
        """Evaluate equations of motion for LTI."""
        return self.A@state + self.B@total_input


class Integrator(LTI):
    """Define linear integrator model."""

    def __init__(self, x_initial):
        """Initialize ABC matrices and create integrator object."""
        n_states = x_initial.shape[self.ROW]
        A = np.zeros((n_states, n_states))
        B = C = np.eye(n_states)
        LTI.__init__(self, A, B, C, x_initial=x_initial)
