"""Provides classes for mechanical systems."""
import numpy as np
from .core import System


class SimpleMechanicalSystem(System):
    """Mechanical System."""

    def __init__(
            self,
            M,
            F,
            C=None,
            G=None,
            Q=None,
            q_initial=None,
            q_dot_initial=None,
            exogenous_input_function=None):
        """Initialize the mechanical system."""
        # Store matrices
        self.M, self.F = M, F

        # Determine number of coordinates from shape of M in order to avoid
        # the user having to supply it
        for attempts in range(1, 100):
            try:
                # Try large number of coordinates until success
                self.n = M(np.zeros((attempts, 1))).shape[0]
                break
            except (IndexError, TypeError):
                pass

        # Initial states values and size
        if q_initial is None:
            q_initial = np.zeros(self.n)
        if q_dot_initial is None:
            q_dot_initial = np.zeros(self.n)
        x_initial = np.append(q_initial, q_dot_initial)
        n_states = self.n*2

        # Determine input/output size by checking size of G:
        self.m = self.F(q_initial).shape[self.COL]
        n_outputs = n_inputs = self.m

        # Store remaining system matrix generators or create
        # them as zeros otherwise
        self.C = C if C is not None else lambda q, q_dot: np.zeros((self.n,
                                                                    self.n))
        self.Q = Q if Q is not None else lambda q, q_dot: np.zeros((self.n, 1))
        self.G = G if G is not None else lambda q: np.zeros((self.n, 1))

        # Initialize system object
        System.__init__(self, n_states, n_inputs, n_outputs,
                        x_initial, exogenous_input_function)

        # Overwrite state names
        self.state_names = ['q_' + str(i) for i in range(self.n)] + \
                           ['q_dot' + str(i) for i in range(self.n)]

    def get_coordinates(self, state):
        """Return q, q_dot from x."""
        q = state[0:self.n]
        q_dot = state[self.n:2*self.n+1]
        return q, q_dot

    def output(self, state, time=None):
        """Return mechanical system Output."""
        q, q_dot = self.get_coordinates(state)
        return self.F(q).T@q_dot

    def state_feedback(self, state, time=None):
        """Return internal mechanical system feedback."""
        return self.zero_input

    def equations_of_motion(self, state, tau, time=None):
        """Evaluate equations of motion for mechanical system."""
        q, q_dot = self.get_coordinates(state)
        M = self.M(q)
        F = self.F(q)
        C = self.C(q, q_dot)
        Q = self.Q(q, q_dot)
        G = self.G(q)
        q_dotdot = np.linalg.solve(
            M, Q+F@tau.reshape(self.m, 1)-G-C@q_dot.reshape(self.n, 1)
        )
        return np.append(q_dot, q_dotdot).reshape(2*self.n)
