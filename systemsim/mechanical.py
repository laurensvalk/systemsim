"""Provides classes for mechanical systems."""
import numpy as np
from .core import System


class LagrangianMechanicalSystem(System):
    """Mechanical system in Lagrangian formulation."""

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

        # Determine input/output size by checking size of F:
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

    @property
    def state_names(self):
        """Return state names as strings."""
        return ['q_' + str(i) for i in range(self.n)] + ['q_dot' + str(i) for i in range(self.n)]

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
        ).reshape(self.n)

        return np.append(q_dot, q_dotdot).reshape(2*self.n)


class HamiltonianMechanicalSystem(System):
    """Mechanical system in Hamiltonian formulation."""

    def __init__(
            self,
            M,
            F,
            dHdq,
            dHdp,
            V,
            R=None,
            q_initial=None,
            p_initial=None,
            q_dot_initial=None,
            exogenous_input_function=None):
        """Initialize the mechanical system."""
        # Store matrices
        self.M = M
        self.F = F

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

        # Assert that initial speed and momenta are not both specified
        assert q_dot_initial is None or p_initial is None, \
            "Specify either q_dot_initial or p_initial, but not both"

        if p_initial is None:
            p_initial = np.zeros(self.n)

        if q_dot_initial is not None:
            p_initial = self.M(q_initial)@q_dot_initial

        x_initial = np.append(q_initial, p_initial)
        n_states = self.n*2

        # Determine input/output size by checking size of F:
        self.m = self.F(q_initial).shape[self.COL]
        n_outputs = n_inputs = self.m

        # Store remaining system generators or create
        # them as zeros otherwise
        self.dHdq = dHdq
        self.dHdp = dHdp
        self.V = V if V is not None else lambda q: 0
        self.R = R if R is not None else lambda q, p: np.zeros((self.n, self.n))

        # Initialize system object
        System.__init__(self, n_states, n_inputs, n_outputs,
                        x_initial, exogenous_input_function)

    @property
    def state_names(self):
        """Return state names as strings."""
        return ['q_' + str(i) for i in range(self.n)] + ['p' + str(i) for i in range(self.n)]

    def H(self, q, p):
        """Compute the Hamiltonian: Total energy."""
        M_inverse_times_p = np.linalg.solve(self.M(q), p)
        return p.T@M_inverse_times_p/2 + self.V(q)

    def get_coordinates(self, state):
        """Return q, p from x."""
        q = state[0:self.n]
        p = state[self.n:2*self.n+1]
        return q, p

    def output(self, state, time=None):
        """Return mechanical system Output."""
        q, p = self.get_coordinates(state)
        return self.F(q).T@self.dHdp(q, p)

    def state_feedback(self, state, time=None):
        """Return internal mechanical system feedback."""
        return self.zero_input

    def equations_of_motion(self, state, tau, time=None):
        """Evaluate equations of motion for mechanical system."""
        # Coordinates
        q, p = self.get_coordinates(state)

        # Hamiltonian gradient
        dHdq = self.dHdq(q, p).reshape((self.n))
        dHdp = self.dHdp(q, p).reshape((self.n))
        dHdx = np.append(dHdq, dHdp)

        # Input and damping matrices
        F = self.F(q)
        R = self.R(q, p)

        # Zero and identities needed in equations of motion
        zero_nn = np.zeros((self.n, self.n))
        zero_nm = np.zeros((self.n, self.m))
        identity = np.eye(self.n, self.n)

        # Evaluation of equations of motion matrices
        J_full = np.append(
            np.append(zero_nn, identity, axis=self.COL),
            np.append(-identity, -R, axis=self.COL),
            axis=self.ROW
        )
        input_full = np.append(zero_nm, F, axis=self.ROW)

        # Return the result of the equations of motion matrices
        x_dot = J_full@dHdx + input_full@tau.reshape(self.m)

        return x_dot
