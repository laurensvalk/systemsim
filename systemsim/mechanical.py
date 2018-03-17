"""Provides classes for mechanical systems."""
import numpy as np
from .core import System


class LagrangianMechanicalSystem(System):
    """Mechanical system in Lagrangian formulation."""

    def __init__(
            self,
            n,
            m,
            M,
            F,
            C=None,
            G=None,
            Q=None,
            q_initial=None,
            q_dot_initial=None,
            exogenous_input_function=None):
        """Initialize the Lagrangian mechanical system."""
        # Number of coordinates and actuators
        self.n = n
        self.m = m
        n_outputs = n_inputs = self.m

        # Initial states values and size
        if q_initial is None:
            q_initial = np.zeros(self.n)
        if q_dot_initial is None:
            q_dot_initial = np.zeros(self.n)

        # System states and initial condition
        x_initial = np.append(q_initial, q_dot_initial)
        n_states = self.n*2

        # Store matrix generators or create them as zeros otherwise
        self.M = M
        self.F = F
        self.C = C if C is not None else lambda q, q_dot: np.zeros((self.n, self.n))
        self.Q = Q if Q is not None else lambda q, q_dot: np.zeros((self.n, ))
        self.G = G if G is not None else lambda q: np.zeros((self.n, ))

        # Assert matrix/vector dimension compatibility:
        assert self.M(q_initial).shape == (self.n, self.n)
        assert self.F(q_initial).shape == (self.n, self.m)
        assert self.G(q_initial).shape == (self.n, )
        assert self.C(q_initial, q_dot_initial).shape == (self.n, self.n)
        assert self.Q(q_initial, q_dot_initial).shape == (self.n,)

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
        # Extract coordinates and velocities from state
        q, q_dot = self.get_coordinates(state)

        # Explicitly reshape input to array if it isn't already
        tau = tau.reshape((self.m, ))

        # Compute matrix/ vector components for equations of motion
        M = self.M(q)
        F = self.F(q)
        C = self.C(q, q_dot)
        Q = self.Q(q, q_dot)
        G = self.G(q)

        # Evaluate generalized accelerations
        q_dotdot = np.linalg.solve(
            M, Q + F@tau - G - C@q_dot
        )

        # Return the state change vector
        return np.append(q_dot, q_dotdot)


class HamiltonianMechanicalSystem(System):
    """Mechanical system in Hamiltonian formulation."""

    def __init__(
            self,
            n,
            m,
            M,
            F,
            dHdq,
            dHdp,
            V,
            R,
            q_initial=None,
            p_initial=None,
            exogenous_input_function=None):
        """Initialize the Hamiltonian mechanical system."""
        # Number of coordinates and actuators
        self.n = n
        self.m = m
        n_outputs = n_inputs = self.m

        # Initial states values and size
        if q_initial is None:
            q_initial = np.zeros(self.n)
        if p_initial is None:
            p_initial = np.zeros(self.n)

        # System states and initial condition
        x_initial = np.append(q_initial, p_initial)
        n_states = self.n*2

        # Store system generators
        self.M = M
        self.F = F
        self.dHdq = dHdq
        self.dHdp = dHdp
        self.V = V
        self.R = R

        # Assert matrix/vector dimension compatibility:
        assert self.M(q_initial).shape == (self.n, self.n)
        assert self.F(q_initial).shape == (self.n, self.m)
        assert self.R(q_initial, p_initial).shape == (self.n, self.n)
        assert self.dHdq(q_initial, p_initial).shape == self.dHdp(q_initial, p_initial).shape == (self.n,)

        # Initialize system object
        System.__init__(self, n_states, n_inputs, n_outputs,
                        x_initial, exogenous_input_function)

    @property
    def state_names(self):
        """Return state names as strings."""
        return ['q_' + str(i) for i in range(self.n)] + ['p' + str(i) for i in range(self.n)]

    @staticmethod
    def p_initial(M, q_initial, q_dot_initial):
        """Convert initial velocity to initial momenta."""
        return M(q_initial)@q_dot_initial

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
        # Extract coordinates and velocities from state
        q, p = self.get_coordinates(state)

        # Explicitly reshape input to array if it isn't already
        tau = tau.reshape((self.m, ))

        # Hamiltonian gradient
        dHdq = self.dHdq(q, p)
        dHdp = self.dHdp(q, p)
        dHdx = np.append(dHdq, dHdp)

        # Input and damping matrices
        F = self.F(q)
        R = self.R(q, p)

        # Zero and identities needed in equations of motion
        zero_nn = np.zeros((self.n, self.n))
        zero_nm = np.zeros((self.n, self.m))
        identity = np.eye(self.n, self.n)

        # Evaluation of equations of motion matrices
        J_matrix = np.append(
            np.append(zero_nn, identity, axis=self.COL),
            np.append(-identity, -R, axis=self.COL),
            axis=self.ROW
        )
        input_matrix = np.append(zero_nm, F, axis=self.ROW)

        # Return the result of the equations of motion matrices
        x_dot = J_matrix@dHdx + input_matrix@tau

        return x_dot
