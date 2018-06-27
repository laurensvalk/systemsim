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
            ell,
            M,
            F,
            V,
            R,
            dH_dq,
            dV_dq,
            animation_kinematics=None,
            q_initial=None,
            p_initial=None,
            exogenous_input_function=None):
        """Initialize the Hamiltonian mechanical system."""
        # Number of coordinates and actuators
        self.n = n
        self.m = m

        # Task output dimension (used in higher controlled class only)
        if ell is None:
            self.ell = self.m
        else:
            self.ell = ell

        n_outputs = n_inputs = self.ell

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
        self.dH_dq = dH_dq
        self.dV_dq = dV_dq
        self.V = V
        self.R = R

        # Assert matrix/vector dimension compatibility:
        assert self.M(q_initial).shape == (self.n, self.n)
        assert self.F(q_initial).shape == (self.n, self.m)
        assert self.R(q_initial, p_initial).shape == (self.n, self.n)
        assert self.dH_dq(q_initial, p_initial).shape == self.dV_dq(q_initial).shape == (self.n,)

        # Store the animations kinematic map
        self.animation_kinematics = animation_kinematics

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
        return self.F(q).T@self.dH_dp(q, p)

    def state_feedback(self, state, time=None):
        """Return internal mechanical system feedback."""
        return self.zero_input

    def dH_dp(self, q, p):
        """Compute dH/dp as M_inverse*p."""
        return np.linalg.solve(self.M(q), p)

    def equations_of_motion(self, state, tau, time=None):
        """Evaluate equations of motion for mechanical system."""
        # Extract coordinates and velocities from state
        q, p = self.get_coordinates(state)

        # Explicitly reshape input to array if it isn't already
        tau = tau.reshape((self.m, ))

        # Hamiltonian gradient
        dH_dq = self.dH_dq(q, p)
        dH_dp = self.dH_dp(q, p)
        dHdx = np.append(dH_dq, dH_dp)

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

    def post_simulation_processing(self):
        """Process simulation results. Compute state dependent signals."""
        # Extract position and momenta at every time sample
        q_and_p = [
            self.get_coordinates(self.state_trajectory[:, k])
            for (k, time) in enumerate(self.simulation_time)
        ]

        # z at every timestep
        self.z_trajectory = np.stack([self.z(q) for (q, p) in q_and_p], axis=self.COL)

        # H at every timestep
        self.H_trajectory = np.stack([self.H(q, p) for (q, p) in q_and_p], axis=self.COL)

    def get_animation_kinematics(self, state):
        """Create line of 2D or 3D points to plot for given state."""
        # Extract coordinates and velocities from state
        q, p = self.get_coordinates(state)

        # Evaluate animation map
        points = self.animation_kinematics(q)

        # Points to draw in x and y
        xpoints = points[0, :].tolist()
        ypoints = points[1, :].tolist()

        # Return the points
        return xpoints, ypoints


class IDAPBCAgent(HamiltonianMechanicalSystem):
    """Hamiltonian mechanical system with IDA-PBC as state feedback.

    This class is similar to HamiltonianMechanicalSystem, but the
    state feedback law now included. It is generated using the
    passivity-based control by damping and interconnection assignment.

    The output function now corresponds the output of the closed loop
    Hamiltonian.
    """

    def __init__(
            self,
            n,
            m,
            ell,
            M,
            F,
            V,
            R,
            dH_dq,
            dV_dq,
            Mcl,
            Vcl,
            dHcl_dq,
            dVcl_dq,
            J2,
            Kv,
            Kd,
            z,
            Psi,
            animation_kinematics=None,
            q_initial=None,
            p_initial=None,
            exogenous_input_function=None):
        """Initialize the IDA Hamiltonian mechanical system."""
        # Store system generators
        self.Mcl = Mcl
        self.Vcl = Vcl
        self.dHcl_dq = dHcl_dq
        self.dVcl_dq = dVcl_dq
        self.J2 = J2
        self.Kv = Kv
        self.Kd = Kd
        self.z = z
        self.Psi = Psi

        # Assert compatible dimensions
        zero = np.zeros(n)
        assert self.Mcl(zero).shape == (n, n)
        assert self.J2(zero, zero).shape == (n, n)
        assert self.Kv().shape == (m, m)
        assert self.Kd().shape == (ell, ell)
        assert self.dHcl_dq(zero, zero).shape == (n,)
        assert self.dVcl_dq(zero).shape == (n,)
        assert self.z(zero).shape == (ell,)
        assert self.Psi(zero).shape == (n, ell)

        # Pass on the remaining arguments to the Hamiltonian system
        HamiltonianMechanicalSystem.__init__(self, n, m, ell, M, F, V, R, dH_dq, dV_dq, animation_kinematics,
                                             q_initial, p_initial, exogenous_input_function)

    def state_feedback(self, state, time):
        """Calculate IDA PBC feedback law."""
        # Extract coordinates and velocities from state
        q, p = self.get_coordinates(state)

        # Compute control law components
        F = self.F(q)
        M = self.M(q)
        J2 = self.J2(q, p)
        Kv = self.Kv()
        Kd = self.Kd()
        Mcl = self.Mcl(q)
        dH_dq = self.dH_dq(q, p)
        dHcl_dq = self.dHcl_dq(q, p)
        dHcl_dp = self.dHcl_dp(q, p)
        ycl = F.T@self.dHcl_dp(q, p)
        Psi = self.Psi(q)

        # Compute the IDA PBC law
        tau = np.linalg.solve(
            F.T@F,
            F.T@(
                dH_dq - Mcl@np.linalg.solve(M, dHcl_dq) + J2@dHcl_dp -
                Mcl@np.linalg.solve(M, Psi@Kd@self.output(state))
            )
        ) - Kv@ycl

        return tau

    def total_input(self, external_control_input, exogenous_input, state_feedback, state, time):
        """Compute total input to system.

        Unlike the default case, the distributed control input enters
        through a state dependent (matched) input matrix.
        """
        # Extract coordinates and velocities from state
        q, p = self.get_coordinates(state)

        # Compute control law components
        F = self.F(q)
        M = self.M(q)
        Psi = self.Psi(q)
        Mcl = self.Mcl(q)

        # Compute effect of distributed control input
        external_inputs_matched = np.linalg.solve(
            F.T@F,
            F.T@Mcl@np.linalg.solve(
                M,
                Psi@(external_control_input+exogenous_input)
            )
        )
        # Return total control signal.
        return state_feedback + external_inputs_matched

    def output(self, state, time=None):
        """Return controlled Hamiltonian output."""
        q, p = self.get_coordinates(state)
        return self.Psi(q).T@np.linalg.solve(self.M(q), p)

    def dHcl_dp(self, q, p):
        """Compute dHcl/dp as Mcl_inverse*p."""
        return np.linalg.solve(self.Mcl(q), p)


class SymbolicIDAPBCAgent(IDAPBCAgent):
    """Like IDAPBCMechanicalSystem, but with symbolic arguments."""

    def __init__(
            self,
            model,
            parameters,
            q_initial=None,
            p_initial=None,
            exogenous_input_function=None):
        """Initialize the IDA Hamiltonian using symbolic definitions."""
        # Substitute parameters into model equations
        for (name, equation) in model.items():
            equation.insert_parameters(parameters)

        # Take the numeric functions from the model to
        # initialize IDA PBC Hamiltonian system
        IDAPBCAgent.__init__(
            self,
            model['M'].expression.shape[0],
            model['F'].expression.shape[1],
            model['Psi'].expression.shape[1],
            model['M'].function,
            model['F'].function,
            model['V'].function,
            model['R'].function,
            model['dH_dq'].function,
            model['dV_dq'].function,
            model['Mcl'].function,
            model['Vcl'].function,
            model['dHcl_dq'].function,
            model['dVcl_dq'].function,
            model['J2'].function,
            model['Kv'].function,
            model['Kd'].function,
            model['z'].function,
            model['Psi'].function,
            model['animation_kinematics'].function,
            q_initial,
            p_initial,
            exogenous_input_function,
        )
