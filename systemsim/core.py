"""Provides general system class from which all other systems are derived."""


# Use numpy for matrices and vectors
import numpy as np

# Use odeint for integrating equations of motion
from scipy.integrate import odeint


class System():
    """Generic system with states, inputs, outputs and simulation methods.

    This class defines the basic system structure from which all the other
    systems are derived. The Equations of motion are unspecified in this
    base class.
    """

    # Matrix/Vector index constants
    ROW = 0
    COL = 1

    def __init__(self, n_states, n_inputs, n_outputs,
                 x_initial=None, exogenous_input_function=None):
        """Initialize system."""
        # Set attributes
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Set initial state to zero if no initial state is supplied
        self.zero_state = np.zeros(n_states)
        self.zero_output = np.zeros(n_outputs)
        self.zero_input = np.zeros(n_inputs)
        self.x_initial = self.zero_state if x_initial is None else x_initial
        assert len(self.x_initial) == n_states

        # Empty simulation results
        self.simulation_time = np.empty(0)
        self.state_trajectory = np.empty((n_states, 0))
        self.output_trajectory = np.empty((n_outputs, 0))

        # Maximum state norm that we consider before cancelling integration
        self.max_state = 1e9

        # Store exogeneous generator if supplied,
        # otherwise create zero generator.
        zero_generator = lambda time: self.zero_input

        if exogenous_input_function is None:
            self.exogenous_input = zero_generator
        else:
            self.exogenous_input = exogenous_input_function

    @property
    def state_names(self):
        """Return state names."""
        return ['x_' + str(i) for i in range(self.n_states)]

    @property
    def input_names(self):
        """Return input names."""
        return ['u_' + str(i) for i in range(self.n_inputs)]

    @property
    def output_names(self):
        """Return output_names names."""
        return ['y_' + str(i) for i in range(self.n_outputs)]

    @staticmethod
    def concatenate(list_of_vectors):
        """Concatenate vectors as large vector. Give empty vector if empty."""
        if list_of_vectors:
            return np.concatenate(list_of_vectors)
        else:
            return np.empty(0)

    def print_dimensions(self):
        """Print the system size."""
        print(self.__dict__)

    def output(self, state, time):
        """Return output given the current state. Returns zero by default."""
        return self.zero_output

    def state_feedback(self, state, time):
        """Generate internal state feedback. Zero by default."""
        return self.zero_input

    def equations_of_motion(self, state, input, time):
        """Evaluate equations of motion."""
        return self.zero_state

    def total_input(self, external_control_input, exogenous_input, state_feedback, time):
        """Compute total input to system.

        By default, this is the sum of the state feedback, the exogenous input,
        and optionally an input from the network.
        """
        return external_control_input + exogenous_input + state_feedback

    def state_change(self, state, external_control_input, time):
        """Evaluate equations of motion subject to exogenous input."""
        exogenous_input = self.exogenous_input(time)
        state_feedback = self.state_feedback(state, time)
        total_input = self.total_input(external_control_input, exogenous_input, state_feedback, time)
        return self.equations_of_motion(state, total_input, time)

    def __state_change_ode(self, state, time):
        """Evaluate ordinary differential equation for ODE solver."""
        # Break the loop if we're exploding
        assert np.amax(np.abs(state)) <= self.max_state, \
            "State is too large. Unstable?"
        # When simulating this system on its own, the external input is zero
        external_control_input = self.zero_input
        # Compute the equation of motion given input at current time
        return self.state_change(state, external_control_input, time)

    def simulate(self, time_range):
        """Integrate the ordinary differential equations."""
        self.state_trajectory = odeint(self.__state_change_ode,
                                       self.x_initial, time_range).T
        self.simulation_time = time_range
        self.compute_output_trajectory()
        # Further processing if needed
        self.post_simulation_processing()

    def compute_output_trajectory(self):
        """Compute the outputs for the previously run simulation."""
        # Shortcut to calculate output at a given time sample k
        # in the simulation history
        output = lambda k: \
            self.output(self.state_trajectory[:, k],
                        self.simulation_time[k]).reshape((self.n_outputs, 1))

        # Calculate the outputs at all time steps and concatenate results
        self.output_trajectory = np.hstack(
            [output(k) for (k, time) in enumerate(self.simulation_time)]
        )

    def post_simulation_processing(self):
        """Process simulation results. The basic class does nothing here."""
        pass

    def plot_state_trajectory(self, states='all'):
        """Return state data dictionary compatible with Plotly Plots."""
        if states == 'all':
            states_to_plot = range(self.n_states)
        else:
            states_to_plot = states
        return [{'x': self.simulation_time,
                 'y': self.state_trajectory[i, :],
                 'name': self.state_names[i]
                 } for i in states_to_plot]

    def plot_output_trajectory(self):
        """Return output data dictionary compatible with Plotly Plots."""
        return [{'x': self.simulation_time,
                 'y': self.output_trajectory[i, :],
                 'name': self.output_names[i]
                 } for i in range(self.n_outputs)]
