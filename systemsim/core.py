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

    def total_input(self, external_control_input, exogenous_input, state_feedback, state, time):
        """Compute total input to system.

        By default, this is the sum of the state feedback, the exogenous input,
        and optionally an input from the network.
        """
        return external_control_input + exogenous_input + state_feedback

    def state_change(self, state, external_control_input, time):
        """Evaluate equations of motion subject to exogenous input."""
        exogenous_input = self.exogenous_input(time)
        state_feedback = self.state_feedback(state, time)
        total_input = self.total_input(external_control_input, exogenous_input, state_feedback, state, time)
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
        self.x_end = self.state_trajectory[:, -1]
        self.compute_output_trajectory()
        # Further processing if needed
        self.post_simulation_processing()

    def compute_output_trajectory(self):
        """Compute the outputs for the previously run simulation."""
        # Calculate and concatenate output at every timestep
        self.output_trajectory = np.stack([
            self.output(self.state_trajectory[:, k], time)
            for (k, time) in enumerate(self.simulation_time)
        ], axis=self.COL)

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

    def make_animation_data(self, time_scale=1, frames_per_second=25):
        """Interpolate simulation results and kinematic map."""
        # Time per frame
        self.animation_time_per_frame = 1/frames_per_second

        # Create time vector of display instants
        time_end = self.simulation_time[-1]
        self.animation_time = np.arange(0, time_end/time_scale, self.animation_time_per_frame)

        # Real time samples corresponding to display time
        self.animation_time_real = self.animation_time*time_scale

        # Interpolate state data to match the display time vector
        self.animation_states = np.stack(
            [
                np.interp(self.animation_time_real, self.simulation_time, self.state_trajectory[i, :])
                for i in range(self.n_states)
            ]
        )

        # Get X and Y coordinates of animated line for each animation time instance
        self.animation_x = []
        self.animation_y = []
        for (k, time) in enumerate(self.animation_time):
            state = self.animation_states[:, k]
            x, y = self.get_animation_kinematics(state)
            self.animation_x.append(x)
            self.animation_y.append(y)

    def get_animation_frames(self):
        """Generate Plotly animation frames."""
        # Plotly frame data format of the kinematics at every simulation time instant
        framedata = [
            {
                'data': [
                    {'x': self.animation_x[i], 'y': self.animation_y[i]}
                ]
            } for i, t in enumerate(self.animation_time)
        ]
        # Take first frame as initial plot (before clicking start)
        firstframe = framedata[0]['data']

        # Legends for frame data
        firstframe[0]['name'] = 'UAV0'

        return firstframe, framedata

    def get_animation_figure(self, xlim, ylim, title):
        """Return Plotly Animation Figure for inline plot."""
        # Get kinematic animation as frama data format
        firstframe, framedata = self.get_animation_frames()

        # Estimated plot time overhead (miliseconds)
        overhead_ms = 7

        # Create the figure object
        animationfigure = {
            'data': firstframe,
            'layout': {
                'xaxis': {'range': xlim, 'autorange': False},
                'yaxis': {'range': ylim, 'autorange': False, 'scaleanchor': 'x'},
                'title': title,
                'updatemenus': [{
                    'type': 'buttons',
                    'buttons': [
                        {
                            'label': 'Start', 'method': 'animate',
                            'args': [
                                framedata, {'frame': {'duration': self.animation_time_per_frame*1000 - overhead_ms, 'redraw': False}}
                            ]
                        },
                        {
                            'label': 'Stop', 'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}
                            }]
                        }
                    ]
                }]
            }
        }
        return animationfigure

    def get_animation_kinematics(self, state):
        """Create line of 2D or 3D points to plot for given state."""
        # Example points through which we will draw a line
        # These could be generated as a function of the instantaneous state
        left = np.array([0, 0])
        middle = np.array([0, 0])
        right = np.array([0, 0])

        # Points to draw in x and y
        points = np.stack([left, middle, right], axis=self.COL)
        xpoints = points[0, :].tolist()
        ypoints = points[1, :].tolist()

        # Return the points
        return xpoints, ypoints
