"""Classes to create new systems out of two or more subsystems."""

import numpy as np
from .core import System


class Collection(System):
    """A collection of disconnected systems operating in the same environment.

    This is in essence a simulation of several open-loop systems operating simultaneously.
    While not very useful in this form, it forms the basis for several classes that
    interconnect the systems to produce a desired feedback system or network interaction.
    """

    def __init__(
            self,
            systems,
            input_signals=[],
            output_signals=[]):
        """Initialize network."""
        # Store systems with names
        self.systems = systems
        # As well as an index lookup table
        self.index = {
            system: self.systems.index(system) for system in systems
        }

        # Initial state of the whole network
        x_dimensions = [s.n_states for s in self.systems]
        n_states = sum(x_dimensions)

        # Initial state of the whole network as a concatation of states
        x_initial = self.concatenate([s.x_initial for s in self.systems])
        # To do the reverse (de-concatenate), we need to remember where to split
        self.x_cut_index = np.cumsum(x_dimensions)[0:-1]

        # Store the input and output signal generators
        self.input_signals = input_signals
        self.output_signals = output_signals

        # Each signal has been specified to act onto certain plants
        # Here we verify that they are of compatible size, and set
        # some signals to zero if they are not explicitly given.
        input_dimensions = []
        input_generators = []
        self.plants_acted_on_per_signal = []
        for (plants_acted_on, generator) in self.input_signals:
            # Keep a list on which plant each signal acts
            self.plants_acted_on_per_signal.append(plants_acted_on)
            # Use (arbitrarily) the first plant to check the input size
            plant0 = plants_acted_on[0]
            # Check that the signal acts on compatible plants
            assert all(plant.n_inputs == plant0.n_inputs for plant in plants_acted_on)
            # Append the result to the list of input dimensions
            input_dimensions.append(plant0.n_inputs)
            # Make explicit any signal that was not specified
            if generator is None:
                generator = lambda time: plant0.zero_input
            input_generators.append(generator)
        # Also create a stacked input generator of all inputs
        exogenous_input_function = lambda time: \
            self.concatenate([signal(time) for signal in input_generators])

        # When a vector of external inputs is supplied, this is where we can split it
        # up into the individual signals
        n_inputs = sum(input_dimensions)
        self.u_cut_index = np.cumsum(input_dimensions)[0:-1]

        # Number of outputs: Size of output evaluated at time 0
        n_outputs = len(self.output(x_initial, time=0))

        # Initialize System object
        System.__init__(self, n_states, n_inputs, n_outputs, x_initial, exogenous_input_function)

    def __as_state_per_system(self, network_state_vector):
        """Cut state vector into list of state vectors."""
        return np.split(network_state_vector, self.x_cut_index)

    def __as_input_per_signal(self, input_vector):
        """Cut input vector of list of inputs."""
        return np.split(input_vector, self.u_cut_index)

    def __output_per_system(self, state_per_system, time):
        """Return the outputs of each system, given their states."""
        return [s.output(state_per_system[i], time) for (i, s) in enumerate(self.systems)]

    def output(self, state_vector, time):
        """Vector output when viewing this whole network as one big system."""
        # Work out the output of each plant separately
        state_per_system = self.__as_state_per_system(state_vector)
        output_per_system = self.__output_per_system(state_per_system, time)
        # Each signal in the total output is a linear combination of plant outputs
        # For each signal we make this linear sum as follows
        output_per_signal = []
        for weighted_plant in self.output_signals:
            signal = sum([
                weight*output_per_system[self.index[plant]] for (plant, weight) in weighted_plant.items()
            ])
            output_per_signal.append(signal)
        # We return the result as a vector
        return self.concatenate(output_per_signal)

    def distributed_law(self, x, y, time):
        """Return the distributed control input for each system.

        There is no interaction by default; other interactions may be
        specified by overriding this one in the inherited class.
        """
        return [s.zero_input for s in self.systems]

    def equations_of_motion(self, state_vector, external_input_vector, time):
        """Evaluate equations of motion.

        Including internal state feedback and subject to exogenous input at
        current time.
        """
        # Convert network state array into states per system
        state_per_system = self.__as_state_per_system(state_vector)
        # Output for each system, given current state
        output_per_system = self.__output_per_system(state_per_system, time)
        # Evaluate distributed control law, given current states and outputs
        distributed_input_per_system = self.distributed_law(state_per_system, output_per_system, time)

        # Convert input array into the different exogenous inputs
        external_input_per_signal = self.__as_input_per_signal(external_input_vector)

        # For each system, compute the distributed input from the connection/network, if present
        total_input_per_system = distributed_input_per_system

        # For each signal, add the external input to the systems upon which its acts
        for (signal_index, (plants_acted_on, generator)) in enumerate(self.input_signals):
            for plant_acted_on in plants_acted_on:
                plant_index = self.index[plant_acted_on]
                total_input_per_system[plant_index] += external_input_per_signal[signal_index]

        # Obtain state change for the whole network, by taking the state
        # change for each systen
        state_change_per_system = [
            s.state_change(
                state_per_system[i],
                total_input_per_system[i],
                time) for (i, s) in enumerate(self.systems)
        ]

        # Return state change in vector form
        return self.concatenate(state_change_per_system)

    def post_simulation_processing(self):
        """Store network simulation for all subsystems."""
        state_trajectory_per_system = self.__as_state_per_system(self.state_trajectory)
        for (i, s) in enumerate(self.systems):
            s.state_trajectory = state_trajectory_per_system[i]
            s.simulation_time = self.simulation_time
            s.compute_output_trajectory()


class Interconnection(Collection):
    """A collection of systems interacting through inputs and outputs."""

    def __init__(
            self,
            systems,
            connections,
            input_functions=[],
            output_functions=[]):
        """Initialize network."""
        # Initialize System object
        Collection.__init__(self, systems, input_functions, output_functions)

        # Store system connections
        self.connections = connections
        # Create neighbor set for each system, given the edges
        self.neighbors = {
            i: [j for j in self.systems if (j, i) in self.connections] for i in self.systems
        }

    def distributed_law(self, x, y, time):
        """Return the distributed control input for each system due to its input/output connections."""
        return [
            sum([
                self.connections[(j, i)]*y[self.index[j]] for j in self.neighbors[i]
            ]) for i in self.systems
        ]


class DistributedSystem(Collection):
    """A collection of systems interacting through information exchange."""

    def __init__(
            self,
            systems,
            weighted_edges,
            leaders={},
            input_functions=[],
            output_functions=[]):
        """Initialize network."""
        # Initialize System object
        Collection.__init__(self, systems, input_functions, output_functions)
        # Store list of edges
        self.weighted_edges = weighted_edges
        # Store interconnection information
        self.weights = {
            edge: weight for (edge, (weight, relative_distance)) in weighted_edges.items()
        }
        # Store interconnection information
        self.relative_distances = {
            edge: relative_distance for (edge, (weight, relative_distance)) in weighted_edges.items()
        }
        # Create neighbor set for each system, given the edges
        self.neighbors = {
            i: [j for j in self.systems if (j, i) in self.weighted_edges] for i in self.systems
        }
        # Store leader settings
        self.leaders = leaders

    def distributed_law(self, x, y, time):
        """Evaluate the same control law for each agent."""
        return [
            sum([
                self.weights[(j, i)]*(y[self.index[j]]-y[self.index[i]]) for j in self.neighbors[i]
            ]) for i in self.systems
        ]

    @staticmethod
    def undirected_edges(weighted_edges):
        """Turn every edge into an undirected one."""
        # Initialize new network topology as empty
        undirected_weighted_edges = {}

        # For each edge in the supplied network, add complementary edge
        for (tail, head), (weight, relative_distance) in weighted_edges.items():
            # Assert that the undirected edge isn't already in the network
            assert not ((head, tail) in weighted_edges and (tail, head) in weighted_edges)
            # Add the original and reversed edge in new topology
            undirected_weighted_edges[(tail, head)] = (weight, relative_distance)
            undirected_weighted_edges[(head, tail)] = (weight, -relative_distance)

        # Return the undirected network topology
        return undirected_weighted_edges


class DistributedMechanicalSystem(DistributedSystem):
    """Interconnection of systems with a common type of interaction."""

    def coordinated_positions(self, q):
        """Compute the generalized positions that are communicated over the network.

        By default, this is the full position coordinate.
        """
        return q

    def distributed_law(self, x, y, time):
        """Evaluate the same control law for each agent."""
        # Coordinates for all agents
        q = [s.get_coordinates(x[i])[0] for (i, s) in enumerate(self.systems)]

        # Exchangable coordinates
        z = self.coordinated_positions(q)

        # For each system, give distributed weighted inputs, given its neighbors
        u = [
            sum([
                self.weights[(i, j)]*(z[self.index[i]]-z[self.index[j]]+self.relative_distances[(i, j)]) for j in self.neighbors[i]
            ]) for i in self.systems
        ]

        # Add leader forces
        for (leader, (gain, target)) in self.leaders.items():
            index = self.index[leader]
            u[index] = u[index] + gain*(z[index]-target)

        return u
