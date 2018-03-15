import numpy as np
from .core import System
import copy

class Interconnection(System):
    """Multiple systems connected through inputs and outputs"""
    def __init__(self, systems, system_connections,
                 input_connections=None,
                 exogenous_input_functions=None,
                 exogenous_output_functions=None):
        """Initialize network"""
        # Store systems with names
        self.systems = systems
        # Extract names for easy reference, always in the same order when using this list
        self.system_names = list(systems.keys())

        # Name error checking
        assert len(self.system_names) == len(set(self.system_names)), "System names must be unique."

        # Store interconnection information
        self.weights = system_connections
        # Extract list of edge tuples
        self.edges = list(system_connections.keys())
        # Create neighbor set for each system, given the edges
        self.neighbor_names = {i: [j for j in self.system_names if (j,i) in self.edges]
                                        for i in self.system_names}

        # Initial state of the whole network
        x_initial = self.__as_state_vector({i: s.x_initial for (i, s) in self.systems.items()})        
        n_states = sum([s.n_states for (i, s) in self.systems.items()])
        assert n_states == x_initial.shape[self.ROW]

        # Dimensions for converting between dictionaries and stacked state arrays
        x_dimensions = {i: s.n_states for (i, s) in self.systems.items()}
        self.x_cut_index = np.cumsum([x_dimensions[i] for i in self.system_names])[0:-1]

        # Store on which systems the external inputs act
        self.input_connections = input_connections if input_connections is not None else {}
        # For each of the signals, find out the dimension by checking the dimension of the systems on which they act
        input_dimensions = {signal: systems[acts_on[0]].n_inputs for (signal, acts_on) in self.input_connections.items()}
        # Sum of all input dimensions
        n_inputs = sum([dimension for (signal, dimension) in input_dimensions.items()])

        # Create a fixed alphabetical list of external inputs
        self.exogenous_input_signal_names = list(self.input_connections.keys())
        # When a vector of external inputs is supplied, this is how we can split them up in individual signals
        self.u_cut_index = np.cumsum([input_dimensions[signal] for signal in self.exogenous_input_signal_names])[0:-1]

        # Store the external input generators, if supplied
        self.exogenous_input_functions = exogenous_input_functions if exogenous_input_functions is not None else {}

        # If no input generator was supplied for some signal, then set it to zero
        for signal in self.exogenous_input_signal_names:
            if signal not in self.exogenous_input_functions:
                self.exogenous_input_functions[signal] = lambda time: np.zeros(input_dimensions[signal])

        # The resulting vector of external signals acting on the network
        exogenous_input_function = lambda time: \
            self.concatenate([self.exogenous_input_functions[signal](time) 
                              for signal in self.exogenous_input_signal_names])

        # TODO remove this temporary depencency; just for getting output size..abs
        self.exogenous_input = exogenous_input_function

        # Store output generators and how they are determined
        self.exogenous_output_functions = exogenous_output_functions if exogenous_output_functions is not None else {}
        self.exogenous_output_names = list(self.exogenous_output_functions.keys())

        # Number of outputs: Size of output evaluated at time 0
        n_outputs = len(self.output(x_initial, time=0))

        # Initialize System object
        System.__init__(self, n_states, n_inputs, n_outputs, x_initial, exogenous_input_function)



        # TODO: THIS STILL DOES NOT CORRECTLY TAKE exo in into account.

    def __as_state_per_system(self, network_state_vector):
        """Converts a stacked vector of states back into a dictionary of states per system"""
        # As intermediate step, cut state vector into list of state vectors
        state_list = np.split(network_state_vector, self.x_cut_index)
        # Return the dictionary of states
        return {system_name: state_list[listindex] for listindex, system_name in enumerate(self.system_names)}

    def __as_state_vector(self, state):
        """Converts a dictionary with states into a large stacked vector"""
        return self.concatenate([state[i] for i in self.system_names])
    
    def __as_input_per_signal(self, input_vector):
        """Converts a stacked vector of inputs back into a dictionary of input per signal"""
        # As intermediate step, cut input vector into list of inputs
        input_list = np.split(input_vector, self.u_cut_index)
        # Return the dictionary of inputs
        return {signal: input_list[listindex] for listindex, signal in enumerate(self.exogenous_input_signal_names)}    

    def __output_per_system(self, x, time):
        """Return the outputs of each system, given their states"""
        return {i: s.output(x[i], time) for (i, s) in self.systems.items()}

    def output(self, state_vector, time):
        """Vector output when viewing this whole network as one big system"""
        state_per_system = self.__as_state_per_system(state_vector)
        exogenous_input_vector = self.exogenous_input(time)
        input_per_signal = self.__as_input_per_signal(exogenous_input_vector)
        output_per_system = self.__output_per_system(state_per_system, time)
        exogenous_output_per_signal = {signal: function(output_per_system, input_per_signal) 
                                       for (signal, function) in self.exogenous_output_functions.items()}
        return self.concatenate([exogenous_output_per_signal[signal_name] for signal_name in self.exogenous_output_names])

    def distributed_law(self, x, y, time):
        """Return the distributed control input for each system, given their outputs computed from their states
        
        This is the default network interaction; other interactions may be specified by overriding this one
        
        """        
        return {i : sum([self.weights[(j,i)]*y[j] for j in self.neighbor_names[i] ])
                    for (i, s) in self.systems.items()}

    def equations_of_motion(self, state_vector, external_input_vector, time):
        """Evaluate equations of motion including internal state feedback and subject to exogenous input at current time"""
        # Convert network state array into states per system
        state_per_system = self.__as_state_per_system(state_vector)
        # Output for each system, given current state
        y_per_system = self.__output_per_system(state_per_system, time)
        # Evaluate distributed control law, given current states and outputs
        distributed_input_per_system = self.distributed_law(state_per_system, y_per_system, time)

        # Convert input array into the different exogenous inputs
        external_input_per_signal = self.__as_input_per_signal(external_input_vector)

        # For each signal, add the external input it to the system upon which its acts
        total_input_per_system = distributed_input_per_system
        for signal_name, systems_acted_on in self.input_connections.items():
            for system_acted_on in systems_acted_on:
                total_input_per_system[system_acted_on] = \
                    total_input_per_system[system_acted_on] +  external_input_per_signal[signal_name]

        # Obtain state change for the whole network, by taking the state change for each systen
        state_change_per_system = {i: s.state_change(state_per_system[i], total_input_per_system[i], time)
                                    for (i, s) in self.systems.items()}

        # Return state change in vector form
        return self.__as_state_vector(state_change_per_system)

    @staticmethod
    def undirected_edges(weighted_edges):
        """For every edge in the provided network, add an according reverse edge to make the network undirected"""

        # Initialize new network topology as supplied
        undirected_weighted_edges = copy.deepcopy(weighted_edges)

        # For each edge in the supplied network, add complementary edge
        for (tail, head), weight in weighted_edges.items():
            # Assert that the added edge isn't already in the network
            assert (head, tail) not in weighted_edges
            # Add the reversed edge
            undirected_weighted_edges[(head, tail)] = weight

        # Return the undirected network topology
        return undirected_weighted_edges

    def PostSimulationProcessing(self):
        """Store network simulation for all subsystems"""
        state_trajectory_per_system = self.__as_state_per_system(self.state_trajectory)
        for (i, s) in self.systems.items():
            s.state_trajectory = state_trajectory_per_system[i]
            s.simulation_time = self.simulation_time
            s.ComputeOutputTrajectory()

class DistributedSystem(Interconnection):
    """Interconnection of systems with a common type of interaction between neighboring agents"""

    def distributed_law(self, x, y, time):
        """ By default we evaluate the same control law for each agent, which may each have different neighbors"""        
        return {i : self.neighbor_interaction(i, x, y, time) for (i, s) in self.systems.items()}

    def neighbor_interaction(self, me, x, y, time):
        """Additional control input for the current agent, given the states and output of its neighbors
        
           By default, we use the weighted sum of the difference between the neighbor output and the output of the agent
        """
        return sum([self.weights[(j,me)]*(y[j]-y[me]) for j in self.neighbor_names[me]])