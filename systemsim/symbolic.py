"""Tools for symbolic derivation of equations of motion."""
from .mechanical import SimpleMechanicalSystem

import sympy as sym
import cloudpickle


def save_lambda(filename, data):
    """Write a Python object (such as a lambda function) to a file."""
    # Save data to disk
    with open(filename, 'wb') as file_descriptor:
        file_descriptor.write(cloudpickle.dumps(data))


def load_lambda(filename):
    """Load a Python object (such as a lambda function) from a file."""
    # Load a dictionary of lambda equations
    with open(filename, 'rb') as f:
        return cloudpickle.load(f)


def make_C(M, q, q_dot):
    """Derive Coriolis matrix from mass matrix and generalized velocities."""
    # Dimension of the mechanical system
    n = M.shape[0]

    # Standard way of deriving C matrix elements from the mass matrix M
    # (See e.g. van der Schaft 2000)
    cijk = [
        [
            [
                (sym.diff(M[k, j], q[i]) + sym.diff(M[k, i], q[j]) - sym.diff(M[i, j], q[k]))/2
                for k in range(n)
            ] for j in range(n)
        ] for i in range(n)
    ]

    # Derive the elements of the C matrix
    ckj = [
        [
            sum([cijk[i][j][k]*q_dot[i] for i in range(n)]) for j in range(n)
        ] for k in range(n)
    ]

    # Return the nested list as a matrix
    return sym.Matrix(ckj)


def make_G(V, q):
    """Derive generalized gravity vector from potential energy expression."""
    # Shape to 1x1 matrix (scalar)
    V = sym.Matrix([V])

    # Return jacobian as a column vector
    return V.jacobian(q).T


class SymbolicMechanicalSystem(SimpleMechanicalSystem):
    """Mechanical system with equations of motion derived symbolically."""

    def __init__(
            self,
            modelfile,
            parameters,
            q_initial=None,
            q_dot_initial=None,
            exogenous_input_function=None):
        """Load previously derived model equations from a file."""
        # Load a dictionary of lambda equations from a file
        model = load_lambda(modelfile)

        # Store parameters for use in inherited classes like controllers
        parameters = parameters

        # Extract a fixed order of parameter names from the model file
        self.parameter_names = model['parameters']
        # Create corresponding ordered list of their values
        self.parameter_values = [parameters[name] for name in self.parameter_names]

        # Create the lambda functions, with the parameter values already
        # substituted
        M = lambda q: model['M'](q, *self.parameter_values)
        G = lambda q: model['G'](q, *self.parameter_values)
        F = lambda q: model['F'](q, *self.parameter_values)
        C = lambda q, q_dot: model['C'](q, q_dot, *self.parameter_values)
        Q = lambda q, q_dot: model['Q'](q, q_dot, *self.parameter_values)
        SimpleMechanicalSystem.__init__(self, M, F, C, G, Q,
                                        q_initial, q_dot_initial,
                                        exogenous_input_function)

        # If the model already contained a state feedback law, apply it
        if 'tau' in model:
            self.state_feedback = lambda x, time: model['tau'](
                self.get_coordinates(x)[0],  # q
                self.get_coordinates(x)[1],  # q_dot
                *self.parameter_values)
