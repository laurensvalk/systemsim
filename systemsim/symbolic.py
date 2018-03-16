"""Tools for symbolic derivation of equations of motion."""
from sympy import lambdify, diff, Matrix
import cloudpickle


class Equation:
    """An equation with variables and parameters."""

    def __init__(
            self,
            expression,   # Symbolic expression
            variables,    # List of symbolic vectors or scalars
            parameters):  # List of symbolic parameters
        """Init and create lambdified function: f(q1, ..., qn, parameters)."""
        self.expression = expression
        self.variables = variables
        self.parameters = parameters

        # Create the numeric lamdified function
        self.lambdified = lambdify([*self.variables, self.parameters], self.expression)

    def insert_parameters(self, parameter_values):
        """Create function f(q, ...) from f(q, ..., parameters) by inserting parameters."""
        # Obtain the numeric values in the same order as the symbols
        value_list = [parameter_values[symbol.name] for symbol in self.parameters]
        # Insert the list of numeric values
        self.function = lambda *variables: self.lambdified(*variables, value_list)

    def info(self):
        """Print the expression."""
        print("Expression: ", self.expression)
        print("Variables: ", self.variables)
        print("Parameters: ", self.parameters)


def save_object(filename, data):
    """Write a Python object to a file."""
    # Save data to disk
    with open(filename, 'wb') as file_descriptor:
        file_descriptor.write(cloudpickle.dumps(data))


def load_object(filename):
    """Load a Python object from a file."""
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
                (diff(M[k, j], q[i]) + diff(M[k, i], q[j]) - diff(M[i, j], q[k]))/2
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
    return Matrix(ckj)


def make_G(V, q):
    """Derive generalized gravity vector from potential energy expression."""
    # Shape to 1x1 matrix (scalar)
    V = Matrix([V])

    # Return jacobian as a column vector
    return V.jacobian(q).T
