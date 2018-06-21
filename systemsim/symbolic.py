"""Tools for symbolic derivation of equations of motion."""
from sympy import lambdify, diff, Matrix


class Equation:
    """An equation with variables and parameters."""

    def __init__(
            self,
            expression,   # Symbolic expression
            variables,    # List of symbolic vectors or scalars
            parameters,   # List of symbolic parameters
            shape_as='auto'):   # Choose output as array, ndarray, or auto:
                                # array if it is one row or one column, ndarray otherwise
        """Init and create lambdified function: f(q1, ..., qn, parameters)."""
        self.expression = expression
        self.variables = variables
        self.parameters = parameters

        # Convert to matrix type if necessary.
        # This ensures the output has an explicit
        # shape.
        if type(self.expression) is not Matrix:
            self.expression = Matrix([self.expression])

        # Check that a compatible shape argument is supplied
        assert shape_as in ['auto', 'array', 'ndarray']

        # Automatically select array if data is scalar or vector
        if shape_as == 'auto':
            if 1 in self.expression.shape:
                shape_as = 'array'
            else:
                shape_as = 'ndarray'

        # Check whether we want to reshape vectors, and whether
        # we are really dealing with a vector
        if shape_as == 'array':
            assert 1 in self.expression.shape
            self.reshape_as_vector = True
            self.number_of_elements = len(self.expression)
        else:
            self.reshape_as_vector = False

        # Lambified expression in ndarray form (privately accessible only)
        self.__lambdified_matrix = lambdify([*self.variables, self.parameters], self.expression)

        # Create the publicly accessible lambdified expression of the form:
        # f(q1, ..., qn, parameters)
        if self.reshape_as_vector:
            # Apply the reshaping step if requested
            self.lambdified = lambda *varargs: self.__lambdified_matrix(*varargs).reshape(self.number_of_elements,)
        else:
            # Otherwise, it is just what we already computed
            self.lambdified = self.__lambdified_matrix

    def insert_parameters(self, parameter_values):
        """Create function f(q, ...) from f(q, ..., parameters) by inserting parameters."""
        # Obtain the numeric values in the same order as the symbols
        value_list = [parameter_values[symbol.name] for symbol in self.parameters]

        if self.reshape_as_vector:
            # Insert the list of numeric values while ensuring ouput is an array
            self.function = lambda *variables: \
                self.__lambdified_matrix(*variables, value_list).reshape(self.number_of_elements,)
        else:
            # Insert the list of numeric values while leaving matrix dimensions unchanged
            self.function = lambda *variables: \
                self.__lambdified_matrix(*variables, value_list)

    def info(self):
        """Print the expression."""
        print("Expression: ", self.expression)
        print("Variables: ", self.variables)
        print("Parameters: ", self.parameters)
        print("Expression is a matrix: ", self.is_matrix)


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
