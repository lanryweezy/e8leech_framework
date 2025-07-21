import numpy as np

class QuantumField:
    """
    A class to represent a quantum field.
    """

    def __init__(self, name, dimension, spin):
        self.name = name
        self.dimension = dimension
        self.spin = spin
        self.field_values = np.zeros(dimension)

    def __repr__(self):
        return f"QuantumField({self.name}, dim={self.dimension}, spin={self.spin})"

class FieldOperator:
    """
    A class to represent a quantum field operator.
    """

    def __init__(self, name, action):
        self.name = name
        self.action = action

    def apply(self, fields):
        """
        Applies the operator to a quantum field or a batch of fields.
        """
        if isinstance(fields, list):
            return [self.action(field) for field in fields]
        else:
            return self.action(fields)

# Example operators
def creation_operator(field):
    """
    A simple creation operator that excites the field.
    """
    field.field_values += np.random.randn(field.dimension)
    return field

def annihilation_operator(field):
    """
    A simple annihilation operator that de-excites the field.
    """
    field.field_values -= np.random.randn(field.dimension)
    return field
