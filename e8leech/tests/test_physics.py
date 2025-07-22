import unittest
import numpy as np
from e8leech.physics.string_compact import compactify_on_torus
from e8leech.physics.quantum_fields import QuantumField
from e8leech.core.e8_lattice import get_e8_basis

class TestPhysics(unittest.TestCase):

    def test_compactify_on_torus(self):
        """
        Tests the string compactification function.
        """
        print("Testing compactify_on_torus...")
        basis = get_e8_basis()
        compactified_basis = compactify_on_torus(basis, 4)
        self.assertEqual(compactified_basis.shape, (8, 4))
        print("...done")

    def test_quantum_field(self):
        """
        Tests the quantum field operators.
        """
        print("Testing quantum_field...")
        basis = get_e8_basis()
        qf = QuantumField(basis)
        self.assertEqual(qf.field.shape, (8,))

        op = qf.get_operator(0)
        self.assertEqual(op.shape, (8, 8))

        creation_op = qf.get_creation_operator(0)
        self.assertEqual(creation_op.shape, (8, 8))

        annihilation_op = qf.get_annihilation_operator(0)
        self.assertEqual(annihilation_op.shape, (8, 8))
        print("...done")

if __name__ == '__main__':
    unittest.main()
