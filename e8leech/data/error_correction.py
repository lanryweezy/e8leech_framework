import numpy as np
from e8leech.data.quantization import quantize_to_e8

def encode(data):
    """
    Encodes data using the E8 lattice code.

    Args:
        data: The data to be encoded. This should be a vector of 8 integers.

    Returns:
        The encoded data.
    """

    # This is a simple encoding scheme where the data is mapped to a lattice point.
    # A more sophisticated scheme would involve a more complex mapping.

    # We will use the basis vectors of the E8 lattice as the codewords.
    # The data is an integer vector that specifies the linear combination of the basis vectors.
    from e8leech.core.e8_lattice import get_e8_basis
    basis = get_e8_basis()

    return np.dot(data, basis)

def decode(received_data):
    """
    Decodes data using the E8 lattice code.

    Args:
        received_data: The received data, which may contain errors.

    Returns:
        The decoded data.
    """

    # Decoding is done by finding the closest lattice point to the received data.
    # This is equivalent to lattice quantization.
    quantized_data = quantize_to_e8(received_data)

    # Now we need to find the integer vector that corresponds to the quantized data.
    # This is done by solving the system of linear equations B*c = v, where B is the basis.
    from e8leech.core.e8_lattice import get_e8_basis
    basis = get_e8_basis()

    try:
        decoded_data = np.linalg.solve(basis.T, quantized_data)
        return np.round(decoded_data).astype(int)
    except np.linalg.LinAlgError:
        return None
