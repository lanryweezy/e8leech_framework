from e8leech.core.golay_code import GolayCode

class ErrorCorrection:
    """
    A class for error correction using the Golay code.
    """

    def __init__(self):
        self.golay = GolayCode()

    def encode(self, data):
        """
        Encodes data using the Golay code.
        The data must be a list or array of 12-bit messages.
        """
        encoded_data = []
        for message in data:
            encoded_data.append(self.golay.encode(message))
        return encoded_data

    def decode(self, data):
        """
        Decodes data using the Golay code.
        The data must be a list or array of 24-bit codewords.
        """
        decoded_data = []
        for codeword in data:
            decoded_data.append(self.golay.decode(codeword))
        return decoded_data
