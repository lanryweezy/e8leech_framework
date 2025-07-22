import numpy as np

class LSH:
    """
    Locality-Sensitive Hashing for approximate nearest neighbor search.
    """
    def __init__(self, num_hashes, input_dim, bucket_width):
        """
        Initializes the LSH model.

        Args:
            num_hashes: The number of hash functions to use.
            input_dim: The dimensionality of the input vectors.
            bucket_width: The width of the hash buckets.
        """
        self.num_hashes = num_hashes
        self.input_dim = input_dim
        self.bucket_width = bucket_width
        self.projections = np.random.randn(self.num_hashes, self.input_dim)
        self.hash_table = {}

    def hash(self, v):
        """
        Hashes a vector.
        """
        hashes = np.floor(np.dot(self.projections, v) / self.bucket_width)
        return tuple(hashes.astype(int))

    def index(self, data):
        """
        Indexes a set of vectors.
        """
        for i, v in enumerate(data):
            h = self.hash(v)
            if h not in self.hash_table:
                self.hash_table[h] = []
            self.hash_table[h].append(i)

    def query(self, vectors, k=1):
        """
        Queries the LSH model for the k nearest neighbors of a batch of vectors.
        """
        results = []
        for v in vectors:
            h = self.hash(v)
            if h in self.hash_table:
                results.append(self.hash_table[h][:k])
            else:
                results.append([])
        return results
