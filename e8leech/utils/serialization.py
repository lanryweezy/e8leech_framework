import json
import numpy as np
from e8leech.core.golay_code import E8Lattice, LeechLattice
from e8leech.crypto.lwe import LWE
from e8leech.ai.equivariant_gnn import EquivariantGNN

def to_json(obj):
    """
    Serializes an object to a JSON string.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (E8Lattice, LeechLattice, LWE, EquivariantGNN)):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def from_json(json_str):
    """
    Deserializes a JSON string to an object.
    """
    data = json.loads(json_str)
    # This is a simplified deserialization process.
    # A real implementation would need to handle different object types.
    return data

def save_model(model, filepath):
    """
    Saves a model to a file.
    """
    with open(filepath, 'w') as f:
        json.dump(model, f, default=to_json)

def load_model(filepath):
    """
    Loads a model from a file.
    """
    with open(filepath, 'r') as f:
        return from_json(f.read())
