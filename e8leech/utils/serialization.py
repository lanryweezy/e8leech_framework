import json
import numpy as np

class E8LeechEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def to_json(obj):
    """
    Serializes an object to JSON.
    """
    return json.dumps(obj, cls=E8LeechEncoder)

def from_json(s):
    """
    Deserializes an object from JSON.
    """
    return json.loads(s)
