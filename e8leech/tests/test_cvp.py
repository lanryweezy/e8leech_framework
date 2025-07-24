import numpy as np
from e8leech.core.cvp import kannan_fincke_pohst

def test_kannan_fincke_pohst():
    basis = np.array([[1, 0], [0, 2]])
    v = np.array([0.6, 0.6])
    p = kannan_fincke_pohst(v, basis)
    assert np.allclose(p, np.array([1, 0]))
