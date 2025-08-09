import numpy as np
from e8leech.core.cvp import kannan_fincke_pohst

def test_kannan_fincke_pohst_2d():
    basis = np.array([[1, 0], [0, 2]])
    v = np.array([0.6, 0.6])
    p = kannan_fincke_pohst(v, basis)
    assert np.allclose(p, np.array([1, 0]))

def test_kannan_fincke_pohst_3d():
    basis = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    v = np.array([0.6, 0.6, 0.6])
    p = kannan_fincke_pohst(v, basis)
    assert np.allclose(p, np.array([1, 0, 0]))
