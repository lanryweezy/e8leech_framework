import numpy as np

def kannan_fincke_pohst_recursive(v, basis, radius):
    """
    Recursive step of the Kannan-Fincke-Pohst algorithm.
    """
    if basis.shape[0] == 1:
        # Base case: the lattice is one-dimensional
        b = basis[0]
        c = np.round(np.dot(v, b) / np.dot(b, b))
        p = c * b
        dist = np.linalg.norm(v - p)
        if dist < radius:
            return p, dist
        else:
            return None, np.inf

    # Recursive step
    b_n = basis[-1]
    basis_proj = basis[:-1] - np.outer(np.dot(basis[:-1], b_n) / np.dot(b_n, b_n), b_n)
    v_proj = v - np.dot(v, b_n) / np.dot(b_n, b_n) * b_n

    best_p = None
    min_dist = radius

    c_n = 0
    while True:
        p_proj, dist_proj = kannan_fincke_pohst_recursive(v_proj, basis_proj, min_dist)
        if p_proj is not None:
            p = p_proj + c_n * b_n
            dist = np.linalg.norm(v - p)
            if dist < min_dist:
                min_dist = dist
                best_p = p

        if c_n == 0:
            c_n = 1
        elif c_n > 0:
            c_n = -c_n
        else:
            c_n = -c_n + 1

        if np.abs(c_n * np.linalg.norm(b_n)) > min_dist:
            break

    return best_p, min_dist

def kannan_fincke_pohst(v, basis):
    """
    Kannan-Fincke-Pohst algorithm for solving the closest vector problem.
    """
    return kannan_fincke_pohst_recursive(v, basis, np.inf)[0]
