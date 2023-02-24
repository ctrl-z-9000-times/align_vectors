"""
"""
import numpy as np

_epsilon = np.finfo(np.float32).eps

def _magnitude(vector):
    return np.sqrt(vector.dot(vector))

def align_vectors(a, b):
    """
    """
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == [3]
    assert b.shape == [3]
    a = a / _magnitude(a)
    b = b / _magnitude(b)
    c = np.dot(a, b) # Cosine of angle
    c1 = (1.0 + c)
    if abs(c1) < _epsilon:
        return -np.eye(3)
    v = np.cross(a, b)
    vx = np.array([ # Skew symmetric cross-product matrix
        [ 0,    -v[2], +v[1]],
        [+v[2],  0,    -v[0]],
        [-v[1], +v[0],  0]])
    return np.eye(3) + vx + np.matmul(vx, vx) * (1.0 / c1)
