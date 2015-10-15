# http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
import numpy as np
import timeit

from math import cos, sin, sqrt
import numpy.random as nr

from scipy import weave

def rotation_matrix_weave(axis, theta, mat = None):
    if mat == None:
        mat = np.eye(3,3)

    support = "#include <math.h>"
    code = """
        double x = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        double a = cos(theta / 2.0);
        double b = -(axis[0] / x) * sin(theta / 2.0);
        double c = -(axis[1] / x) * sin(theta / 2.0);
        double d = -(axis[2] / x) * sin(theta / 2.0);

        mat[0] = a*a + b*b - c*c - d*d;
        mat[1] = 2 * (b*c - a*d);
        mat[2] = 2 * (b*d + a*c);

        mat[3*1 + 0] = 2*(b*c+a*d);
        mat[3*1 + 1] = a*a+c*c-b*b-d*d;
        mat[3*1 + 2] = 2*(c*d-a*b);

        mat[3*2 + 0] = 2*(b*d-a*c);
        mat[3*2 + 1] = 2*(c*d+a*b);
        mat[3*2 + 2] = a*a+d*d-b*b-c*c;
    """

    weave.inline(code, ['axis', 'theta', 'mat'], support_code = support, libraries = ['m'])

    return mat







import numpy as np
import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

v = [3, 5, 0]
axis = [4, 4, 1]
theta = 1.2 

print(np.dot(rotation_matrix(axis,theta), v)) 
# [ 2.74911638  4.77180932  1.91629719]



>>> import timeit
>>> 
>>> setup = """
... import numpy as np
... import numpy.random as nr
... 
... from rotation_matrix_test import rotation_matrix_weave
... from rotation_matrix_test import rotation_matrix_numpy
... 
... mat1 = np.eye(3,3)
... theta = nr.random()
... axis = nr.random(3)
... """
>>> 
>>> timeit.repeat("rotation_matrix_weave(axis, theta, mat1)", setup=setup, number=100000)
[0.36641597747802734, 0.34883809089660645, 0.3459300994873047]
>>> timeit.repeat("rotation_matrix_numpy(axis, theta)", setup=setup, number=100000)
[7.180983066558838, 7.172032117843628, 7.180462837219238]

