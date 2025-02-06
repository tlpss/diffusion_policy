"""
utils to work with 6D representation of rotation that is continuous, constructed by (x,y) of the rotation matrix.
https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Zhou_On_the_Continuity_CVPR_2019_supplemental.pdf

"""
from scipy.spatial.transform import Rotation as R
import numpy as np

def convert_rotvec_to_6D_representation(rotvec):
    """Converts a rotation vector to a 6D representation.

    Args:
        rotvec (np.array): Rotation vector.

    Returns:
        np.array: 6D representation of the rotation, using the first two axis of the rotation matrix (x,y)
    """
    rotmat = R.from_rotvec(rotvec).as_matrix()
    x,y,z = rotmat[:,0], rotmat[:,1], rotmat[:,2]
    return np.concatenate([x,y])

def convert_6D_rotation_to_rotvec(rot6D):
    """Converts a 6D representation of a rotation to a rotation vector.

    Args:
        rot6D (np.array): 6D representation of the rotation.

    Returns:
        np.array: Rotation vector.
    """

    rotmat = convert_6D_rotation_to_rotation_matrix(rot6D)
    return R.from_matrix(rotmat).as_rotvec()

def convert_6D_rotation_to_rotation_matrix(rot6D):
    """Converts a 6D representation of a rotation to a rotation matrix using GS-orthonormalization.

    Args:
        rot6D (np.array): 6D representation of the rotation.

    Returns:
        np.array: Rotation matrix.
    """
    x = rot6D[:3] / np.linalg.norm(rot6D[:3])
    y = rot6D[3:] - np.dot(rot6D[3:], x) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


if __name__ == "__main__":
    # test by generating 1000 random rotation vectors
    # and converting them to 6D representation and back

    for _ in range(1000):
        rotvec = np.random.rand(3)
        rot6D = convert_rotvec_to_6D_representation(rotvec)
        rotvec_recovered = convert_6D_rotation_to_rotvec(rot6D)
        assert np.allclose(rotvec, rotvec_recovered)
        rot6D_recovered = convert_rotvec_to_6D_representation(rotvec_recovered)
        assert np.allclose(rot6D, rot6D_recovered)