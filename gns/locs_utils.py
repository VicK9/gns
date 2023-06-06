import numpy as np
import torch
import torch.nn.functional as F


# Note: This is the model that is used in the paper
"""
General Notes:
dicts are used to store the parameters for each of the modules
params.get('key', default) is used to get the value of a key in the dict
if the key is not present, the default value is returned

"""

"""
Geometric primitives, rotations, cartesian to spherical
"""


def rotation_matrix(ndim, theta, phi=None, psi=None, /):
    """
    theta, phi, psi: yaw, pitch, roll

    NOTE: We assume that each angle is has the shape [dims] x 1
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    if ndim == 2:
        R = torch.stack(
            [
                torch.cat([cos_theta, -sin_theta], -1),
                torch.cat([sin_theta, cos_theta], -1),
            ],
            -2,
        )
        return R
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    R = torch.stack(
        [
            torch.cat([cos_phi * cos_theta, -sin_theta, sin_phi * cos_theta], -1),
            torch.cat([cos_phi * sin_theta, cos_theta, sin_phi * sin_theta], -1),
            torch.cat([-sin_phi, torch.zeros_like(cos_theta), cos_phi], -1),
        ],
        -2,
    )
    return R


def cart_to_n_spherical(x, symmetric_theta=False):
    """Transform Cartesian to n-Spherical Coordinates

    NOTE: Not tested thoroughly for n > 3

    Math convention, theta: azimuth angle, angle in x-y plane

    x: torch.Tensor, [dims] x D
    return rho, theta, phi
    """
    ndim = x.size(-1)

    rho = torch.norm(x, p=2, dim=-1, keepdim=True)

    theta = torch.atan2(x[..., [1]], x[..., [0]])
    if not symmetric_theta:
        theta = theta + (theta < 0).type_as(theta) * (2 * np.pi)

    if ndim == 2:
        return rho, theta

    cum_sqr = (
        rho
        if ndim == 3
        else torch.sqrt(torch.cumsum(torch.flip(x**2, [-1]), dim=-1))[..., 2:]
    )
    EPS = 1e-7
    phi = torch.acos(torch.clamp(x[..., 2:] / (cum_sqr + EPS), min=-1.0, max=1.0))

    return rho, theta, phi


def velocity_to_rotation_matrix(vel):
    num_dims = vel.size(-1)
    orientations = cart_to_n_spherical(vel)[1:]
    R = rotation_matrix(num_dims, *orientations)
    return R


def gram_schmidt(vel, acc):
    """Gram-Schmidt orthogonalization"""
    # normalize
    e1 = F.normalize(vel, dim=-1)
    # orthogonalize
    u2 = acc - torch.sum(e1 * acc, dim=-1, keepdim=True) * e1
    # normalize
    e2 = F.normalize(u2, dim=-1)
    # cross product
    e3 = torch.cross(e1, e2)

    frame1 = torch.stack([e1, e2, e3], dim=-1)
    return frame1


def rotation_matrices_to_quaternions(rotations: torch.Tensor) -> torch.Tensor:
    # Ensure input tensor has the correct shape
    assert (rotations.dim() == 4 or rotations.dim() == 3) and rotations.shape[-2:] == (
        3,
        3,
    ), f"Expected tensor of shape [...,3,3], got {rotations.shape}"

    # Extract the rotation matrix components
    r11, r12, r13 = rotations[..., 0, 0], rotations[..., 0, 1], rotations[..., 0, 2]
    r21, r22, r23 = rotations[..., 1, 0], rotations[..., 1, 1], rotations[..., 1, 2]
    r31, r32, r33 = rotations[..., 2, 0], rotations[..., 2, 1], rotations[..., 2, 2]

    # Compute the quaternion components
    qw = torch.sqrt(torch.clamp(1.0 + r11 + r22 + r33, min=1e-8)) / 2.0
    qx = torch.sqrt(torch.clamp(1.0 + r11 - r22 - r33, min=1e-8)) / 2.0
    qy = torch.sqrt(torch.clamp(1.0 - r11 + r22 - r33, min=1e-8)) / 2.0
    qz = torch.sqrt(torch.clamp(1.0 - r11 - r22 + r33, min=1e-8)) / 2.0

    # Determine the signs of the quaternion components
    qx = torch.where(r32 - r23 < 0, -qx, qx)
    qy = torch.where(r13 - r31 < 0, -qy, qy)
    qz = torch.where(r21 - r12 < 0, -qz, qz)

    # Combine the quaternion components into a tensor of shape [B,N,4]
    quaternions = torch.stack((qw, qx, qy, qz), dim=-1)

    return quaternions


def rotation_matrix_to_euler(R, num_dims, normalize=True):
    """Convert rotation matrix to euler angles

    In 3 dimensions, we follow the ZYX convention
    NOTE: Use torch.clamp to avoid numerical errors everything has to be in [-1, 1]

    """
    if num_dims == 2:
        euler = torch.atan2(R[..., 1, [0]], R[..., 0, [0]])
    else:
        euler = torch.stack(
            [
                torch.atan2(R[..., 1, 0], R[..., 0, 0]),
                torch.asin(torch.clamp(-R[..., 2, 0], min=-1, max=1)),
                torch.atan2(R[..., 2, 1], R[..., 2, 2]),
            ],
            -1,
        )

    if normalize:
        euler = euler / np.pi
    return euler


def rotate(x, R):
    return torch.einsum("...ij,...j->...i", R, x)
