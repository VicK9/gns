import torch


def multiply_matrices(A, B):
    """
    Input
        A - [B, N, 3, 3]
        B - [B, N, 3, 3]
    Output
        C - [B, N, 3, 3]
    """
    return torch.einsum("bijk,bikm->bijm", A, B)


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis_from_2_vectors(v1, v2):
    e1 = normalize_vector(v1, dim=-1)

    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)  # (N, L, 3)

    mat = torch.cat(
        [e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1
    )  # (N, L, 3, 3_index)
    return mat
