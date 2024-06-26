import torch

@torch.jit.script
def one_hot(x: torch.Tensor, num_classes: int = -1, dtype: torch.dtype = torch.float32):
    """A one-hot encoding function faster than the PyTorch one working with torch.int32 and returning a float Tensor"""
    if num_classes < 0:
        num_classes = x.max() + 1
    res = torch.zeros(x.shape[0], x.shape[1], num_classes, device=x.device, dtype=dtype)
    tmp = torch.meshgrid(
        torch.arange(x.shape[0], device=x.device),
        torch.arange(x.shape[1], device=x.device),
        indexing="ij",
    )
    index = (tmp[0], tmp[1], x)
    values = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=dtype)
    res.index_put_(index, values)
    return res

def get_ortho(mat: torch.Tensor):
    """Orthonormalize the column vectors of a matrix.

    Parameters
    ----------
    mat : torch.Tensor
        Matrix to orthonormalized. (a, b)

    Returns
    -------
    torch.Tensor
        Orthonormalized matrix. (a, b)
    """
    res = mat.clone()
    n, d = mat.shape

    u0 = mat[:, 0] / mat[:, 0].norm()
    res[:, 0] = u0
    for i in range(1, d):
        ui = mat[:, i]
        for j in range(i):
            ui -= (ui @ res[:, j]) * res[:, j]
        res[:, i] = ui / ui.norm()
    return res


def compute_U(
    M: torch.Tensor,
    weights: torch.Tensor,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute the first right eigenvector of the dataset.

    Parameters
    ----------
    M : torch.Tensor
        Dataset. (n_sample, n_visible)
    weights : torch.Tensor
        Weights of each sample (n_sample,)
    intrinsic_dimension : int
        Number of principal axis to compute.
    device : torch.device
        Device.
    dtype : torch.dtype
        Dtype

    Returns
    -------
    torch.Tensor
        Right eigenvectors. (n_dim, n_visible)
    """
    M = M * torch.sqrt(weights)
    num_samples, num_visibles = M.shape
    max_iter = 100
    err_threshold = 1e-15
    curr_v = (
        torch.rand(num_samples, d, device=device, dtype=dtype) * 2 - 1
    )
    u = torch.rand(num_visibles, d, device=device, dtype=dtype)
    curr_id_mat = (
        torch.rand(d, d, device=device, dtype=dtype)
        * 2
        - 1
    )
    for n in range(max_iter):
        v = curr_v.clone()
        curr_v = M @ u
        if num_samples < num_visibles:
            id_mat = (v.T @ curr_v) / num_samples
            curr_v = get_ortho(curr_v)
        curr_u = M.T @ curr_v
        if num_visibles <= num_samples:
            id_mat = (u.T @ curr_u) / num_samples
            curr_u = get_ortho(curr_u)
        u = curr_u.clone()
        if (id_mat - curr_id_mat).norm() < err_threshold:
            break
        curr_id_mat = id_mat.clone()
    u = get_ortho(u)
    return u