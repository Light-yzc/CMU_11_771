from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.
    query = query.transpose(0, 1)
    key = key.transpose(0, 1)

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    theta_i = torch.arange((head_dim // 2), device=device)
    theta_i = theta ** ((-2 * theta_i) / head_dim)
    theta_j_i = torch.arange(max_seq_len, device=device,dtype=theta_i.dtype).unsqueeze(1) @ theta_i.unsqueeze(0)  #max_len, dim / 2
    cos_t_i = torch.cos(theta_j_i).unsqueeze(1) # max_len, 1, dim/2
    cos_t_i = cos_t_i.unsqueeze(1)[:seqlen,:,:,:] # len_seq ，1， 1， dim/2
    sin_t_i = torch.sin(theta_j_i).unsqueeze(1)
    sin_t_i = sin_t_i.unsqueeze(1)[:seqlen,:,:,:]
    _query_real = query_imag * cos_t_i - query_real * sin_t_i
    _query_image = query_real * sin_t_i + query_imag * cos_t_i
    _key_real = key_imag * cos_t_i - key_real * sin_t_i  #len_seq, bn, n_head, head_dim / 2
    _key_image = key_real * sin_t_i + key_imag * cos_t_i
    # x_Rota_even = x_even * cos - x_odd * sin
    query_out = torch.stack([_query_real, _query_image], dim=-1).flatten(start_dim=-2).transpose(1, 0)
    key_out = torch.stack([_key_real, _key_image], dim=-1).flatten(start_dim=-2).transpose(1, 0)

    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out