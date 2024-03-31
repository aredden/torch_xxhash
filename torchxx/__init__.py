import torch
from torchxx_cpp_ext import calc_hash_64, calc_hash_32, calc_hash_128

def xxhash_64(t: torch.Tensor) -> str:
    """
    Hashes a tensor using xxhash64
    
    Tensor must be on CPU, but dtype can be anything.

    Parameters
    ----------
    t : torch.Tensor
        The tensor to hash
    
    Returns
    -------
    str
        The hash of the tensor

    """
    return calc_hash_64(t)

def xxhash_32(t: torch.Tensor) -> str:
    """
    Hashes a tensor using xxhash32

    
    Tensor must be on CPU, but dtype can be anything.

    Parameters
    ----------
    t : torch.Tensor
        The tensor to hash
    
    Returns
    -------
    str
        The hash of the tensor
    
    """
    return calc_hash_32(t)

def xxhash_128(t: torch.Tensor) -> str:
    """
    Hashes a tensor using xxhash128

        
    Tensor must be on CPU, but dtype can be anything.

    Parameters
    ----------
    t : torch.Tensor
        The tensor to hash
    
    Returns
    -------
    str
        The hash of the tensor

    """
    return calc_hash_128(t)
