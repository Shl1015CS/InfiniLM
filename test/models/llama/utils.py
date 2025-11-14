"""
Utility functions for InfiniLM Llama model tests.

This module provides shared utility functions for tensor conversion,
parameter name normalization, and tensor comparison.
"""

from typing import Tuple, Dict
import torch

try:
    import infinicore
except ImportError:
    infinicore = None


def normalize_param_name(name: str) -> str:
    """Normalize parameter name (remove 'model.' prefix if present)"""
    if name.startswith("model."):
        return name[6:]  # Remove "model." prefix
    return name


def to_infinicore_dtype(torch_dtype):
    """Convert PyTorch data type to infinicore data type"""
    if infinicore is None:
        raise ImportError("InfiniCore package not found")

    if torch_dtype == torch.float32:
        return infinicore.float32
    elif torch_dtype == torch.float16:
        return infinicore.float16
    elif torch_dtype == torch.bfloat16:
        return infinicore.bfloat16
    elif torch_dtype == torch.int8:
        return infinicore.int8
    elif torch_dtype == torch.int16:
        return infinicore.int16
    elif torch_dtype == torch.int32:
        return infinicore.int32
    elif torch_dtype == torch.int64:
        return infinicore.int64
    elif torch_dtype == torch.uint8:
        return infinicore.uint8
    elif torch_dtype == torch.bool:
        return infinicore.bool
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def torch_to_infinicore_tensor(torch_tensor, infini_device):
    """
    Convert PyTorch tensor to InfiniCore tensor.

    Args:
        torch_tensor: PyTorch tensor
        infini_device: InfiniCore device object

    Returns:
        InfiniCore tensor
    """
    if infinicore is None:
        raise ImportError("InfiniCore package not found")

    # Ensure tensor is contiguous (but keep it on its current device)
    torch_tensor = torch_tensor.contiguous()

    # Convert dtype
    infini_dtype = to_infinicore_dtype(torch_tensor.dtype)

    # Create InfiniCore tensor from torch tensor's data pointer
    if torch_tensor.is_contiguous():
        return infinicore.from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            dtype=infini_dtype,
            device=infini_device,
        )
    else:
        return infinicore.strided_from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            list(torch_tensor.stride()),
            dtype=infini_dtype,
            device=infini_device,
        )


def to_torch_dtype(infini_dtype):
    """Convert InfiniCore data type to PyTorch data type"""
    if infinicore is None:
        raise ImportError("InfiniCore package not found")

    # infini_dtype is a dtype object from infinicore.dtype
    # Access the underlying enum value for comparison
    from infinicore.lib import _infinicore

    # Get underlying enum value
    if hasattr(infini_dtype, '_underlying'):
        underlying = infini_dtype._underlying
    else:
        # If it's not a dtype object, try to use it directly
        underlying = infini_dtype

    # Compare underlying enum values
    if underlying == _infinicore.DataType.F32:
        return torch.float32
    elif underlying == _infinicore.DataType.F16:
        return torch.float16
    elif underlying == _infinicore.DataType.BF16:
        return torch.bfloat16
    elif underlying == _infinicore.DataType.I8:
        return torch.int8
    elif underlying == _infinicore.DataType.I16:
        return torch.int16
    elif underlying == _infinicore.DataType.I32:
        return torch.int32
    elif underlying == _infinicore.DataType.I64:
        return torch.int64
    elif underlying == _infinicore.DataType.U8:
        return torch.uint8
    elif underlying == _infinicore.DataType.BOOL:
        return torch.bool
    else:
        raise ValueError(
            f"Unsupported infinicore dtype: {infini_dtype} (underlying enum: {underlying})")


def infinicore_to_torch_tensor(infini_tensor, torch_reference):
    """
    Convert InfiniCore tensor to PyTorch tensor for comparison.

    Args:
        infini_tensor: InfiniCore tensor (can be raw C++ tensor or Python wrapper)
        torch_reference: PyTorch tensor reference (for shape and device)

    Returns:
        PyTorch tensor with InfiniCore data on the same device as torch_reference
    """
    if infinicore is None:
        raise ImportError("InfiniCore package not found")

    # Wrap raw C++ tensor in Python Tensor wrapper if needed
    # get_parameter returns a raw _infinicore.Tensor, but we need infinicore.Tensor
    if not hasattr(infini_tensor, '_underlying'):
        # It's a raw C++ tensor, wrap it in the Python Tensor class
        infini_tensor = infinicore.Tensor(infini_tensor)

    # Get device from reference tensor
    ref_device = torch_reference.device

    # Determine target InfiniCore device
    if ref_device.type == "cuda":
        target_infini_device = infinicore.device("cuda", ref_device.index)
    else:
        target_infini_device = infinicore.device("cpu", 0)

    # Ensure source tensor is on the target device and contiguous
    # This is important when GPU support is compiled - we need to explicitly
    # move tensors to the correct device and make them contiguous
    # When GPU support is compiled but we're using CPU, we need to be extra careful
    try:
        # For CPU, always ensure tensor is explicitly on CPU and contiguous
        if ref_device.type == "cpu":
            cpu_device = infinicore.device("cpu", 0)
            # Move to CPU if not already there
            if hasattr(infini_tensor, 'device'):
                source_device = infini_tensor.device
                if str(source_device) != str(cpu_device):
                    infini_tensor = infini_tensor.to(cpu_device)
            # Ensure contiguous
            if not infini_tensor.is_contiguous():
                infini_tensor = infini_tensor.contiguous()
        else:
            # For GPU, ensure on target device and contiguous
            if hasattr(infini_tensor, 'device'):
                source_device = infini_tensor.device
                source_device_str = str(source_device)
                target_device_str = str(target_infini_device)
                if source_device_str != target_device_str:
                    infini_tensor = infini_tensor.to(target_infini_device)
            if not infini_tensor.is_contiguous():
                infini_tensor = infini_tensor.contiguous()
    except Exception as e:
        # If device operations fail, try to ensure contiguous at least
        if hasattr(infini_tensor, 'is_contiguous') and not infini_tensor.is_contiguous():
            infini_tensor = infini_tensor.contiguous()

    # Create a PyTorch tensor with the same shape, dtype, and device as reference
    torch_result = torch.zeros(
        list(infini_tensor.shape),
        dtype=to_torch_dtype(infini_tensor.dtype),
        device=ref_device,
    )

    # For CPU, use a workaround: create an intermediate tensor and copy through it
    # This avoids issues with rearrange when GPU support is compiled
    if ref_device.type == "cpu":
        # Check if source tensor is on CUDA - if so, we need pinned memory
        source_is_cuda = False
        source_cuda_device = None
        if hasattr(infini_tensor, 'device'):
            source_device = infini_tensor.device
            source_device_str = str(source_device)
            source_is_cuda = source_device_str.startswith("cuda")
            if source_is_cuda:
                # Extract CUDA device index from device string (e.g., "cuda:0")
                try:
                    cuda_index = int(source_device_str.split(
                        ":")[1]) if ":" in source_device_str else 0
                    source_cuda_device = infinicore.device("cuda", cuda_index)
                except:
                    source_cuda_device = infinicore.device("cuda", 0)

        # If source is on CUDA, we need to ensure the intermediate CPU tensor
        # uses pinned memory. The copy_from function will handle setting the
        # CUDA context, but we need to create the intermediate with pin_memory=True
        # so it gets pinned host memory that CUDA can safely copy to.
        # Note: The empty() function will check the current runtime when pin_memory=True.
        # Since copy_from sets the context to CUDA before copying, we create the
        # intermediate with pin_memory=True, and even if it initially gets regular
        # memory, the copy operation should still work. However, for better performance
        # and reliability, we try to use .to() method which handles device transfers more safely.

        # Try using .to() method first, which handles device transfers internally
        try:
            # Use .to() to move tensor to CPU - this should handle the transfer safely
            cpu_tensor = infini_tensor.to(target_infini_device)
            if not cpu_tensor.is_contiguous():
                cpu_tensor = cpu_tensor.contiguous()

            # Create temp tensor from PyTorch and copy from the CPU tensor
            temp_tensor = torch_to_infinicore_tensor(
                torch_result, target_infini_device)
            temp_tensor.copy_(cpu_tensor)
        except Exception as e:
            # Fallback: create intermediate tensor and copy through it
            # Create an intermediate contiguous tensor on CPU
            # Use pin_memory=True if source is CUDA to ensure proper D2H copy
            intermediate = infinicore.empty(
                list(infini_tensor.shape),
                dtype=infini_tensor.dtype,
                device=target_infini_device,
                pin_memory=source_is_cuda  # Pin memory if copying from CUDA
            )

            # Copy source to intermediate first
            try:
                intermediate.copy_(infini_tensor)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to copy tensor to intermediate: {e2}")

            # Now create temp tensor from PyTorch and copy from intermediate
            temp_tensor = torch_to_infinicore_tensor(
                torch_result, target_infini_device)
            temp_tensor.copy_(intermediate)
    else:
        # For GPU, use direct copy
        temp_tensor = torch_to_infinicore_tensor(
            torch_result, target_infini_device)
        temp_tensor.copy_(infini_tensor)

    return torch_result


def tensor_all_close(tensor1: torch.Tensor, tensor2: torch.Tensor,
                     rtol: float = 1e-5, atol: float = 1e-5) -> Tuple[bool, Dict]:
    """
    Compare two tensors for approximate equality.

    Args:
        tensor1: First tensor to compare
        tensor2: Second tensor to compare
        rtol: Relative tolerance (default: 1e-5)
        atol: Absolute tolerance (default: 1e-5)

    Returns:
        Tuple of (is_close, stats_dict) where stats_dict contains:
        - max_abs_diff: Maximum absolute difference
        - mean_abs_diff: Mean absolute difference
        - max_rel_diff: Maximum relative difference
        - is_close: Boolean indicating if tensors are close
    """
    if tensor1.shape != tensor2.shape:
        return False, {"error": "Shape mismatch", "shape1": tensor1.shape, "shape2": tensor2.shape}

    diff = (tensor1 - tensor2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    relative_max_diff = (diff / tensor2.abs().clamp(min=1e-8)).max().item()

    is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

    stats = {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "max_rel_diff": relative_max_diff,
        "is_close": is_close
    }

    return is_close, stats
