import torch


def check_and_get_common_dim(tensor_list):
    """
    Check if all tensors in the list have the same number of dimensions.

    Args:
    - tensor_list (list or tuple of Tensors): List of tensors to be checked.

    Returns:
    - common_dim (int): The common number of dimensions if all tensors have the same.

    Raises:
    - ValueError: If the tensors do not have the same number of dimensions.
    """

    if not tensor_list:
        raise ValueError("The list of tensors is empty.")

    common_dim = tensor_list[0].dim()

    if not all(tensor.dim() == common_dim for tensor in tensor_list):
        raise ValueError("Not all tensors have the same number of dimensions.")

    return common_dim


def pad_and_stack_tensors(tensor_list, pad_dim=-1, pad_value=0):
    """
    Pad tensors along a specified dimension and then stack them along a new first dimension.

    Args:
    - tensor_list (list or tuple of Tensors): List of tensors to be padded and stacked.
    - pad_dim (int): Dimension along which to pad. Default is -1 (the last dimension).
    - pad_value (float): The value to use for padding.

    Returns:
    - stacked_tensor (Tensor): A new tensor that is formed by stacking the padded tensors along a new first dimension.
    """

    # Figure out the max size along the padding dimension
    max_size = max(tensor.shape[pad_dim] for tensor in tensor_list)

    # Initialize a list to hold the padded tensors
    padded_tensors = []

    # index for pad_spec based on pad_dim
    if pad_dim >= 0:
        pad_dim = pad_dim - tensor_list[0].dim()
    index = -pad_dim * 2 - 1

    for tensor in tensor_list:
        # Calculate the amount of padding needed for this tensor
        pad_size = max_size - tensor.shape[pad_dim]

        # Generate a pad specification for this tensor
        # This will be a tuple of zeros, except for the padding dimension
        pad_spec = [0 for _ in range(tensor.dim() * 2)]
        pad_spec[index] = pad_size  # Update padding size
        pad_spec = tuple(pad_spec)  # Convert back to tuple
        print(pad_spec)
        # Perform the padding operation
        padded_tensor = torch.nn.functional.pad(tensor, pad=pad_spec, value=pad_value)

        # Append to list of padded tensors
        padded_tensors.append(padded_tensor)

    print(padded_tensors)
    # Stack the padded tensors along a new first dimension
    stacked_tensor = torch.stack(padded_tensors, dim=0)

    return stacked_tensor


# Example usage
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6], [4, 5, 6]])
tensor2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
tensor3 = torch.tensor([[1], [2], [3]])

result = pad_and_stack_tensors([tensor1, tensor2, tensor3], pad_dim=-1, pad_value=0)
print(result)
