import numpy as np
import math
def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    N, C, H, W = x.shape
    
    output_height = math.ceil((H-kernel_size) / stride) + 1 
    output_width = math.ceil((W-kernel_size) / stride) + 1 

    output_matrix = np.zeros((N, C, output_height, output_width), dtype=x.dtype)

    for n in range(N):
        for c in range(C):
            for i in range(output_height):
                for j in range(output_width):
                    start_i = i * stride
                    start_j = j * stride
            
                    output_matrix[n][c][i][j] = np.max(
                        x[n, c, start_i: start_i + kernel_size, start_j: start_j + kernel_size]
                    )

    return output_matrix


