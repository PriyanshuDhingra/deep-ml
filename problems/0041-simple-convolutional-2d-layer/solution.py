import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape
	input_padded = np.pad(input_matrix, pad_width=padding, mode='constant', constant_values=0)
	output_matrix = np.zeros((
		(input_height + 2*padding - kernel_height) // stride + 1,
		(input_width + 2*padding - kernel_height) // stride + 1,
	))
	for i in range(output_matrix.shape[0]):
		for j in range(output_matrix.shape[1]):
			start_row = i * stride
			start_column = j * stride
			window = input_padded[
				start_row : start_row + kernel_height,
				start_column : start_column + kernel_width
			]
			output_matrix[i][j] = np.sum(window * kernel)
    
	return output_matrix
