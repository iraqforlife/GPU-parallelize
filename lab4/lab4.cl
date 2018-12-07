__kernel void HeatTransfer(__global const double *matrix, __global const double *matrixPrevious, __global const int M, __global const int N, __global const double scaler)
{
	int i = get_global_id(0);
	int x = floor(i / N);
	int y = i % N;

	if (x > 0 && x < M - 1 && y > 0 && y < N - 1) {
		double x1 = matrix[(x - 1) *_N + y];// matrix[x-1][y]
		double x2 = matrix[(x + 1) * N + y];// matrix[x+1][y]
		double y1 = matrix[x * N + y - 1];// matrix[x][y-1]
		double y2 = matrix[x * N + y + 1];// matrix[x][y+1]
		matrixPrevious[i] = (1 - (4 * scaler)) * matrix[i] + scaler * (x1 + x2 + y1 + y2);
	}
}

__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {

	// Get the index of the current element to be processed
	int i = get_global_id(0);

	// Do the operation
	C[i] = A[i] + B[i];
}
