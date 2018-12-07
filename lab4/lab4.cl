__kernel void HeatTransfer(__global const float *matrixPresent, __global const float *matrixPrevious, __global const int M, __global const int N, __global const float scaler)
{
	int i = get_global_id(0);
	int x = floor(i / N);
	int y = i % N;

	if (x > 0 && x < M - 1 && y > 0 && y < N - 1) {
		float x1 = matrixPresent[(x - 1) *_N + y];// matrix[x-1][y]
		float x2 = matrixPresent[(x + 1) * N + y];// matrix[x+1][y]
		float y1 = matrixPresent[x * N + y - 1];// matrix[x][y-1]
		float y2 = matrixPresent[x * N + y + 1];// matrix[x][y+1]
		matrixPrevious[i] = (1 - (4 * scaler)) * matrixPresent[i] + scaler * (x1 + x2 + y1 + y2);
	}
}