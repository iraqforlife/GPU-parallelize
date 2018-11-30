__kernel void HeatTransfer()
{
	int id = get_global_id(0);
}

__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {

	// Get the index of the current element to be processed
	int i = get_global_id(0);

	// Do the operation
	C[i] = A[i] + B[i];
}