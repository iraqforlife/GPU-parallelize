#include "stdafx.h"
#include "lab4.h"
#include <string>
using namespace log645;
	
	int main(int argc, char *argv[])
	{
		int m = atoi(argv[1]);
		int n = atoi(argv[2]);
		int k = atoi(argv[3]);
		float td = atof(argv[4]);
		float h = atof(argv[5]);
		
		Lab4 worker(m,n,k,td,h);

		// Create the two input vectors
		int i;
		const int LIST_SIZE = 1024;
		int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
		int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
		for (i = 0; i < LIST_SIZE; i++) {
			A[i] = i;
			B[i] = LIST_SIZE - i;
		}

		// Load the kernel source code into the array source_str
		char *fp;
		char *source_str;
		size_t source_size;

		fp = worker.oclLoadProgSource("lab4.cl", "", &source_size);
		if (!fp) {
			fprintf(stderr, "Failed to load kernel.\n");
			exit(1);
		}
		source_str = (char*)malloc(MAX_SOURCE_SIZE);

		// Get platform and device information
		cl_platform_id platform_id = NULL;
		cl_device_id device_id = NULL;
		cl_uint ret_num_devices;
		cl_uint ret_num_platforms;
		cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
		ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
			&device_id, &ret_num_devices);

		// Create an OpenCL context
		cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

		// Create a command queue
		cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

		// Create memory buffers on the device for each vector 
		cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
			LIST_SIZE * sizeof(int), NULL, &ret);
		cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
			LIST_SIZE * sizeof(int), NULL, &ret);
		cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			LIST_SIZE * sizeof(int), NULL, &ret);

		// Copy the lists A and B to their respective memory buffers
		ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
			LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
			LIST_SIZE * sizeof(int), B, 0, NULL, NULL);

		// Create a program from the kernel source
		cl_program program = clCreateProgramWithSource(context, 1,
			(const char **)&source_str, (const size_t *)&source_size, &ret);

		// Build the program
		ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

		// Create the OpenCL kernel
		cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

		// Set the arguments of the kernel
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
		ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

		// Execute the OpenCL kernel on the list
		size_t global_item_size = LIST_SIZE; // Process the entire lists
		size_t local_item_size = 64; // Divide work items into groups of 64
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
			&global_item_size, &local_item_size, 0, NULL, NULL);

		// Read the memory buffer C on the device to the local variable C
		int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
		ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
			LIST_SIZE * sizeof(int), C, 0, NULL, NULL);

		// Display the result to the screen
		for (i = 0; i < LIST_SIZE; i++)
			printf("%d + %d = %d\n", A[i], B[i], C[i]);

		// Clean up
		ret = clFlush(command_queue);
		ret = clFinish(command_queue);
		ret = clReleaseKernel(kernel);
		ret = clReleaseProgram(program);
		ret = clReleaseMemObject(a_mem_obj);
		ret = clReleaseMemObject(b_mem_obj);
		ret = clReleaseMemObject(c_mem_obj);
		ret = clReleaseCommandQueue(command_queue);
		ret = clReleaseContext(context);
		free(A);
		free(B);
		free(C);
		return 0;
}
namespace log645
{
	Lab4::Lab4(int m, int n, int k, float td, float h)
	{
		printf("\nm %d; n %d; k %d; td %.2f, h %.2f\n", m, n, k, td, h);
		_M = m;
		_N = n;
		_K = k;
		_td = td;
		_h = h;
		//to avoir recomputing each time
		_scaler = (_td) / (_h * _h);
		printf("\nm %d; n %d; k %d; td %.2f, h %.2f scaler %.2f\n", _M, _N, _K, _td, _h, _scaler);
		system("pause");
		// Timers
		struct timespec requestStart, requestEnd;
		double tempExecutionParallele;
		double tempExecutionSequentiel;

		Init();

	}
	Lab4::~Lab4()
	{
		delete _matrix;
		delete _matrixPrevious;
	}
	void log645::Lab4::Reset()
	{
		for (int i = 0; i < _M*_N; i++) {
			int x = floor(i / _N);
			int y = i % _N;

			if (x == 0 || x == _M - 1 || y == 0 || y == _N - 1) {
				_matrix[i] = 0;
			}
			else {
				_matrix[i] = x * (_M - x - 1) * (2.0 * x / _M) * y * (_N - y - 1) * (1.0 * y / _N);
			}
			_matrixPrevious[i] = _matrix[i];
		}
	}
	void log645::Lab4::Init()
	{
		_matrixSize = _N * _M;
		_matrix = (double *)malloc(sizeof(double) * _matrixSize);
		_matrixPrevious = (double *)malloc(sizeof(double) * _matrixSize);
		/*_numberOfProcessorToUse = 0;
		//determine # of proc to use for ideale slicing
		for (int i = _nbProc; i > 0; i--) {
			if (_matrixSize % i == 0)
			{
				_numberOfProcessorToUse = i;
				break;
			}
		}
		//_numberOfProcessorToUse=30;
		_blockSize = _matrixSize / _numberOfProcessorToUse;
		if (_rank < _numberOfProcessorToUse)
			_startPosition = _blockSize * _rank;
		else
			_startPosition = 0;
			*/
		Reset();
	}
	void Lab4::Work()
	{
		double temp = 1 - (4 * _scaler);
		for (int k = 0; k < _K; k++) {
			for (int i = 0; i < _matrixSize - 1; i++) {
				int x = floor(i / _N);
				int y = i % _N;
				//U(i,j,k)=(1-4 td/h2)xU(i, j, k-1) + (td / h2)x[U(i - 1, j, k-1) + U(i + 1, j, k-1) + U(i, j - 1, k-1) + U(i, j + 1, k-1)]
				if (x > 0 && x < _M - 1 && y > 0 && y < _N - 1) {
					double x1 = _matrixPrevious[(x - 1) * _N + y];// matrix[x-1][y]
					double x2 = _matrixPrevious[(x + 1) * _N + y];// matrix[x+1][y]
					double y1 = _matrixPrevious[x * _N + y - 1];// matrix[x][y-1]
					double y2 = _matrixPrevious[x * _N + y + 1];// matrix[x][y+1]

					_matrix[i] = (temp) * _matrixPrevious[i] + _scaler * (x1 + x2 + y1 + y2);
				}
			}
			Copy();
		}
	}
	void Lab4::Affiche()
	{
		printf("\n");
		for (int y = _N - 1; y >= 0; y--) {
			for (int x = 0; x < _M; x++) {
				int index = x * _N + y;
				printf("%0.2f | ", _matrix[index]);
			}
			printf("\n\n");
		}
		printf("\n");
	}
	void Lab4::Copy()
	{
		memcpy(_matrixPrevious, _matrix, sizeof(double) * _matrixSize);
	}
	//////////////////////////////////////////////////////////////////////////////
	//! Loads a Program file and prepends the cPreamble to the code.
	//!
	//! @return the source string if succeeded, 0 otherwise
	//! @param cFilename program filename
	//! @param cPreamble code that is prepended to the loaded file, typically a set of #defines or a header
	//! @param szFinalLength returned length of the code string
	//////////////////////////////////////////////////////////////////////////////

	char * Lab4::oclLoadProgSource(const char * cFilename, const char * cPreamble, size_t * szFinalLength)
	{
		// locals
		FILE* pFileStream = NULL;
		size_t szSourceLength;
		// open the OpenCL source code file
		if (fopen_s(&pFileStream, cFilename, "rb") != 0)
		{
			return NULL;
		}
		size_t szPreambleLength = strlen(cPreamble);
		// get the length of the source code
		fseek(pFileStream, 0, SEEK_END);
		szSourceLength = ftell(pFileStream);
		fseek(pFileStream, 0, SEEK_SET);
		// allocate a buffer for the source code string and read it in
		char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1);
		memcpy(cSourceString, cPreamble, szPreambleLength);
		if (fread((cSourceString)+szPreambleLength, szSourceLength, 1, pFileStream) != 1)
		{
			fclose(pFileStream);
			free(cSourceString);
			return 0;
		}
		// close the file and return the total length of the combined (preamble + source) string
		fclose(pFileStream);
		if (szFinalLength != 0)
		{
			*szFinalLength = szSourceLength + szPreambleLength;
		}
		cSourceString[szSourceLength + szPreambleLength] = '\0';
		return cSourceString;
	}
}