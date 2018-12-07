#include "stdafx.h"
#include "lab4.h"
#include <string>
using namespace log645;
	
	int main(int argc, char *argv[])
	{
		// get input parameters
		int m = atoi(argv[1]);
		int n = atoi(argv[2]);
		int k = atoi(argv[3]);
		float td = atof(argv[4]);
		float h = atof(argv[5]);
		
		Lab4 worker(m,n,k,td,h);
		return 0;
}
namespace log645
{
	Lab4::Lab4(int m, int n, int k, float td, float h)
	{
		_M = m;
		_N = n;
		_K = k;
		_td = td;
		_h = h;
		//to avoir recomputing each time
		_scaler = (_td) / (_h * _h);
		printf("\nm %d; n %d; k %d; td %.2f, h %.2f scaler %.2f\n", _M, _N, _K, _td, _h, _scaler);
		
		// Timers
		struct timespec requestStart, requestEnd;
		double tempExecutionParallele;
		double tempExecutionSequentiel;

		Init();
		printf("init");
		//todo
		printf("b4");
		ParallelWork();
		printf("done");
		system("pause");
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
		}
		Copy();
	}
	void log645::Lab4::Init()
	{
		_matrixSize = _N * _M;
		_matrixBufferSize = sizeof(float) * _matrixSize;
		_matrix = (float *)malloc(_matrixBufferSize);
		_matrixPrevious = (float *)malloc(_matrixBufferSize);
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
	// Displays error message if the operation wasn't a success
	void Lab4::checkForError(cl_int status, char* taskDescription)
	{
		if (status != CL_SUCCESS)
		{
			printf("Error while %s, code : %d\n", taskDescription, status);
		}
	}
	void Lab4::ParallelWork()
	{
		// Load the kernel source code into the array source_str
		char *programFile;
		size_t source_size;

		programFile = oclLoadProgSource("lab4.cl", "", &source_size);
		// Get platform and device information
		cl_platform_id platform_id = nullptr;
		cl_device_id device_id = nullptr;
		cl_uint ret_num_devices;
		cl_uint ret_num_platforms;
		cl_int status = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
		status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,&device_id, &ret_num_devices);
		checkForError(status, "Error: platforms");
		printf("platform\n");

		// Create an OpenCL context
		cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
		printf("context\n");
		// Create a command queue
		cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &status);
		printf("queue\n");
		// Create memory buffers on the device for each matrix M,N,scaler
		cl_mem matrix_present_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, _matrixBufferSize, NULL, &status);
		cl_mem matrix_previous_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, _matrixBufferSize, NULL, &status);
		cl_mem m_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &status);
		cl_mem n_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &status);
		cl_mem scaler_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
		printf("created buffers\n");
		// Copy the matrix to their respective memory buffers
		status = clEnqueueWriteBuffer(command_queue, matrix_present_mem_obj, CL_TRUE, 0, _matrixBufferSize, _matrix, 0, NULL, NULL);
		checkForError(status, "Error: present matrix queue");
		status = clEnqueueWriteBuffer(command_queue, matrix_previous_mem_obj, CL_TRUE, 0, _matrixBufferSize, _matrixPrevious, 0, NULL, NULL);
		checkForError(status, "Error: previous matrix queue");
		status = clEnqueueWriteBuffer(command_queue, m_mem_obj, CL_TRUE, 0, sizeof(int), &_M, 0, NULL, NULL);
		checkForError(status, "Error: M");
		status = clEnqueueWriteBuffer(command_queue, n_mem_obj, CL_TRUE, 0, sizeof(int), &_N, 0, NULL, NULL);
		checkForError(status, "Error: N");
		status = clEnqueueWriteBuffer(command_queue, n_mem_obj, CL_TRUE, 0, sizeof(float), &_scaler, 0, NULL, NULL);
		checkForError(status, "Error: Scaler");
		printf("copied the data\n");
		// Create a program from the kernel source
		cl_program program = clCreateProgramWithSource(context, 1,(const char **)&programFile, (const size_t *)&source_size, &status);
		checkForError(status, "Error: source program");
		printf("program create\n");
		// Build the program
		status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
		printf("program build\n");
		checkForError(status, "Error: building program");
		// Create the OpenCL kernel
		cl_kernel kernel = clCreateKernel(program, "HeatTransfer", &status);
		printf("kernel\n");
		// Set the arguments of the kernel
		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matrix_present_mem_obj);
		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&matrix_previous_mem_obj);
		status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&m_mem_obj);
		status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&n_mem_obj);
		status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&scaler_mem_obj);
		printf("kernel args\n");
		// Execute the OpenCL kernel on the matrix
		size_t global_item_size = _matrixSize; // Process the entire matrix 
		size_t local_item_size = 128; // Divide work items into groups of 128
		/*ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
			&global_item_size, &local_item_size, 0, NULL, NULL);*/
		printf("looping\n");
		// loop on time
		for(int i(1); i < _K; i++)
		{
			printf("k %d\n",i);
			cl_event complete;
			if (i % 2 == 0) // even
			{
				clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matrix_present_mem_obj);
				clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&matrix_previous_mem_obj);
			}
			else//odd
			{
				clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matrix_previous_mem_obj);
				clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&matrix_present_mem_obj);
			}
			
			status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,&global_item_size, &local_item_size, 0, NULL, &complete);
			clWaitForEvents(1, &complete);
			clReleaseEvent(complete);
		}

		// Read the memory buffer matrix on the device to the local variable matrix
		status = clEnqueueReadBuffer(command_queue, matrix_present_mem_obj, CL_TRUE, 0, _matrixBufferSize, _matrix, 0, NULL, NULL);
		// Read appropriate buffer and display the result
		if (_K % 2 == 0)
			status = clEnqueueReadBuffer(command_queue, matrix_present_mem_obj, true, 0, _matrixBufferSize, _matrix, 0, nullptr, nullptr);
		else
			status = clEnqueueReadBuffer(command_queue, matrix_present_mem_obj, true, 0, _matrixBufferSize, _matrix, 0, nullptr, nullptr);

		checkForError(status, "Error: reading matrix");
		// Display the result to the screen
		Affiche();

		// Clean up
		status = clFlush(command_queue);
		status = clFinish(command_queue);
		status = clReleaseKernel(kernel);
		status = clReleaseProgram(program);
		status = clReleaseMemObject(matrix_present_mem_obj);
		status = clReleaseMemObject(matrix_previous_mem_obj);
		status = clReleaseCommandQueue(command_queue);
		status = clReleaseContext(context);
		free(_matrix);
		free(_matrixPrevious);

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