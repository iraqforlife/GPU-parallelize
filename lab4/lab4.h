#pragma once
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

#define TAG_FOR_NEXT 1
#define TAG_FOR_GATHER 2

#define BILLION 1E9
namespace log645
{
	class Lab4
	{
	public:
		Lab4(int m, int n, int np, float td, float h);
		~Lab4();
		char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength);

	private:
		//input
		int _M;
		int _N;
		int _K;
		float _td;
		float _h;
		//working vars
		int _matrixSize;
		float _matrixBufferSize;
		int _blockSize;
		int _startPosition;
		float _scaler;
		//multi processors
		int _rank;
		int _numberOfProcessorToUse;
		int _nbProc;
		//matrix
		float *_matrix;
		float *_matrixPrevious;


		void Reset();

		/**
		* Initialise la matrice
		*/
		void Init();

		/**
		* Traite le problème de façon séqquentielle
		*/
		void Work();

		/**
		* Affihce la matrice
		*/
		void Affiche();

		/**
		* Copie matrix dans matCopy
		*/
		void Copy();

		void checkForError(cl_int status, char * taskDescription);

		/**
		* Traite le probleme de facon parallele
		*/
		void ParallelWork();
		
	};

}
