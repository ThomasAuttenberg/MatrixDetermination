#pragma once
#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "device_launch_parameters.h"

//__device__ double bufferMatrix[10000][10000];

__global__ void device_prepare_flags(bool* flags, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		flags[index] = false;
	}
}

__global__ void device_subMatrixTriangulationStep(double* m, double* stringkoefs, int startpos, int size, bool* flags) {

	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stringIndex = index / size;
	int inStringIndex = index % size;
	if (stringIndex > startpos && inStringIndex >= startpos && stringIndex < size && inStringIndex < size) {
		if (inStringIndex == startpos) {
			stringkoefs[stringIndex] = m[stringIndex * size+inStringIndex] / m[size*startpos + startpos];
			m[stringIndex * size + inStringIndex] = 0;
			flags[stringIndex] = true;
		}

		if (inStringIndex == startpos) return;


		while (flags[stringIndex] == false);

		m[index] -= (m[size*startpos + inStringIndex] * stringkoefs[stringIndex]);
	}
}

__global__ void device_findRowWithMaxElem(double* bufferMatrix, int size, int coloumn, double* output) {

	double maxEl = DBL_MIN;
	double row_i = coloumn;

	for (int i = coloumn; i < size; i++) {

		if (abs(bufferMatrix[i * size + coloumn]) > maxEl) {
			row_i = i;
			maxEl = abs(bufferMatrix[i * size + coloumn]);
		}

	}

	*output = row_i;

}

double* stringkoefs;
bool* device_flags;

static class SqMatrixCalculator
{
public:

	typedef double** Matrix;
	typedef double* MRow;
	typedef double MElement;

	enum {
		CPU,
		GPU
	};


	static MElement getDeterminator(Matrix matrix, int size, int type) {
		bool sign = 1;
		Matrix triangularMatrix = triangulation(matrix, size, type, &sign);
		MElement determinator = 1;

		for (int string = 0; string < size; string++) {
			determinator *= triangularMatrix[string][string];
		}
		delete[] triangularMatrix;
		return sign == 1 ? determinator : -determinator;
	}

	static Matrix triangulation(Matrix matrix, int size, short type, bool* sign = nullptr) {

		Matrix tMatrix;

		if (type == CPU) {

			tMatrix = copyMatrix(matrix, size);

			for (int coloumn = 0; coloumn < size; coloumn++) {

				int maxElString = findRowWithMaxElem(tMatrix, size, coloumn, coloumn);

				if (maxElString != coloumn) {
					swapStrings(tMatrix[maxElString], tMatrix[coloumn]);
					if (sign != nullptr) *sign = !(*sign);

				}

				for (int string = coloumn + 1; string < size; string++) {
					MElement koeff = tMatrix[string][coloumn] / tMatrix[coloumn][coloumn];

					sumStrings(tMatrix[string], tMatrix[coloumn], -koeff, size);

					tMatrix[string][coloumn] = 0;
				}

			}
		}

		if (type == GPU) {

			double* bufferMatrix;;

			initBuffer(matrix, bufferMatrix, size);

			for (int coloumn = 0; coloumn < size; coloumn++) {
				int maxElString = GPUfindRowWithMaxElem(bufferMatrix, size, coloumn);
				if (maxElString != coloumn) {
					GPUswapStrings(bufferMatrix, coloumn, maxElString, size);
					if (sign != nullptr) *sign = !(*sign);
				}
				initGPUTriangulationContext(size);
				GPU_TriangulationStep(bufferMatrix, coloumn, size);
			}
			freeGPUTriangulationContext();

			tMatrix = takeBuffer(bufferMatrix, size);
			freeBuffer(bufferMatrix);

		}

		return tMatrix;

	}

	static Matrix copyMatrix(Matrix matrix, int size) {
		Matrix newMatrix = new MRow[size];
		for (int i = 0; i < size; i++) {
			newMatrix[i] = new MElement[size];
			for (int j = 0; j < size; j++) {
				newMatrix[i][j] = matrix[i][j];
			}
		}
		return newMatrix;
	}

	static int findRowWithMaxElem(Matrix matrix, int size, int from, int coloumn = -1) {
		int Row_i = 0;
		MElement maxEl = -std::numeric_limits<MElement>::infinity();

		if (coloumn < 0 || coloumn >= size)
			for (int string = from; string < size; string++) {
				for (int col = 0; col < size; col++) {
					if (abs(matrix[string][col]) > maxEl) {
						Row_i = string;
						maxEl = abs(matrix[string][col]);
					}
				}
			}

		if (coloumn >= 0 && coloumn < size)
			for (int string = from; string < size; string++) {
				if (abs(matrix[string][coloumn]) > maxEl) {
					Row_i = string;
					maxEl = abs(matrix[string][coloumn]);
				}
			}

		return Row_i;

	}

	static void swapStrings(MRow& a, MRow& b) {
		double* temp = a;
		a = b;
		b = temp;
	}

	static void sumStrings(MRow& sum, MRow& summand, MElement k, int size) {
		if (sum != summand) {
			for (int i = 0; i < size; i++) {
				sum[i] += (summand[i] * k);
			}
		}
	}

private:
	
	static void freeBuffer(double* matrix) {
		cudaFree(matrix);
	}

	static void freeGPUTriangulationContext() {
		cudaFree(stringkoefs);
		cudaFree(device_flags);
	}

	static void initGPUTriangulationContext(int size) {
		cudaMalloc((void**)&stringkoefs, (size + 1) * sizeof(double));
		cudaMalloc((void**)&device_flags, (size + 1) * sizeof(bool));
	}

	static double GPUfindRowWithMaxElem(double* bufferMatrix, int size, int coloumn) {

		double* output;
		cudaMalloc((void**)&output, sizeof(double));;
		device_findRowWithMaxElem << <1, 1 >> > (bufferMatrix, size, coloumn, output);
		double host_output;
		cudaMemcpy(&host_output, output, sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(output);
		return host_output;
	}

	static void GPUswapStrings(double* bufferMatrix, int a, int b, int size) {
		double* temp;
		cudaMalloc((void**)&temp, sizeof(double) * size);
		cudaMemcpy(temp, bufferMatrix + size * a, sizeof(double) * (size), cudaMemcpyDeviceToDevice);
		cudaMemcpy(bufferMatrix + size * a, bufferMatrix + size * b, sizeof(double) * (size), cudaMemcpyDeviceToDevice);
		cudaMemcpy(bufferMatrix + size * b, temp, sizeof(double) * (size), cudaMemcpyDeviceToDevice);
		cudaFree(temp);

	}

	static Matrix takeBuffer(double* bufferMartix, int size) {
		MRow host_device_buffer = new MElement[size * size];
		cudaMemcpy(host_device_buffer, bufferMartix, sizeof(MElement) * size * size, cudaMemcpyDeviceToHost);
		cudaFree(bufferMartix);

		Matrix output = new MRow[size];

		for (int i = 0; i < size; i++) {
			output[i] = new MElement[size];
			for (int j = 0; j < size; j++) {
				output[i][j] = host_device_buffer[i * size + j];
			}
		}
		delete[] host_device_buffer;
		//std::cout << "\n\n" << *host_device_buffer << "\n\n";
		return output;

	}

	static void initBuffer(Matrix m, double*& bufferMatrix, int size) {
		MRow matrix = new MElement[size * size];
		for (int i = 0; i < size; i++) {
			memcpy(matrix + i * size, m[i], sizeof(MElement) * size);
		}
		cudaMalloc((void**)&(bufferMatrix), sizeof(MElement) * size * size);
		cudaMemcpy(bufferMatrix, matrix, sizeof(MElement) * size * size, cudaMemcpyHostToDevice);
		delete[] matrix;
	}

	static void GPU_TriangulationStep(double* bufferMatrix, int startpos, int size) {

		int preparingBlocksNumber = (size / 512) + 1;
		device_prepare_flags<<<preparingBlocksNumber, 512 >>>(device_flags, size);
		int blocksNumber = (size*size / 512) + 1;
		device_subMatrixTriangulationStep <<<blocksNumber, 512>> > (bufferMatrix, stringkoefs, startpos, size, device_flags);

	}



};
