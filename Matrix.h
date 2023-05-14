#pragma once
#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "device_launch_parameters.h"

//__device__ double bufferMatrix[10000][10000];


__global__ void setValue(double** setting, double* setter, int* i) {
	setting[*i] = setter;
}

__global__ void device_sumStrings(double* sum, double* summand, double k) {
	sum[threadIdx.x] += (summand[threadIdx.x] * k);
}

__global__ void device_subMatrixTriangulationStep(double* m, double* stringkoefs, int startpos, int size, bool* flags) {
	//__shared__ double k;
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stringIndex = index / size;
	int inStringIndex = index % size;
	if (stringIndex > startpos && inStringIndex >= startpos && stringIndex < size && inStringIndex < size) {
		if (inStringIndex == startpos) {
			stringkoefs[stringIndex] = m[stringIndex * size+inStringIndex] / m[size*startpos + startpos];
			//printf("%f %f %d\n", m[stringIndex * size + inStringIndex] / m[size * startpos + startpos], stringkoefs[stringIndex], stringIndex);
			m[stringIndex * size + inStringIndex] = 0;
			flags[stringIndex] = true;
		}

		//__threadfence();

		if (inStringIndex == startpos) return;


		while (flags[stringIndex] == false);

		m[index] -= (m[size*startpos + inStringIndex] * stringkoefs[stringIndex]);
			//printf("NATIVE: %f %f %d\n", m[stringIndex * size + inStringIndex] / m[size * startpos + startpos], stringkoefs[stringIndex], stringIndex);;
	}
	/*
	* 
	if (stringIndex > 0 && stringIndex >= startpos && stringIndex<size && inStringIndex<size && inStringIndex >= startpos) {
		if (inStringIndex == startpos) {
			stringkoefs[stringIndex] = m[stringIndex * size] / m[(stringIndex - 1) * size];
			__syncthreads();
			m[inStringIndex] = 0;
		}
		else {
			m[stringIndex * size + inStringIndex] -= (m[(stringIndex - 1) * size + inStringIndex] * stringkoefs[stringIndex]);
		}
	}*/
	
	/*
	int StringNumber = size * (startpos + blockIdx.x + 1) + (startpos);
	if (threadIdx.x == 0) {
		k = m[size * (startpos + blockIdx.x + 1) + (startpos)] / m[size * startpos + startpos];
		m[size * (startpos + blockIdx.x + 1) + (startpos)] = 0;
	}

	if (threadIdx.x != 0) {
		m[size * (startpos + blockIdx.x + 1) + (startpos + threadIdx.x)] -= (m[size * startpos + startpos + threadIdx.x] * k);
	}*/
}

__global__ void device_swap_strings(double* bufferMatrix, int a, int b, int size) {
	double* temp = new double[size];
	memcpy(temp, bufferMatrix + size * a, sizeof(double) * (size));
	memcpy(bufferMatrix + size * a, bufferMatrix + size * b, sizeof(double) * (size));
	memcpy(bufferMatrix + size * b, temp, sizeof(double) * (size));
	delete[] temp;
}

__global__ void device_findRowWithMaxElem(double* bufferMatrix, int* size, int* coloumn, double* output) {

	double maxEl = DBL_MIN;
	double row_i = *coloumn;

	for (int i = *coloumn; i < *size; i++) {

		if (abs(bufferMatrix[i * *size + *coloumn]) > maxEl) {
			row_i = i;
			maxEl = abs(bufferMatrix[i * *size + *coloumn]);
		}

	}

	*output = row_i;

}


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
				GPU_subMatrixTriangulation(bufferMatrix, coloumn, size);
			}

			tMatrix = takeBuffer(bufferMatrix, size);

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


	static double GPUfindRowWithMaxElem(double* bufferMatrix, int size, int coloumn) {
		int* dev_size, * dev_coloumn;
		double* output;
		cudaMalloc((void**)&dev_size, sizeof(int));
		cudaMalloc((void**)&dev_coloumn, sizeof(int));
		cudaMalloc((void**)&output, sizeof(double));
		cudaMemcpy(dev_size, &size, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_coloumn, &coloumn, sizeof(int), cudaMemcpyHostToDevice);

		device_findRowWithMaxElem << <1, 1 >> > (bufferMatrix, dev_size, dev_coloumn, output);
		double host_output;
		cudaMemcpy(&host_output, output, sizeof(double), cudaMemcpyDeviceToHost);
		return host_output;
	}

	static void GPUswapStrings(double* bufferMatrix, int a, int b, int size) {
		int* dev_a, * dev_b, * dev_size;
		cudaMalloc((void**)&dev_a, sizeof(int));
		cudaMalloc((void**)&dev_b, sizeof(int));
		cudaMalloc((void**)&dev_size, sizeof(int));
		cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, &b, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_size, &size, sizeof(int), cudaMemcpyHostToDevice);

		device_swap_strings <<<1, 1 >>> (bufferMatrix, a, b, size);

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
		//std::cout << "\n\n" << *matrix << "\n\n";
		cudaMalloc((void**)&(bufferMatrix), sizeof(MElement) * size * size);
		cudaMemcpy(bufferMatrix, matrix, sizeof(MElement) * size * size, cudaMemcpyHostToDevice);
		delete[] matrix;
	}

	static void GPU_subMatrixTriangulation(double* bufferMatrix, int startpos, int size) {
		
		double* stringkoefs;
		cudaMalloc((void**)&stringkoefs, (size + 1) * sizeof(double));
		bool* flags = (bool*)malloc(sizeof(bool) * (size+1));
		bool* device_flags;
		for (int i = 0; i < size + 1; i++) flags[i] = false;
		cudaMalloc((void**)&device_flags, (size + 1) * sizeof(bool));
		cudaMemcpy(device_flags, flags, sizeof(bool) * (size + 1), cudaMemcpyHostToDevice);
	
		device_subMatrixTriangulationStep <<<size, 1000>> > (bufferMatrix, stringkoefs, startpos, size, device_flags);
		cudaFree(stringkoefs);

	}

	static void GPU_sumStrings(MRow& sum, MRow& summand, MElement k, int size) {

		MRow dev_sum;
		cudaMalloc((void**)&dev_sum, sizeof(MElement) * size);
		cudaMemcpy(dev_sum, sum, sizeof(MElement) * size, cudaMemcpyHostToDevice);


		MRow dev_summand;
		cudaMalloc((void**)&dev_summand, sizeof(MElement) * size);
		cudaMemcpy(dev_summand, summand, sizeof(MElement) * size, cudaMemcpyHostToDevice);


		device_sumStrings << <1, size >> > (dev_sum, dev_summand, k);
		cudaMemcpy(sum, dev_sum, sizeof(MElement) * size, cudaMemcpyDeviceToHost);

		cudaFree(dev_sum);
		cudaFree(dev_summand);
	}


};
