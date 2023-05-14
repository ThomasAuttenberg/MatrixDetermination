#pragma once
#include "cuda_runtime.h"



static class GPUMatrixCalculator
{
public:
	typedef double** Matrix;
	typedef double* MRow;
	typedef double MElement;

	static MElement getDeterminator(Matrix matrix, int size) {
		bool sign = 1;
		Matrix triangularMatrix = triangulation(matrix, size, &sign);
		MElement determinator = 1;

		for (int string = 0; string < size; string++) {
			determinator *= triangularMatrix[string][string];
		}

		return sign == 1 ? determinator : -determinator;
	}

	static Matrix triangulation(Matrix matrix, int size, bool* sign = nullptr) {

		Matrix tMatrix = copyMatrix(matrix, size);

		for (int coloumn = 0; coloumn < size; coloumn++) {

			int maxElString = findRowWithMaxElem(tMatrix, size, coloumn, coloumn);

			if (maxElString != coloumn) {
				swapStrings(tMatrix[maxElString], tMatrix[coloumn]);
				if (sign != nullptr) *sign = !(*sign);
			}

			for (int string = coloumn + 1; string < size; string++) {
				MElement koeff = tMatrix[string][coloumn] / tMatrix[coloumn][coloumn];
				sumStrings(tMatrix[string], tMatrix[coloumn], -koeff, size);
			}

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
					if (matrix[string][col] > maxEl) {
						Row_i = string;
						maxEl = matrix[string][col];
					}
				}
			}

		if (coloumn >= 0 && coloumn < size)
			for (int string = from; string < size; string++) {
				if (matrix[string][coloumn] > maxEl) {
					Row_i = string;
					maxEl = matrix[string][coloumn];
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

	static void GPU_sumstrings(MRow& sum, MRow& summand, MElement k, int size) {
		MRow dev_sum;
		cudaMalloc((void**)&dev_sum, sizeof(MElement) * size);
		cudaMemcpy(dev_sum, sum, sizeof(MElement) * size, cudaMemcpyHostToDevice);
		MRow dev_summand;
		cudaMalloc((void**)&dev_summand, sizeof(MElement) * size);
		cudaMemcpy(dev_summand, sum, sizeof(MElement) * size, cudaMemcpyHostToDevice);
		MElement* dev_k;
		cudaMalloc((void**)&dev_k, sizeof(MElement));
		cudaMemcpy(&dev_k, &k, sizeof(MElement), cudaMemcpyHostToDevice);
		device_sumStrings(dev_sum, dev_summand, dev_k);
		cudaMemcpy(&sum, dev_sum, sizeof(MElement) * size, cudaMemcpyDeviceToHost);
	}


};

__global__ static void device_sumStrings(MatrixMRow sum, MRow summand, MElement* k) {
	sum[threadIdx.x] += (summand[threadIdx.x] * *k);
}

