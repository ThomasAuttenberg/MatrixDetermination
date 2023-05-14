#pragma once
#define _CRT_SECURE_NO_WARNINGS //Константа против ошибок
#include "device_launch_parameters.h" //Библиотека для параллельных вычислений на графических процессорах 
NVIDIA.
#include <cuda_runtime.h> //Библиотека для работы с CUDA
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <time.h>
#include <locale.h>
class temp
{

	cudaError_t err = cudaSuccess; //Инициализация переменной err типа cudaError_t и присвоение ей значения 

		void check_err() { //Функция для определения ошибок при выполнении вычислений с использованием CUDA
		if (err != cudaSuccess) { //Проверка на ошибку при помощи стандартной перменной выполнения 
				fprintf(stderr, "Failed ", cudaGetErrorString(err)); //Если произойдет ошибка, то выведется текст 
				exit(EXIT_FAILURE); //Завершение работы
		}
	}

	void getMatrix(int N, float* A) { //Функция для инициализации матрицы размера N*N 
		for (int i = 0; i < N; i++) { //Цикл для прохода по строкам
			for (int j = 0; j < N; j++) //Цикл для прохода по столбцам
				A[i + j * N] = rand() % 10; //Записть в каждую ячейку матрицы значения от 0 до 9 
				A[i + N * N] = 0; //Установка конца матрицы
		}
	}

	__global__ void gauss_stage1(float* a, int n, float x, int N) { //Определение ядра CUDA-функции, которая выполняет первую стадию алгоритма Гаусса для приведения матрицы к диагональному виду(Массив над которым
			int i = blockDim.x * blockIdx.x + threadIdx.x; //I это номер текущего потока CUDA, вычисляемый на основе параметров blockDim, blockIdx и threadIdx
			if (i <= N - n + 1) { //Условие гарантирует, что операции выполняются только для элементов матрицы, 
					a[n + N * (i + n)] /= x; //Деление элемента матрицы на значение X
			}
	}

	__global__ void gauss_stage2(float* a, int n, int i, float x, int N) { //Определение ядра CUDA-функции, которая 
			int j = blockDim.x * blockIdx.x + threadIdx.x; //J это номер текущего потока CUDA, вычисляемый на 
			if (j <= N - n - 1) { //Это условие гарантирует, что операции выполняются только для элементов матрицы, 
				a[i + N * (j + n + 1)] -= a[n + N * (j + n + 1)] * x; //Если условие выполнено, то выполняется 
			}
	}

	__global__ void gauss_stage3(float* a, int n, int N) { //Определение ядра CUDA-функции, которая выполняет третью 
			int i = blockDim.x * blockIdx.x + threadIdx.x; //I это номер текущего потока CUDA, вычисляемый на основе 
			if (i < n) { //Это условие гарантирует, что операции выполняются только для элементов матрицы, 
					a[i + N * N] -= a[n + N * N] * a[i + n * N]; //Выполняется операция вычитания из элемента 
			}
	}


	void findDeterminant(int N) { //Функция для нахождения определителя матрицы на GPU с использованием CUDA
		check_err(); //функция, которая проверяет ошибки, возникшие при предыдущих операциях CUDA

		int threadsPerBlock = 128, //Количество потоков в блоке.
		blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock, //Количество блоков в сетке
		size = sizeof(float) * N * (N + 1); //Расчет размера матрицы в байтах
		float* A = (float*)malloc(N * (N + 1) * sizeof(float)); //Массив, содержащий матрицу и вектор свободных 
		getMatrix(N, A); //Функция, которая заполняет матрицу случайными числами от 0 до 9
		float* _A = NULL; //Указатель на выделенную память для массива A на GPU
		err = cudaMalloc((void**)&_A, size); check_err(); //Выделение памяти на GPU

		err = cudaMemcpy(_A, A, size, cudaMemcpyHostToDevice); check_err(); //Копирование данных из массива 

			for (int i = 0; i < N; i++) { //Цикл выполняющий первые два этапа метода Гаусса
				gauss_stage1 << <blocksPerGrid, threadsPerBlock >> > (_A, i, A[i + i * N], N); //Вызов ядра для 
					for (int j = i + 1; j < N; j++)
						gauss_stage2 << <blocksPerGrid, threadsPerBlock >> > (_A, i, j, A[j + i * N], N); //Вызов 
			}

		for (int i = N - 1; i > 0; i--) //Цикл выполняющий третий этам метода Гаусса с шагом -1
			gauss_stage3 << <blocksPerGrid, threadsPerBlock >> > (_A, i, N); //Вызов ядра для выполнения 
			
	
		double det = 1.0; //Инициализация определителя матрицы

		for (int j = 0; j < N; j++) //Цикл для расчета определитля матрицы
			det *= A[j + N * N]; //Умножение всех элементов диагонали матрицы
		cudaFree(_A); //Освобождение памяти на GPU
		free(A); //Освобождение памяти на хосте
	

	int main(void) {
		srand(time(NULL));
		const int size = 1000; //Максимальная размерность матрицы
		FILE* text;
		for (int N = 2; N <= size; N++) { //Цикл для создания матриц размерностью от 2 до N и нахождение их 
				srand(time(NULL));
			int before = clock(); //Инициализация переменной для записи времени нахождения определителя 
			findDeterminant(N); //вызов функции нахождения определителя
			double time = (clock() - before) / (double)CLOCKS_PER_SEC; //Фиксация времени
			printf("Time: %.3f sec.\n", time); //Вывод данных о времени в консоль
			text = fopen("GPU.txt", "a"); //Открите файла для записи времени
			fprintf(text, "%d %.3f\n", N, time); //Запись времени в файл
			fclose(text); //Закрытие файла
		}
		return 0;
	}
};

