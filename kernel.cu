// Lab1-2_Matrix_Determination.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <stdio.h>
#include "Matrix.h"
#include <random>
#include <fstream>
#include <chrono>


double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main()
{
  /*  double matrix_[5][5] = {
        {5,1,3,5,6},
        {3,4,6,8,9},
        {5,16,3,4,2},
        {6,9,7,1,4},
        {8,5,3,2,1}
    };

    */

    srand(time(NULL));

  
    std::ofstream CPU_file("CPU16.txt");
    std::ofstream GPU_file("GPU16.txt");

    int N = 5000;
    double** matrix = new double* [N];

    for (int i = 0; i < N; i++) {
        matrix[i] = new double[N];
        for (int j = 0; j < N; j++) {
            //matrix[i][j] = fRand(-2, 2);
            matrix[i][j] = fRand(0.5,1);
        }
    }

    /*
    for (int i = 0; i < 5; i++) {
        std::cout << std::endl;
        for (int j = 0; j < 5; j++) {
                std::cout << matrix[i][j] << " ";
        }
    }


    std::cout << "\n\n";
    double Tmtrx = SqMatrixCalculator::getDeterminator(matrix, 500, SqMatrixCalculator::GPU);

    std::cout << "\n\n";
    double Tmtrx2 = SqMatrixCalculator::getDeterminator(matrix, 500, SqMatrixCalculator::CPU);

    std::cout << Tmtrx << " " << Tmtrx2;

    std::cout << CLOCKS_PER_SEC;
    */
    
    
    for (int matrixSize = 2650; matrixSize < N; matrixSize+=10) {
        auto begin_CPU = std::chrono::steady_clock::now();
        double determinator = SqMatrixCalculator::getDeterminator(matrix, matrixSize, SqMatrixCalculator::CPU);
        auto end_CPU = std::chrono::steady_clock::now();
        auto elapsed_ms_CPU = std::chrono::duration_cast<std::chrono::milliseconds>(end_CPU - begin_CPU);

        auto begin_GPU = std::chrono::steady_clock::now();
        double determinator2 = SqMatrixCalculator::getDeterminator(matrix, matrixSize, SqMatrixCalculator::GPU);
        auto end_GPU = std::chrono::steady_clock::now();
        auto elapsed_ms_GPU = std::chrono::duration_cast<std::chrono::milliseconds>(end_GPU - begin_GPU);

        CPU_file << matrixSize << " " << elapsed_ms_CPU.count() << std::endl;
        GPU_file << matrixSize << " " << elapsed_ms_GPU.count() << std::endl;

    }
    
    
    


    

}

