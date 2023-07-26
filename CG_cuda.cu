#include <iostream>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>

// Ядро CUDA для умножения матрицы на вектор
__global__ void MatrVecMul(double* matrix, double* vector, double* result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += matrix[idx * size + i] * vector[i];
        }
        result[idx] = sum;
    }
}



double AlphaCalc(double* r, double* p, double* c, int n) {
    double res1 = 0, res2 = 0;
    for(size_t i = 0; i < n; i++) {
        res1 += r[i] * r[i];
        res2 += c[i] * p[i];
    }
    return res1 / res2;
}


double* x(double* x_prev, double alpha, double* p_prev, double* x_new, int n) {
    for(size_t i = 0; i < n; i++) {
        x_new[i] = x_prev[i] + alpha * p_prev[i];
    }
    return x_new;
}


double* r(double* r_prev, double* r_new, double alpha, double* c, int n) {
    for(size_t i = 0; i < n; i++) {
        r_new[i] = r_prev[i] - alpha * c[i];
    }
    return r_new;
}


double Beta(double* r_prev, double* r_new, int n) {
  double res1 = 0, res2 = 0;
  for(size_t i = 0; i < n; i++) {
      res1 += r_prev[i] * r_prev[i];
      res2 += r_new[i] * r_new[i];
  }
  return res2 / res1;
}


double* p(double* r_new, double beta, double* p_prev, double* p_new, int n) {
  for(size_t i = 0; i < n; i++) {
      p_new[i] = r_new[i] + beta * p_prev[i];
  }
  return p_new;
}


double Norm(double* r_new, int n) {
  double res = 0;
  for(size_t i = 0; i < n; i++) {
    res += r_new[i] * r_new[i];
  }
  return sqrt(res);
}



int main()
{
    const int N = 662;
    std::ifstream file("662_bus.mtx"); // открываем файл для чтения
    int n, m, nnz; // размерность матрицы и количество ненулевых элементов
    int maxIter = 5000;
    double alpha, beta, eps = 0.000001;
    if(file.is_open()) {
        std::cout << "File status: OPEN." << std::endl;
        file >> n >> m >> nnz; // считываем размерность матрицы и количество ненулевых элементов
        std::cout << "Size matrix: " << n << " x " << m << std::endl;
        std::cout << "Non zero values: " << nnz * 2 - n << std::endl;
    }
    else {
        std::cout << "Error. File not open.";
    }

    // Создание хост-матрицы и хост-вектора
    double a_matrix[N][N];
    double b[N]; // вектор правой части
    double c[N]; // вектор-результат

    // Формирование входной матрицы и вектора-результата
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a_matrix[i][j] = 0;
            c[j] = 0;
        }
    }
    int row, col;
    double value;
    for (int i = 0; i < nnz; i++) {
        file >> row >> col >> value; // считываем строку, столбец и значение элемента
        if(row != col) {
            a_matrix[row-1][col-1] = value;
            a_matrix[col-1][row-1] = value;
        }
        else {
            a_matrix[row-1][col-1] = value;
        }
    }
    file.close();

    // Формирование вектора правой части как сумма элементов строки
    double res = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            res += a_matrix[i][j];
        }
        b[i] = res;
        res = 0;
    }

   


    // Выделение памяти на устройстве для матрицы, вектора и результата
    double* deviceMatrix;
    double* deviceVector;
    double* deviceResult;
    cudaMalloc((void**)&deviceMatrix, N * N * sizeof(double));
    cudaMalloc((void**)&deviceVector, N * sizeof(double));
    cudaMalloc((void**)&deviceResult, N * sizeof(double));

    


    /*=======================НАЧАЛО РЕШЕНИЯ=============================*/
    // Выбор начального приближения

    double* r_prev = new double[N];
    double* r_new = new double[N];
    double* p_prev = new double[N];
    double* p_new = new double[N];
    double* x_prev = new double[N];
    double* x_new = new double[N];
    



    for(size_t i = 0; i < N; i++) { 
        r_prev[i] = b[i]; //r0 = b
        p_prev[i] = r_prev[i]; //p0 = r0
        x_prev[i] = 0; //x0 = [0, 0 ... 0, 0]
    }
    
   
    cudaEvent_t start, stop;
    float gpu_time = 0.0; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);
    for(size_t i = 1; i < maxIter; i++) {
         // Копирование данных из хоста в память на устройстве
        cudaMemcpy(deviceMatrix, a_matrix, N * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceVector, p_prev, N * sizeof(double), cudaMemcpyHostToDevice);
        // Вычисление матрицы-вектора произведения
        MatrVecMul<<<1, N>>>(deviceMatrix, deviceVector, deviceResult, N);
        // Копирование результата обратно на хост
        cudaMemcpy(c, deviceResult, N * sizeof(double), cudaMemcpyDeviceToHost);
        alpha = AlphaCalc(r_prev, p_prev, c, N);
        x_new = x(x_prev, alpha, p_prev, x_new, N);
        r_new = r(r_prev, r_new, alpha, c, N);
        beta = Beta(r_prev, r_new, N);
        if(Norm(r_new, N) < eps) {
            std::cout << "Количество итераций = " << i << std::endl;
            break;
        }
        if(i % 100 == 0){ // Вывод нормы вектора r каждые 100 шагов
            std::cout << "Norm [" << i << "]: " << Norm(r_new, N) << std::endl;
        }
        p_new = p(r_new, beta, p_prev, p_new, N); 
        std::copy_n(p_new, N, p_prev);
        std::copy_n(r_new, N, r_prev);
        std::copy_n(x_new, N, x_prev);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Time of method: %.6f seconds\n", gpu_time * 0.001);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    std::cout << "Answer => x = ";
    for (int i = 0; i < N; i++) {
        std::cout << x_new[i] << ' ';
    }
    std::cout << std::endl;

    
     // Освобождение памяти на устройстве
    cudaFree(deviceMatrix);
    cudaFree(deviceVector);
    cudaFree(deviceResult);
    delete [] r_prev;
    delete [] p_prev;
    delete [] x_new;
    delete [] x_prev;
    delete [] r_new;
    delete [] p_new;
    return 0;
}