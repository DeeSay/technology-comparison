#include <sycl/sycl.hpp>
#include <iostream>
#include <limits>
#include <fstream>
#include <cmath>
#include <time.h>
#include <algorithm>
#include <iterator>
#include <cstring>
#include "dpc_common.hpp"

using namespace sycl;


double* MatrVecMul(queue Q, double** a_matrix, double* vec, double* c, int N) {
  event e = Q.submit([&](auto &h) {
    h.parallel_for(range(N), [=](id<1> idx) {
      double sum = 0;
      for (int k = 0; k < N; k++) {
        sum += a_matrix[idx][k] * vec[k];
      }
      c[idx] = sum;
    });
  });
  Q.wait();

  return c;
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


constexpr size_t N = 960; //Обязательная константа (размер матрицы) для параллельного цикла


int main(int argc, char const *argv[])
{
  std::ifstream file("nos3.mtx"); // открываем файл для чтения
  int n, m, nnz; // размерность матрицы и количество ненулевых элементов
  int maxIter = 5000;
  double alpha, beta, eps = 0.000001;
  if(file.is_open()) {
      std::cout << "File status: OPEN." << std::endl;
      file >> n >> m >> nnz; // считываем размерность матрицы и количество ненулевых элементов
      nnz = nnz * 2 - n;
      std::cout << "Size matrix: " << n << " x " << m << std::endl;
      std::cout << "Non zero values: " << nnz << std::endl;
  }
  else {
      std::cout << "Error. File not open.";
  }

  queue Q(default_selector_v, property::queue::enable_profiling{});
  std::cout << "Device: " << Q.get_device().get_info<info::device::name>() << "\n";
  double **a_matrix = static_cast<double **>(malloc_shared(N * sizeof(double*), Q));
  double *b = static_cast<double *>(malloc_shared(N * sizeof(double), Q)); // Вектор правой части
  double *c = static_cast<double *>(malloc_shared(N * sizeof(double), Q)); // Вектор результат M x V
  for (int i = 0; i < N; i++)
  {
      a_matrix[i] = static_cast<double *>(malloc_shared(N * sizeof(double), Q));
  }

  // Формирование входной матрицы
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a_matrix[i][j] = 0;
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


  /*=======================НАЧАЛО РЕШЕНИЯ=============================*/
  // Выбор начального приближения
  
  double* r_prev = static_cast<double *>(malloc_shared(N * sizeof(double), Q));
  double* r_new = static_cast<double *>(malloc_shared(N * sizeof(double), Q));
  double* p_prev = static_cast<double *>(malloc_shared(N * sizeof(double), Q));
  double* p_new = static_cast<double *>(malloc_shared(N * sizeof(double), Q));
  double* x_prev = static_cast<double *>(malloc_shared(N * sizeof(double), Q));
  double* x_new = static_cast<double *>(malloc_shared(N * sizeof(double), Q));
  double* k_ptr;
  

 
  for(size_t i = 0; i < n; i++) { 
    r_prev[i] = b[i]; //r0 = b
    p_prev[i] = r_prev[i]; //p0 = r0
    x_prev[i] = 0; //x0 = [0, 0 ... 0, 0]
  }

  
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  for(size_t i = 1; i < maxIter; i++) {
    k_ptr = MatrVecMul(Q, a_matrix, p_prev, c, N);
    alpha = AlphaCalc(r_prev, p_prev, c, N);
    x_new = x(x_prev, alpha, p_prev, x_new, N);
    r_new = r(r_prev, r_new, alpha, c, N);
    beta = Beta(r_prev, r_new, n);
    if(Norm(r_new, N) < eps) {
      std::cout << "Количество итераций = " << i << std::endl;
      break;
    }
    if(i % 100 == 0){ // Вывод нормы вектора r каждые 100 шагов
      std::cout << "Norm [" << i << "]: " << Norm(r_new, N) << std::endl;
    }
    p_new = p(r_new, beta, p_prev, p_new, n); 
    std::copy_n(p_new, N, p_prev);
    std::copy_n(r_new, N, r_prev);
    std::copy_n(x_new, N, x_prev);
  }
  
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Total time = " << elapsed_seconds.count() << std::endl;
  

  std::cout << "Answer => x = ";
  for (int i = 0; i < N; i++) {
    std::cout << x_new[i] << ' ';
  }
  std::cout << std::endl;


for (int i = 0; i < N; i++) {
    free(a_matrix[i], Q);
}
free(a_matrix, Q);
free(b, Q);
free(c, Q);
free(r_prev, Q);
free(p_prev, Q);
free(x_new);
free(x_prev);
free(r_new, Q);
free(p_new, Q);
return 0;
}

