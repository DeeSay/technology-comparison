#include <iostream>
#include <limits>
#include <fstream>
#include <cmath>
#include <time.h>
#include <algorithm>
#include <iterator>
#include <cstring>
#include <cstdlib>
#include <omp.h>


double* MatrVecMul(double** a_matrix, double* vec, double* c, int N) {
  double sum;
  int threadsNum = 4;
  omp_set_num_threads(threadsNum);
  #pragma omp parallel for private(sum) 
    for (int i = 0; i < N; i++) {
      sum = 0;
      for (int j = 0; j < N; j++) {
          sum += a_matrix[i][j] * vec[j];
      }
      c[i] = sum;
    }
  
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




int main(int argc, char const *argv[])
{
  std::ifstream file("nos3.mtx"); // открываем файл для чтения
  int N, m, nnz; // размерность матрицы и количество ненулевых элементов
  int maxIter = 5000;
  double alpha, beta, eps = 0.000001;
  if(file.is_open()) {
      std::cout << "File status: OPEN." << std::endl;
      file >> N >> m >> nnz; // считываем размерность матрицы и количество ненулевых элементов
      nnz = nnz * 2 - N;
      std::cout << "Size matrix: " << N << " x " << m << std::endl;
      std::cout << "Non zero values: " << nnz << std::endl;
  }
  else {
      std::cout << "Error. File not open.";
  }

  double **a_matrix = new double *[N];
  double *b = new double[N]; // Вектор правой части
  double *c = new double[N]; // Вектор результат M x V
  for (int i = 0; i < N; i++)
  {
      a_matrix[i] = new double[N];
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
  
  double* r_prev = new double[N];
  double* r_new = new double[N];
  double* p_prev = new double[N];
  double* p_new = new double[N];
  double* x_prev = new double[N];
  double* x_new = new double[N];
  double* k_ptr;
  

 
  for(size_t i = 0; i < N; i++) { 
    r_prev[i] = b[i]; //r0 = b
    p_prev[i] = r_prev[i]; //p0 = r0
    x_prev[i] = 0; //x0 = [0, 0 ... 0, 0]
  }

  
  double start_time, end_time;
  start_time = omp_get_wtime(); 
  for(size_t i = 1; i < maxIter; i++) {
    k_ptr = MatrVecMul(a_matrix, p_prev, c, N);
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
  
  end_time = omp_get_wtime();
  double tt = end_time - start_time;
  std::cout << "Total time = " << tt << std::endl;
  

  std::cout << "Answer => x = ";
  for (int i = 0; i < N; i++) {
    std::cout << x_new[i] << ' ';
  }
  std::cout << std::endl;


for (int i = 0; i < N; i++) {
    delete [] a_matrix[i];
}
delete [] a_matrix;
delete [] b;
delete [] c;
delete [] r_prev;
delete [] p_prev;
delete [] x_new;
delete [] x_prev;
delete [] r_new;
delete [] p_new;
return 0;
}

