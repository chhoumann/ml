#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <mkl.h> // sudo apt-get install intel-mkl

#define TILE_SIZE 32

// Function to multiply two matrices A and B
std::vector<std::vector<double>> multiply(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &B) {
    std::size_t n = A.size();
    std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));

    #pragma omp parallel for
    for (std::size_t ii = 0; ii < n; ii+=TILE_SIZE) {
        for (std::size_t jj = 0; jj < n; jj+=TILE_SIZE) {
            for (std::size_t kk = 0; kk < n; kk+=TILE_SIZE) {
                for (std::size_t i = ii; i < std::min(ii+TILE_SIZE,n); i++) {
                    for (std::size_t j = jj; j < std::min(jj+TILE_SIZE,n); j++) {
                        for (std::size_t k = kk; k < std::min(kk+TILE_SIZE,n); k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }

    return C;
}

int mkl() {
    std::size_t n = 1024; // Set matrix size
    double alpha = 1.0;
    double beta = 0.0;

    double *A = (double *)mkl_malloc(n * n * sizeof(double), 64); // Initialize matrix A with 1.0
    double *B = (double *)mkl_malloc(n * n * sizeof(double), 64); // Initialize matrix B with 1.0
    double *C = (double *)mkl_malloc(n * n * sizeof(double), 64); // Result matrix

    for (std::size_t i = 0; i < n * n; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                n, n, n, alpha, A, n, B, n, beta, C, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double gflops = 2.0 * n * n * n / diff.count() / 1e12;

    std::cout << "Time: " << diff.count() << "s\n";
    std::cout << "MKL TFLOPS: " << gflops << "\n";

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}

int main() {
    std::size_t n = 1024; // Set matrix size
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0)); // Initialize matrix A with 1.0
    std::vector<std::vector<double>> B(n, std::vector<double>(n, 1.0)); // Initialize matrix B with 1.0

    auto start = std::chrono::high_resolution_clock::now();
    auto C = multiply(A, B);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double tflops = 2.0 * n * n * n / diff.count() / 1e12;

    std::cout << "Time: " << diff.count() << "s\n";
    std::cout << "OMP TFLOPS: " << tflops << "\n";

    /*
    Time: 12.0888s
    OMP TFLOPS: 0.000177643
    Time: 0.0190309s
    MKL TFLOPS: 0.112842

    Which is super dissapointing because numpy gets 0.59 TFLOPS on my machine.
    */

    mkl();
    return 0;
}
