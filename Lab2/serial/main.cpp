#include <iostream>
#include <vector>
#include <ctime>

int main() {
    int n = 0;

    std::cout << "Serial matrix-matrix multiplication\n";

    // Ввід розміру n
    while (true) {
        std::cout << "Enter matrix size n (n > 0): ";
        std::cin >> n;

        if (!std::cin.good()) {
            std::cin.clear();
            std::cin.ignore(32767, '\n');
            std::cout << "Input error. Please enter a positive integer.\n";
            continue;
        }
        if (n <= 0) {
            std::cout << "Size must be positive.\n";
            continue;
        }
        break;
    }

    std::cout << "Using n = " << n << "\n";

    // Матриці A, B, C (n x n)
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> C(n * n);

    // Ініціалізація:
    // A[i,j] = i + j + 1
    // B[i,j] = (i == j ? 1 : 0)  (майже одинична)
    int i, j, k;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            A[i * n + j] = static_cast<double>(i + j + 1);
            B[i * n + j] = (i == j) ? 1.0 : 0.0;
            C[i * n + j] = 0.0;
        }
    }

    std::cout << "Initial matrices prepared.\n";

    // Вимір часу множення
    clock_t t_start = clock();

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            double sum = 0.0;
            for (k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }

    clock_t t_finish = clock();
    double elapsed = (double)(t_finish - t_start) / CLOCKS_PER_SEC;

    std::cout << "Computation finished.\n";
    std::cout << "Execution time (serial): " << elapsed << " s\n";

    if (n <= 5) {
        std::cout << "Result matrix C:\n";
        for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                std::cout << C[i * n + j] << "\t";
            }
            std::cout << "\n";
        }
    }

    std::cout << "Program completed successfully.\n";
    return 0;
}
