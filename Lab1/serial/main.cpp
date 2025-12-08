#include <iostream>
#include <vector>
#include <ctime>

int main() {
    int n = 0;

    std::cout << "Serial matrix-vector multiplication\n";

    // simple input with validation
    while (true) {
        std::cout << "Enter matrix/vector size (n > 0): ";
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

    std::vector<double> A(n * n);
    std::vector<double> x(n);
    std::vector<double> y(n);

    // Same initialization as in MPI version
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0;
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = static_cast<double>(i + j + 1);
        }
    }

    std::cout << "Initial data prepared.\n";

    clock_t t_start = clock();

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        const double* row = &A[i * n];
        for (int j = 0; j < n; ++j) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }

    clock_t t_finish = clock();
    double elapsed = (double)(t_finish - t_start) / CLOCKS_PER_SEC;

    std::cout << "Computation finished.\n";
    std::cout << "Execution time: " << elapsed << " s\n";

    if (n <= 10) {
        std::cout << "Result vector:\n";
        for (int i = 0; i < n; ++i) {
            std::cout << y[i] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Program completed successfully.\n";
    return 0;
}
