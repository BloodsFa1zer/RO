#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <cstdlib>

const double TOL = 1e-6;

// Dummy data initialization - simple data for correctness check
void DummyDataInitialization(std::vector<double>& A, std::vector<double>& b, int n) {
    A.resize(n * n);
    b.resize(n);
    
    // Lower triangular matrix with ones on diagonal
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j <= i) {
                A[i * n + j] = (i == j) ? 1.0 : static_cast<double>(i - j + 1);
            } else {
                A[i * n + j] = 0.0;
            }
        }
        // b chosen so that solution is vector of ones
        b[i] = static_cast<double>(i + 1) * (i + 2) / 2.0;
    }
}

// Random data initialization - for time measurements
void RandomDataInitialization(std::vector<double>& A, std::vector<double>& b, int n) {
    A.resize(n * n);
    b.resize(n);
    
    srand(static_cast<unsigned>(time(nullptr)));
    
    // Lower triangular matrix with random values, non-zero diagonal
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j <= i) {
                A[i * n + j] = static_cast<double>(rand()) / RAND_MAX * 10.0 + 1.0;
            } else {
                A[i * n + j] = 0.0;
            }
        }
        // Ensure diagonal is non-zero
        if (std::abs(A[i * n + i]) < 1e-10) {
            A[i * n + i] = 1.0;
        }
        b[i] = static_cast<double>(rand()) / RAND_MAX * 10.0;
    }
}

// Find pivot row - row with maximum absolute value in column k
int FindPivotRow(const std::vector<double>& A, int n, int k, const std::vector<bool>& used) {
    int pivotRow = k;
    double maxVal = std::abs(A[k * n + k]);
    
    for (int i = k + 1; i < n; ++i) {
        if (!used[i]) {
            double val = std::abs(A[i * n + k]);
            if (val > maxVal) {
                maxVal = val;
                pivotRow = i;
            }
        }
    }
    
    return pivotRow;
}

// Swap two rows in matrix A and vector b
void SwapRows(std::vector<double>& A, std::vector<double>& b, int n, int row1, int row2) {
    if (row1 == row2) return;
    
    // Swap rows in matrix
    for (int j = 0; j < n; ++j) {
        std::swap(A[row1 * n + j], A[row2 * n + j]);
    }
    
    // Swap elements in vector
    std::swap(b[row1], b[row2]);
}

// Serial column elimination (Gaussian elimination step)
void SerialColumnElimination(std::vector<double>& A, std::vector<double>& b, int n, int k) {
    double pivot = A[k * n + k];
    
    if (std::abs(pivot) < TOL) {
        std::cerr << "Warning: Pivot element is too small at column " << k << std::endl;
        return;
    }
    
    // Normalize pivot row
    for (int j = k; j < n; ++j) {
        A[k * n + j] /= pivot;
    }
    b[k] /= pivot;
    
    // Eliminate column below pivot
    for (int i = k + 1; i < n; ++i) {
        double factor = A[i * n + k];
        for (int j = k; j < n; ++j) {
            A[i * n + j] -= factor * A[k * n + j];
        }
        b[i] -= factor * b[k];
    }
}

// Back substitution
void BackSubstitution(const std::vector<double>& A, const std::vector<double>& b, 
                      std::vector<double>& x, int n) {
    x.resize(n);
    
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i * n + j] * x[j];
        }
        x[i] = sum / A[i * n + i];
    }
}

// Serial Gaussian elimination with partial pivoting
void SerialGaussianElimination(std::vector<double>& A, std::vector<double>& b, int n) {
    std::vector<bool> used(n, false);
    
    for (int k = 0; k < n; ++k) {
        // Find pivot row
        int pivotRow = FindPivotRow(A, n, k, used);
        
        // Swap rows if needed
        if (pivotRow != k) {
            SwapRows(A, b, n, k, pivotRow);
        }
        
        used[k] = true;
        
        // Perform elimination
        SerialColumnElimination(A, b, n, k);
    }
}

// Serial result calculation (main function)
void SerialResultCalculation(std::vector<double>& A, std::vector<double>& b, 
                            std::vector<double>& x, int n) {
    // Make copies to preserve original data
    std::vector<double> A_copy = A;
    std::vector<double> b_copy = b;
    
    // Forward elimination
    SerialGaussianElimination(A_copy, b_copy, n);
    
    // Back substitution
    BackSubstitution(A_copy, b_copy, x, n);
}

// Check result correctness (for dummy data)
bool CheckResult(const std::vector<double>& x, int n) {
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        double diff = std::abs(x[i] - 1.0);
        if (diff > TOL) {
            std::cout << "Error at index " << i << ": expected 1.0, got " << x[i] 
                      << ", diff = " << diff << std::endl;
            correct = false;
        }
    }
    return correct;
}

int main() {
    int n = 0;
    
    std::cout << "Serial Gaussian elimination method\n";
    
    // Input with validation
    while (true) {
        std::cout << "Enter system size (n > 0): ";
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
    
    std::vector<double> A, b, x;
    
    // Choose initialization type
    int initType = 0;
    std::cout << "Choose initialization:\n";
    std::cout << "1 - Dummy data (for correctness check)\n";
    std::cout << "2 - Random data (for time measurements)\n";
    std::cout << "Enter choice (1 or 2): ";
    std::cin >> initType;
    
    if (initType == 1) {
        DummyDataInitialization(A, b, n);
        std::cout << "Using dummy data initialization.\n";
    } else {
        RandomDataInitialization(A, b, n);
        std::cout << "Using random data initialization.\n";
    }
    
    std::cout << "Initial data prepared.\n";
    
    // Measure time
    clock_t start = clock();
    
    SerialResultCalculation(A, b, x, n);
    
    clock_t finish = clock();
    double duration = (finish - start) / double(CLOCKS_PER_SEC);
    
    std::cout << "\nTime of execution: " << duration << " seconds\n";
    
    // Check result for dummy data
    if (initType == 1) {
        if (CheckResult(x, n)) {
            std::cout << "\n=== Result is correct (all values are 1.0) ===\n";
        } else {
            std::cout << "\n*** Result is incorrect ***\n";
        }
    }
    
    // Print result for small systems
    if (n <= 10) {
        std::cout << "\nSolution vector x:\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "x[" << i << "] = " << x[i] << "\n";
        }
    }
    
    return 0;
}
