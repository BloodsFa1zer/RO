#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <ctime>

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
    
    srand(static_cast<unsigned>(time(nullptr)) + 42);
    
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

// Compute row distribution among processes
void ComputeRowDistribution(int n, int world_size, 
                           std::vector<int>& pProcNum, 
                           std::vector<int>& pProcInd) {
    pProcNum.resize(world_size);
    pProcInd.resize(world_size);
    
    int base = n / world_size;
    int extra = n % world_size;
    
    pProcInd[0] = 0;
    for (int i = 0; i < world_size; ++i) {
        pProcNum[i] = base + (i < extra ? 1 : 0);
        if (i > 0) {
            pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
        }
    }
}

// Find which process owns a given row
int FindProcessForRow(int row, const std::vector<int>& pProcInd, int world_size) {
    for (int i = 0; i < world_size - 1; ++i) {
        if (row >= pProcInd[i] && row < pProcInd[i + 1]) {
            return i;
        }
    }
    return world_size - 1;
}

// Data distribution using MPI_Scatterv
void DataDistribution(const std::vector<double>& pMatrix,
                     std::vector<double>& pProcRows,
                     const std::vector<double>& pVector,
                     std::vector<double>& pProcVector,
                     int Size,
                     int RowNum,
                     const std::vector<int>& pProcNum,
                     const std::vector<int>& pProcInd,
                     int ProcRank,
                     int ProcNum) {
    // Prepare send counts and displacements for matrix
    std::vector<int> sendCounts(ProcNum);
    std::vector<int> sendDispls(ProcNum);
    
    for (int i = 0; i < ProcNum; ++i) {
        sendCounts[i] = pProcNum[i] * Size;
        sendDispls[i] = pProcInd[i] * Size;
    }
    
    pProcRows.resize(RowNum * Size);
    MPI_Scatterv(const_cast<double*>(pMatrix.data()), sendCounts.data(), sendDispls.data(),
                 MPI_DOUBLE, pProcRows.data(), RowNum * Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Prepare send counts and displacements for vector
    std::vector<int> vecSendCounts(ProcNum);
    std::vector<int> vecSendDispls(ProcNum);
    
    for (int i = 0; i < ProcNum; ++i) {
        vecSendCounts[i] = pProcNum[i];
        vecSendDispls[i] = pProcInd[i];
    }
    
    pProcVector.resize(RowNum);
    MPI_Scatterv(const_cast<double*>(pVector.data()), vecSendCounts.data(), vecSendDispls.data(),
                 MPI_DOUBLE, pProcVector.data(), RowNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Parallel Gaussian elimination - forward elimination
void ParallelGaussianElimination(std::vector<double>& pProcRows,
                                 std::vector<double>& pProcVector,
                                 int Size,
                                 int RowNum,
                                 const std::vector<int>& pProcNum,
                                 const std::vector<int>& pProcInd,
                                 int ProcRank,
                                 int ProcNum) {
    std::vector<double> pivotRow(Size + 1); // Row + b element
    
    for (int k = 0; k < Size; ++k) {
        // Find process that owns row k
        int pivotProc = FindProcessForRow(k, pProcInd, ProcNum);
        
        // Get local row index on pivot process
        int localRow = k - pProcInd[pivotProc];
        
        // Process that owns pivot row normalizes it and prepares for broadcast
        if (ProcRank == pivotProc) {
            double pivot = pProcRows[localRow * Size + k];
            if (std::abs(pivot) < TOL) {
                if (ProcRank == 0) {
                    std::cerr << "Warning: Pivot element is too small at column " << k << std::endl;
                }
            } else {
                // Normalize pivot row
                for (int j = k; j < Size; ++j) {
                    pivotRow[j] = pProcRows[localRow * Size + j] / pivot;
                }
                pivotRow[Size] = pProcVector[localRow] / pivot;
            }
        }
        
        // Broadcast pivot row to all processes
        MPI_Bcast(pivotRow.data(), Size + 1, MPI_DOUBLE, pivotProc, MPI_COMM_WORLD);
        
        // Each process eliminates in its local rows
        for (int i = 0; i < RowNum; ++i) {
            int globalRow = pProcInd[ProcRank] + i;
            if (globalRow > k) {
                double factor = pProcRows[i * Size + k];
                for (int j = k; j < Size; ++j) {
                    pProcRows[i * Size + j] -= factor * pivotRow[j];
                }
                pProcVector[i] -= factor * pivotRow[Size];
            } else if (globalRow == k && ProcRank == pivotProc) {
                // Update pivot row itself
                for (int j = k; j < Size; ++j) {
                    pProcRows[i * Size + j] = pivotRow[j];
                }
                pProcVector[i] = pivotRow[Size];
            }
        }
    }
}

// Parallel back substitution
void ParallelBackSubstitution(const std::vector<double>& pProcRows,
                              std::vector<double>& pProcVector,
                              std::vector<double>& pProcResult,
                              int Size,
                              int RowNum,
                              const std::vector<int>& pProcNum,
                              const std::vector<int>& pProcInd,
                              int ProcRank,
                              int ProcNum) {
    pProcResult.resize(RowNum);
    std::vector<double> x(Size, 0.0);
    
    for (int k = Size - 1; k >= 0; --k) {
        // Find process that owns row k
        int pivotProc = FindProcessForRow(k, pProcInd, ProcNum);
        int localRow = k - pProcInd[pivotProc];
        
        double xk = 0.0;
        
        // Process that owns row k computes x[k]
        if (ProcRank == pivotProc) {
            double sum = pProcVector[localRow];
            for (int j = k + 1; j < Size; ++j) {
                sum -= pProcRows[localRow * Size + j] * x[j];
            }
            xk = sum / pProcRows[localRow * Size + k];
            x[k] = xk;
        }
        
        // Broadcast x[k] to all processes
        MPI_Bcast(&xk, 1, MPI_DOUBLE, pivotProc, MPI_COMM_WORLD);
        x[k] = xk;
        
        // Each process updates its local b vector
        for (int i = 0; i < RowNum; ++i) {
            int globalRow = pProcInd[ProcRank] + i;
            if (globalRow < k) {
                pProcVector[i] -= pProcRows[i * Size + k] * xk;
            }
        }
    }
    
    // Store local parts of solution
    for (int i = 0; i < RowNum; ++i) {
        int globalRow = pProcInd[ProcRank] + i;
        pProcResult[i] = x[globalRow];
    }
}

// Parallel result calculation
void ParallelResultCalculation(std::vector<double>& pProcRows,
                              std::vector<double>& pProcVector,
                              std::vector<double>& pProcResult,
                              int Size,
                              int RowNum,
                              const std::vector<int>& pProcNum,
                              const std::vector<int>& pProcInd,
                              int ProcRank,
                              int ProcNum) {
    // Forward elimination
    ParallelGaussianElimination(pProcRows, pProcVector, Size, RowNum,
                                pProcNum, pProcInd, ProcRank, ProcNum);
    
    // Back substitution
    ParallelBackSubstitution(pProcRows, pProcVector, pProcResult, Size, RowNum,
                            pProcNum, pProcInd, ProcRank, ProcNum);
}

// Result collection using MPI_Gatherv
void ResultCollection(const std::vector<double>& pProcResult,
                     std::vector<double>& pResult,
                     int Size,
                     int RowNum,
                     const std::vector<int>& pProcNum,
                     const std::vector<int>& pProcInd,
                     int ProcRank,
                     int ProcNum) {
    // Prepare receive counts and displacements
    std::vector<int> recvCounts(ProcNum);
    std::vector<int> recvDispls(ProcNum);
    
    for (int i = 0; i < ProcNum; ++i) {
        recvCounts[i] = pProcNum[i];
        recvDispls[i] = pProcInd[i];
    }
    
    if (ProcRank == 0) {
        pResult.resize(Size);
    }
    
    MPI_Gatherv(const_cast<double*>(pProcResult.data()), RowNum, MPI_DOUBLE,
                ProcRank == 0 ? pResult.data() : nullptr,
                recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

// Serial version for comparison
void SerialGaussianElimination(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int k = 0; k < n; ++k) {
        // Find pivot
        int pivotRow = k;
        double maxVal = std::abs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A[i * n + k]) > maxVal) {
                maxVal = std::abs(A[i * n + k]);
                pivotRow = i;
            }
        }
        
        // Swap rows
        if (pivotRow != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(A[k * n + j], A[pivotRow * n + j]);
            }
            std::swap(b[k], b[pivotRow]);
        }
        
        // Eliminate
        double pivot = A[k * n + k];
        if (std::abs(pivot) < TOL) continue;
        
        for (int j = k; j < n; ++j) {
            A[k * n + j] /= pivot;
        }
        b[k] /= pivot;
        
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i * n + k];
            for (int j = k; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }
}

void SerialBackSubstitution(const std::vector<double>& A, const std::vector<double>& b,
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

void SerialResultCalculation(std::vector<double>& A, std::vector<double>& b,
                             std::vector<double>& x, int n) {
    std::vector<double> A_copy = A;
    std::vector<double> b_copy = b;
    
    SerialGaussianElimination(A_copy, b_copy, n);
    SerialBackSubstitution(A_copy, b_copy, x, n);
}

// Check result correctness
bool CheckResult(const std::vector<double>& x_par, const std::vector<double>& x_ser, int n) {
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        double diff = std::abs(x_par[i] - x_ser[i]);
        if (diff > TOL) {
            std::cout << "Mismatch at index " << i << ": parallel=" << x_par[i]
                      << ", serial=" << x_ser[i] << ", diff=" << diff << std::endl;
            correct = false;
        }
    }
    return correct;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int ProcNum = 0;
    int ProcRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    
    int n = 0;
    
    if (ProcRank == 0) {
        std::cout << "Parallel Gaussian elimination method (MPI)\n";
        std::cout << "Number of processes: " << ProcNum << "\n";
        
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
    }
    
    // Broadcast n to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Compute row distribution
    std::vector<int> pProcNum, pProcInd;
    if (ProcRank == 0) {
        ComputeRowDistribution(n, ProcNum, pProcNum, pProcInd);
    } else {
        pProcNum.resize(ProcNum);
        pProcInd.resize(ProcNum);
    }
    MPI_Bcast(pProcNum.data(), ProcNum, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(pProcInd.data(), ProcNum, MPI_INT, 0, MPI_COMM_WORLD);
    
    int RowNum = pProcNum[ProcRank];
    
    // Initialize data on rank 0
    int initType = 2; // Default to random
    std::vector<double> pMatrix, pVector;
    if (ProcRank == 0) {
        std::cout << "Choose initialization:\n";
        std::cout << "1 - Dummy data (for correctness check)\n";
        std::cout << "2 - Random data (for time measurements)\n";
        std::cout << "Enter choice (1 or 2): ";
        std::cin >> initType;
        
        if (initType == 1) {
            DummyDataInitialization(pMatrix, pVector, n);
            std::cout << "Using dummy data initialization.\n";
        } else {
            RandomDataInitialization(pMatrix, pVector, n);
            std::cout << "Using random data initialization.\n";
        }
    } else {
        pMatrix.resize(n * n);
        pVector.resize(n);
    }
    
    // Broadcast initialization type (for consistency, though not strictly needed)
    MPI_Bcast(&initType, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Distribute data
    std::vector<double> pProcRows, pProcVector;
    DataDistribution(pMatrix, pProcRows, pVector, pProcVector, n, RowNum,
                    pProcNum, pProcInd, ProcRank, ProcNum);
    
    if (ProcRank == 0) {
        std::cout << "Data distributed.\n";
    }
    
    // Measure time
    MPI_Barrier(MPI_COMM_WORLD);
    double Start = MPI_Wtime();
    
    // Parallel Gaussian elimination
    std::vector<double> pProcResult;
    ParallelResultCalculation(pProcRows, pProcVector, pProcResult, n, RowNum,
                              pProcNum, pProcInd, ProcRank, ProcNum);
    
    // Collect result
    std::vector<double> pResult;
    ResultCollection(pProcResult, pResult, n, RowNum, pProcNum, pProcInd, ProcRank, ProcNum);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double Finish = MPI_Wtime();
    double Duration = Finish - Start;
    
    if (ProcRank == 0) {
        std::cout << "\nTime of execution: " << Duration << " seconds\n";
        
        // Compare with serial version
        std::vector<double> A_ser = pMatrix;
        std::vector<double> b_ser = pVector;
        std::vector<double> x_ser;
        
        clock_t start_ser = clock();
        SerialResultCalculation(A_ser, b_ser, x_ser, n);
        clock_t finish_ser = clock();
        double duration_ser = (finish_ser - start_ser) / double(CLOCKS_PER_SEC);
        
        std::cout << "Serial time: " << duration_ser << " seconds\n";
        std::cout << "Speedup: " << duration_ser / Duration << "\n";
        
        if (CheckResult(pResult, x_ser, n)) {
            std::cout << "\n=== Parallel result matches serial computation ===\n";
        } else {
            std::cout << "\n*** Parallel result differs from serial computation ***\n";
        }
        
        // Print result for small systems
        if (n <= 10) {
            std::cout << "\nSolution vector x:\n";
            for (int i = 0; i < n; ++i) {
                std::cout << "x[" << i << "] = " << pResult[i] << "\n";
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
