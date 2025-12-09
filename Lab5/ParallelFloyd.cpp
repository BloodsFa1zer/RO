// ParallelFloyd.cpp
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <string>
#include <mpi.h>

const int NO_EDGE = -1;
const double INFINITY_PERCENT = 50.0;

int ProcRank = 0;
int ProcNum = 1;

// ===== Допоміжні функції (спільні і для серійної, і для паралельної реалізацій) =====

int MinDistance(int a, int b) {
    if (a == NO_EDGE && b == NO_EDGE) return NO_EDGE;
    if (a == NO_EDGE) return b;
    if (b == NO_EDGE) return a;
    return (a < b) ? a : b;
}

void PrintMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::printf("%7d", matrix[i * cols + j]);
        }
        std::printf("\n");
    }
}

// Той самий приклад ініціалізації, що й у серійній версії
void DummyDataInitialization(int *matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = i; j < size; ++j) {
            if (i == j) {
                matrix[i * size + j] = 0;
            } else if (i == 0) {
                matrix[i * size + j] = j;
            } else {
                matrix[i * size + j] = NO_EDGE;
            }
        }
    }

    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            matrix[j * size + i] = matrix[i * size + j];
        }
    }
}

void RandomDataInitialization(int *matrix, int size) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i == j) {
                matrix[i * size + j] = 0;
            } else {
                int r = std::rand() % 100;
                if (r < INFINITY_PERCENT) {
                    matrix[i * size + j] = NO_EDGE;
                } else {
                    matrix[i * size + j] = (std::rand() % 10) + 1;
                }
            }
        }
    }
}

// Серійний Floyd для перевірки результатів паралельного алгоритму
void SerialFloyd(int *matrix, int size) {
    for (int k = 0; k < size; ++k) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (matrix[i * size + k] == NO_EDGE ||
                    matrix[k * size + j] == NO_EDGE) {
                    continue;
                }
                int t1 = matrix[i * size + j];
                int t2 = matrix[i * size + k] + matrix[k * size + j];
                matrix[i * size + j] = MinDistance(t1, t2);
            }
        }
    }
}

// ===== Розклад рядків між процесами =====

void ComputeRowsLayout(int size,
                       std::vector<int> &rowsPerProc,
                       std::vector<int> &rowDispls) {
    rowsPerProc.resize(ProcNum);
    rowDispls.resize(ProcNum);

    int rest = size;
    int offset = 0;

    for (int r = 0; r < ProcNum; ++r) {
        int count = rest / (ProcNum - r);
        rowsPerProc[r] = count;
        rowDispls[r] = offset;
        offset += count;
        rest -= count;
    }
}

// ===== Ініціалізація / завершення =====

void ProcessInitialization(int *&fullMatrix,
                           int *&localRows,
                           int &size,
                           int &rowNum,
                           bool quiet = false) {
    if (ProcRank == 0) {
        fullMatrix = new int[size * size];

        // Той самий приклад, що й у серійному варіанті
        DummyDataInitialization(fullMatrix, size);
        // Або випадкова матриця:
        // RandomDataInitialization(fullMatrix, size);
        
        if (!quiet) {
            std::printf("Using graph with %d vertices\n", size);
        }
    }

    // Передаємо розмір усім процесам
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> rowsPerProc;
    std::vector<int> rowDispls;
    ComputeRowsLayout(size, rowsPerProc, rowDispls);

    rowNum = rowsPerProc[ProcRank];
    localRows = new int[rowNum * size];
}

void ProcessTermination(int *fullMatrix, int *localRows) {
    if (ProcRank == 0) {
        delete [] fullMatrix;
    }
    delete [] localRows;
}

// ===== Розподіл / збирання даних =====

void DataDistribution(int *fullMatrix,
                      int *localRows,
                      int size,
                      int rowNum) {
    std::vector<int> rowsPerProc;
    std::vector<int> rowDispls;
    ComputeRowsLayout(size, rowsPerProc, rowDispls);

    std::vector<int> sendCounts(ProcNum);
    std::vector<int> sendDispls(ProcNum);

    for (int r = 0; r < ProcNum; ++r) {
        sendCounts[r] = rowsPerProc[r] * size;
        sendDispls[r] = rowDispls[r] * size;
    }

    MPI_Scatterv(fullMatrix,
                 sendCounts.data(),
                 sendDispls.data(),
                 MPI_INT,
                 localRows,
                 rowNum * size,
                 MPI_INT,
                 0,
                 MPI_COMM_WORLD);
}

void ResultCollection(int *fullMatrix,
                      int *localRows,
                      int size,
                      int rowNum) {
    std::vector<int> rowsPerProc;
    std::vector<int> rowDispls;
    ComputeRowsLayout(size, rowsPerProc, rowDispls);

    std::vector<int> recvCounts(ProcNum);
    std::vector<int> recvDispls(ProcNum);

    for (int r = 0; r < ProcNum; ++r) {
        recvCounts[r] = rowsPerProc[r] * size;
        recvDispls[r] = rowDispls[r] * size;
    }

    MPI_Gatherv(localRows,
                rowNum * size,
                MPI_INT,
                fullMatrix,
                recvCounts.data(),
                recvDispls.data(),
                MPI_INT,
                0,
                MPI_COMM_WORLD);
}

// ===== Розсилка рядка k усім процесам =====

void RowDistribution(const int *localRows,
                     int size,
                     int rowNum,
                     int k,
                     int *rowBuffer,
                     const std::vector<int> &rowsPerProc,
                     const std::vector<int> &rowDispls) {
    int ownerRank = 0;
    for (int r = 0; r < ProcNum; ++r) {
        if (k >= rowDispls[r] && k < rowDispls[r] + rowsPerProc[r]) {
            ownerRank = r;
            break;
        }
    }

    if (ProcRank == ownerRank) {
        int localIndex = k - rowDispls[ProcRank];
        std::copy(localRows + localIndex * size,
                  localRows + (localIndex + 1) * size,
                  rowBuffer);
    }

    MPI_Bcast(rowBuffer, size, MPI_INT, ownerRank, MPI_COMM_WORLD);
}

// ===== Паралельний Floyd =====

void ParallelFloyd(int *localRows, int size, int rowNum) {
    std::vector<int> rowsPerProc;
    std::vector<int> rowDispls;
    ComputeRowsLayout(size, rowsPerProc, rowDispls);

    std::vector<int> rowBuffer(size);

    for (int k = 0; k < size; ++k) {
        RowDistribution(localRows,
                        size,
                        rowNum,
                        k,
                        rowBuffer.data(),
                        rowsPerProc,
                        rowDispls);

        for (int i = 0; i < rowNum; ++i) {
            int ikIndex = i * size + k;
            if (localRows[ikIndex] == NO_EDGE) continue;

            for (int j = 0; j < size; ++j) {
                if (rowBuffer[j] == NO_EDGE) continue;

                int current = localRows[i * size + j];
                int viaK = localRows[ikIndex] + rowBuffer[j];
                localRows[i * size + j] = MinDistance(current, viaK);
            }
        }
    }
}

// ===== Вивід та тестування =====

void ParallelPrintMatrix(int *localRows, int size, int rowNum) {
    for (int r = 0; r < ProcNum; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (ProcRank == r) {
            std::printf("ProcRank = %d\n", ProcRank);
            std::printf("Proc rows:\n");
            PrintMatrix(localRows, rowNum, size);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void TestDistribution(int *fullMatrix,
                      int *localRows,
                      int size,
                      int rowNum) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (ProcRank == 0) {
        std::printf("Initial adjacency matrix:\n");
        PrintMatrix(fullMatrix, size, size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ParallelPrintMatrix(localRows, size, rowNum);
}

bool CompareMatrices(const int *a, const int *b, int size) {
    for (int i = 0; i < size * size; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void TestResult(int *fullMatrix, int *serialMatrix, int size) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (ProcRank == 0) {
        SerialFloyd(serialMatrix, size);
        if (CompareMatrices(fullMatrix, serialMatrix, size)) {
            std::printf("Results of serial and parallel algorithms are identical\n");
        } else {
            std::printf("Results of serial and parallel algorithms are NOT identical. Check your code.\n");
        }
    }
}

// ===== main =====

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    int size = 0;
    bool quiet = false;

    // Перевірка аргументів командного рядка
    if (ProcRank == 0) {
        if (argc >= 2) {
            size = std::atoi(argv[1]);
            if (size < ProcNum) {
                std::fprintf(stderr, "Error: Size must be >= number of processes (%d)\n", ProcNum);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            quiet = (argc >= 3 && std::string(argv[2]) == "quiet");
        } else {
            std::printf("Parallel Floyd algorithm\n");
            do {
                std::printf("Enter the number of vertices: ");
                std::scanf("%d", &size);
                if (size < ProcNum) {
                    std::printf("The number of vertices should be greater "
                                "than or equal to the number of processes\n");
                }
            } while (size < ProcNum);
        }
    }

    // Передаємо розмір усім процесам
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *fullMatrix = nullptr;
    int *localRows = nullptr;
    int rowNum = 0;

    ProcessInitialization(fullMatrix, localRows, size, rowNum, quiet);

    int *serialMatrix = nullptr;
    if (ProcRank == 0) {
        serialMatrix = new int[size * size];
        std::copy(fullMatrix, fullMatrix + size * size, serialMatrix);
    }

    double start = MPI_Wtime();

    DataDistribution(fullMatrix, localRows, size, rowNum);
    //TestDistribution(fullMatrix, localRows, size, rowNum); // для налагодження

    ParallelFloyd(localRows, size, rowNum);

    ResultCollection(fullMatrix, localRows, size, rowNum);

    // Якщо треба — можна вивести результат:
    // if (ProcRank == 0 && !quiet && size <= 10) {
    //     PrintMatrix(fullMatrix, size, size);
    // }

    // Перевірка результату (опційно)
    //TestResult(fullMatrix, serialMatrix, size);

    double finish = MPI_Wtime();

    if (ProcRank == 0) {
        if (quiet) {
            std::printf("%d %f\n", size, finish - start);
        } else {
            std::printf("Time of execution: %f\n", finish - start);
        }
        delete [] serialMatrix;
    }

    ProcessTermination(fullMatrix, localRows);

    MPI_Finalize();
    return 0;
}
