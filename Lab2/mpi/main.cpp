#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

void init_global_matrices(std::vector<double>& A,
                          std::vector<double>& B,
                          int n) {
    A.resize(n * n);
    B.resize(n * n);

    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            A[i * n + j] = static_cast<double>(i + j + 1);
            B[i * n + j] = (i == j) ? 1.0 : 0.0; // проста структура для контролю
        }
    }
}

void distribute_matrix(const std::vector<double>& global,
                       std::vector<double>& local,
                       int n,
                       int blockSize,
                       int q,
                       int world_rank,
                       MPI_Comm cart_comm,
                       int tag) {
    int world_size;
    MPI_Comm_size(cart_comm, &world_size);

    local.resize(blockSize * blockSize);

    int coords[2];
    MPI_Status status;

    if (world_rank == 0) {
        // відправляємо blocks усім процесам
        int rank;
        for (rank = 0; rank < world_size; ++rank) {
            MPI_Cart_coords(cart_comm, rank, 2, coords);
            int blockRow = coords[0];
            int blockCol = coords[1];

            std::vector<double> buf(blockSize * blockSize);
            int i, j;
            for (i = 0; i < blockSize; ++i) {
                for (j = 0; j < blockSize; ++j) {
                    int global_i = blockRow * blockSize + i;
                    int global_j = blockCol * blockSize + j;
                    buf[i * blockSize + j] = global[global_i * n + global_j];
                }
            }

            if (rank == 0) {
                local = buf;
            } else {
                MPI_Send(&buf[0], blockSize * blockSize, MPI_DOUBLE,
                         rank, tag, cart_comm);
            }
        }
    } else {
        MPI_Recv(&local[0], blockSize * blockSize, MPI_DOUBLE,
                 0, tag, cart_comm, &status);
    }
}

void collect_matrix(const std::vector<double>& local,
                    std::vector<double>& global,
                    int n,
                    int blockSize,
                    int q,
                    int world_rank,
                    MPI_Comm cart_comm) {
    int world_size;
    MPI_Comm_size(cart_comm, &world_size);

    int coords[2];
    MPI_Status status;

    if (world_rank == 0) {
        global.assign(n * n, 0.0);

        int rank;
        for (rank = 0; rank < world_size; ++rank) {
            std::vector<double> buf;
            if (rank == 0) {
                buf = local;
            } else {
                buf.resize(blockSize * blockSize);
                MPI_Recv(&buf[0], blockSize * blockSize, MPI_DOUBLE,
                         rank, 3, cart_comm, &status);
            }

            MPI_Cart_coords(cart_comm, rank, 2, coords);
            int blockRow = coords[0];
            int blockCol = coords[1];

            int i, j;
            for (i = 0; i < blockSize; ++i) {
                for (j = 0; j < blockSize; ++j) {
                    int global_i = blockRow * blockSize + i;
                    int global_j = blockCol * blockSize + j;
                    global[global_i * n + global_j] =
                        buf[i * blockSize + j];
                }
            }
        }
    } else {
        MPI_Send(&local[0], blockSize * blockSize, MPI_DOUBLE,
                 0, 3, cart_comm);
    }
}

void block_multiply_add(const std::vector<double>& A,
                        const std::vector<double>& B,
                        std::vector<double>& C,
                        int blockSize) {
    int i, j, k;
    for (i = 0; i < blockSize; ++i) {
        for (j = 0; j < blockSize; ++j) {
            double sum = 0.0;
            for (k = 0; k < blockSize; ++k) {
                sum += A[i * blockSize + k] * B[k * blockSize + j];
            }
            C[i * blockSize + j] += sum;
        }
    }
}

void serial_multiply(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C,
                     int n) {
    C.assign(n * n, 0.0);

    int i, j, k;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            double sum = 0.0;
            for (k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

bool compare_matrices(const std::vector<double>& C1,
                      const std::vector<double>& C2,
                      int n,
                      double tol) {
    int total = n * n;
    for (int i = 0; i < total; ++i) {
        double diff = C1[i] - C2[i];
        if (diff < 0) diff = -diff;
        if (diff > tol) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    setvbuf(stdout, NULL, _IONBF, 0);

    if (world_rank == 0) {
        std::cout << "Parallel matrix-matrix multiplication (Fox algorithm, MPI)\n";
    }

    // розмір матриці n
    int n = 0;
    if (world_rank == 0) {
        std::cout << "Enter matrix size n (n > 0): ";
        std::cin >> n;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n <= 0) {
        if (world_rank == 0) {
            std::cout << "Invalid n. Exiting.\n";
        }
        MPI_Finalize();
        return 0;
    }

    // Перевірка, що кількість процесів — квадрат цілого
    int q = (int)std::sqrt((double)world_size);
    if (q * q != world_size) {
        if (world_rank == 0) {
            std::cout << "Number of processes must be a perfect square (q*q).\n";
        }
        MPI_Finalize();
        return 0;
    }

    if (n % q != 0) {
        if (world_rank == 0) {
            std::cout << "Matrix size n must be divisible by q = " << q << ".\n";
        }
        MPI_Finalize();
        return 0;
    }

    int blockSize = n / q;

    // Створення 2D декартової топології
    int dims[2];
    dims[0] = q;
    dims[1] = q;
    int periods[2];
    periods[0] = 1; // по рядках (для циклічного зсуву B)
    periods[1] = 0; // по стовпцях без циклу
    int reorder = 1;

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);
    int row = coords[0];
    int col = coords[1];

    // Комунікатор рядка для broadcast блоків A
    MPI_Comm row_comm;
    MPI_Comm_split(cart_comm, row, col, &row_comm);

    // Глобальні матриці на процесі 0
    std::vector<double> A_global;
    std::vector<double> B_global;
    std::vector<double> C_global;

    if (world_rank == 0) {
        init_global_matrices(A_global, B_global, n);
        std::cout << "Global matrices initialized (n = " << n << ").\n";
    }

    // Розподіл блоків A і B по процесах (checkerboard)
    std::vector<double> A_local;
    std::vector<double> B_local;
    std::vector<double> C_local(blockSize * blockSize, 0.0);

    distribute_matrix(A_global, A_local, n, blockSize, q, world_rank, cart_comm, 0);
    distribute_matrix(B_global, B_local, n, blockSize, q, world_rank, cart_comm, 1);

    std::vector<double> A_temp(blockSize * blockSize);

    MPI_Barrier(cart_comm);
    double t_start = MPI_Wtime();

    // Алгоритм Fox
    int stage;
    for (stage = 0; stage < q; ++stage) {
        int rootCol = (row + stage) % q; // у цьому стовпці джерело для A

        // у row_comm ранги відповідають col, тому root = rootCol
        if (col == rootCol) {
            A_temp = A_local;
        }

        MPI_Bcast(&A_temp[0], blockSize * blockSize, MPI_DOUBLE,
                  rootCol, row_comm);

        // C_local += A_temp * B_local
        block_multiply_add(A_temp, B_local, C_local, blockSize);

        // циклічний зсув B вздовж стовпчика (по рядках)
        int src, dst;
        MPI_Cart_shift(cart_comm, 0, -1, &src, &dst);
        MPI_Status status;
        MPI_Sendrecv_replace(&B_local[0], blockSize * blockSize, MPI_DOUBLE,
                             dst, 2, src, 2, cart_comm, &status);
    }

    MPI_Barrier(cart_comm);
    double t_finish = MPI_Wtime();
    double local_time = t_finish - t_start;
    double max_time = 0.0;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (world_rank == 0) {
        std::cout << "=== Parallel execution time (Fox, " << world_size
                  << " processes): " << max_time << " s ===\n";
    }

    // Збір результату C
    collect_matrix(C_local, C_global, n, blockSize, q, world_rank, cart_comm);

    // Перевірка коректності на процесі 0 (послідовне множення)
    if (world_rank == 0) {
        std::vector<double> C_serial;
        serial_multiply(A_global, B_global, C_serial, n);

        bool ok = compare_matrices(C_global, C_serial, n, 1e-6);
        if (ok) {
            std::cout << "=== Parallel result matches serial computation ===\n";
        } else {
            std::cout << "*** Parallel result differs from serial computation ***\n";
        }

        if (n <= 5) {
            std::cout << "Result matrix C (parallel):\n";
            int i, j;
            for (i = 0; i < n; ++i) {
                for (j = 0; j < n; ++j) {
                    std::cout << C_global[i * n + j] << "\t";
                }
                std::cout << "\n";
            }
        }
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();
    return 0;
}
