#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

const double TOL = 1e-6;

// Initialize matrix A and vector x only on rank 0
void init_global_data(int n, std::vector<double>& A, std::vector<double>& x) {
    A.resize(n * n);
    x.resize(n);

    for (int i = 0; i < n; ++i) {
        x[i] = 1.0;
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = static_cast<double>(i + j + 1);
        }
    }
}

// Compute how many rows each rank gets
void compute_rows_distribution(int n, int world_size, std::vector<int>& rows_per_rank) {
    rows_per_rank.assign(world_size, 0);
    int base  = n / world_size;
    int extra = n % world_size;

    for (int r = 0; r < world_size; ++r) {
        rows_per_rank[r] = base + (r < extra ? 1 : 0);
    }
}

// Local matrix-vector multiplication
void multiply_local(
    const std::vector<double>& localA,
    const std::vector<double>& x,
    std::vector<double>& local_y,
    int n,
    int local_rows
) {
    local_y.assign(local_rows, 0.0);

    for (int i = 0; i < local_rows; ++i) {
        double sum = 0.0;
        const double* row = &localA[i * n];
        for (int j = 0; j < n; ++j) {
            sum += row[j] * x[j];
        }
        local_y[i] = sum;
    }
}

// Serial multiplication for check (rank 0 only)
void multiply_serial(
    const std::vector<double>& A,
    const std::vector<double>& x,
    std::vector<double>& y,
    int n
) {
    y.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        const double* row = &A[i * n];
        for (int j = 0; j < n; ++j) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

// Compare parallel vs serial result
void check_result(
    const std::vector<double>& par,
    const std::vector<double>& seq,
    int n
) {
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        double diff = std::fabs(par[i] - seq[i]);
        if (diff > TOL) {
            std::cout << "Mismatch at index " << i
                      << ": parallel=" << par[i]
                      << ", serial=" << seq[i]
                      << ", diff=" << diff << '\n';
            ok = false;
        }
    }

    if (ok) {
        std::cout << "\n=== Parallel result matches serial computation ===\n";
    } else {
        std::cout << "\n*** Parallel result differs from serial computation ***\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // disable buffering so each rank prints immediately
    setvbuf(stdout, nullptr, _IONBF, 0);

    int n = 0;
    if (world_rank == 0) {
        std::cout << "Parallel matrix-vector multiplication (MPI)\n";
        std::cout << "Enter matrix/vector size: ";
        std::cin >> n;

        if (n < world_size) {
            std::cout << "Error: matrix size must be >= number of MPI processes\n";
        }
    }

    // Broadcast size
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n <= 0 || n < world_size) {
        if (world_rank == 0) {
            std::cout << "Invalid size. Exiting.\n";
        }
        MPI_Finalize();
        return 0;
    }

    // Row distribution
    std::vector<int> rows_per_rank(world_size);
    if (world_rank == 0) {
        compute_rows_distribution(n, world_size, rows_per_rank);
    }
    MPI_Bcast(rows_per_rank.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

    int local_rows = rows_per_rank[world_rank];

    std::vector<double> globalA;
    std::vector<double> x;
    std::vector<double> y_parallel;
    std::vector<double> y_serial;

    if (world_rank == 0) {
        init_global_data(n, globalA, x);
        y_parallel.resize(n);
    } else {
        x.resize(n);
    }

    // Broadcast vector x
    MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Prepare Scatterv
    std::vector<int> send_counts;
    std::vector<int> displs;

    if (world_rank == 0) {
        send_counts.resize(world_size);
        displs.resize(world_size);

        int offset = 0;
        for (int r = 0; r < world_size; ++r) {
            send_counts[r] = rows_per_rank[r] * n;
            displs[r]      = offset;
            offset        += send_counts[r];
        }
    }

    std::vector<double> localA(local_rows * n);

    MPI_Scatterv(
        world_rank == 0 ? globalA.data() : nullptr,
        world_rank == 0 ? send_counts.data() : nullptr,
        world_rank == 0 ? displs.data() : nullptr,
        MPI_DOUBLE,
        localA.data(), local_rows * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    std::vector<double> local_y;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    multiply_local(localA, x, local_y, n, local_rows);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_finish = MPI_Wtime();
    double local_time = t_finish - t_start;
    double max_time   = 0.0;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    if (world_rank == 0) {
        recv_counts.resize(world_size);
        recv_displs.resize(world_size);
        int offset2 = 0;
        for (int r = 0; r < world_size; ++r) {
            recv_counts[r] = rows_per_rank[r];
            recv_displs[r] = offset2;
            offset2       += recv_counts[r];
        }
    }

    MPI_Gatherv(
        local_y.data(), local_rows, MPI_DOUBLE,
        world_rank == 0 ? y_parallel.data() : nullptr,
        world_rank == 0 ? recv_counts.data() : nullptr,
        world_rank == 0 ? recv_displs.data() : nullptr,
        MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    std::cout << "\nRank " << world_rank << ": local_rows = " << local_rows << "\n";
    if (local_rows <= 5) {
        std::cout << "Local partial result: ";
        for (int i = 0; i < local_rows; ++i) {
            std::cout << local_y[i] << " ";
        }
        std::cout << "\n";
    }

    if (world_rank == 0) {
        std::cout << "\n=== Parallel execution time (" << world_size
                  << " processes): " << max_time << " s ===\n";

        multiply_serial(globalA, x, y_serial, n);
        check_result(y_parallel, y_serial, n);

        if (n <= 10) {
            std::cout << "\nFull result vector (parallel):\n";
            for (int i = 0; i < n; ++i) {
                std::cout << y_parallel[i] << " ";
            }
            std::cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
