// ParallelGS.cpp
// Паралельний алгоритм Гаусса–Зейделя для задачі Діріхле
// Блочно-смужкова декомпозиція по рядках, MPI

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

// Глобальні змінні MPI
static int ProcNum = 0;   // кількість процесів
static int ProcRank = -1; // ранг поточного процесу

// Прототипи
void DummyDataInitialization(double* pMatrix, int Size);
void PrintMatrix(double* pMatrix, int RowCount, int ColCount);

void ProcessInitialization(double* &pMatrix,
                           double* &pProcRows,
                           int &Size,
                           int &RowNum,
                           double &Eps);

void ProcessTermination(double* pMatrix, double* pProcRows);

void DataDistribution(double* pMatrix,
                      double* pProcRows,
                      int Size,
                      int RowNum);

void TestDistribution(double* pMatrix,
                      double* pProcRows,
                      int Size,
                      int RowNum);

void ExchangeData(double* pProcRows, int Size, int RowNum);
double IterationCalculation(double* pProcRows, int Size, int RowNum);

void ParallelResultCalculation(double* pProcRows,
                               int Size,
                               int RowNum,
                               double Eps,
                               int &Iterations);

void ResultCollection(double* pMatrix,
                      double* pProcRows,
                      int Size,
                      int RowNum);

int main(int argc, char* argv[]) {
    double* pMatrix    = nullptr; // повна матриця (лише на процесі 0)
    double* pProcRows  = nullptr; // смуга рядків на кожному процесі
    int Size           = 0;       // розмір сітки (Size x Size)
    int RowNum         = 0;       // кількість рядків у смузі (з урахуванням 2 граничних)
    double Eps         = 0.0;     // точність
    int Iterations     = 0;       // кількість ітерацій
    double Start, Finish, Duration;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0) {
        std::printf("Parallel Gauss - Seidel algorithm\n");
    }

    // Ввід параметрів, розподіл пам'яті, ініціалізація граничних умов
    ProcessInitialization(pMatrix, pProcRows, Size, RowNum, Eps);

    // Розподіл даних (смуга рядків на кожен процес)
    DataDistribution(pMatrix, pProcRows, Size, RowNum);

    MPI_Barrier(MPI_COMM_WORLD);
    Start = MPI_Wtime();

    // Паралельний метод Гаусса–Зейделя
    ParallelResultCalculation(pProcRows, Size, RowNum, Eps, Iterations);

    Finish   = MPI_Wtime();
    Duration = Finish - Start;

    // Збирання результатів на процесі 0
    ResultCollection(pMatrix, pProcRows, Size, RowNum);

    if (ProcRank == 0) {
        std::printf("\nNumber of iterations: %d\n", Iterations);
        if (Size <= 10) {
            std::printf("\nResult matrix:\n");
            PrintMatrix(pMatrix, Size, Size);
        }
        std::printf("\nParallel execution time: %f s\n", Duration);
    }

    ProcessTermination(pMatrix, pProcRows);
    MPI_Finalize();
    return 0;
}

// ------------------------ Допоміжні функції ---------------------------

// Початкове задання: 100 на границі, 0 всередині
void DummyDataInitialization(double* pMatrix, int Size) {
    for (int i = 0; i < Size; ++i) {
        for (int j = 0; j < Size; ++j) {
            bool isBoundary = (i == 0) || (i == Size - 1) ||
                              (j == 0) || (j == Size - 1);
            if (isBoundary) {
                pMatrix[i * Size + j] = 100.0;
            } else {
                pMatrix[i * Size + j] = 0.0;
            }
        }
    }
}

// Форматований вивід матриці
void PrintMatrix(double* pMatrix, int RowCount, int ColCount) {
    for (int i = 0; i < RowCount; ++i) {
        for (int j = 0; j < ColCount; ++j) {
            std::printf("%7.4f ", pMatrix[i * ColCount + j]);
        }
        std::printf("\n");
    }
}

// Ініціалізація: ввід Size, Eps, перевірки, broadcast, виділення пам'яті
void ProcessInitialization(double* &pMatrix,
                           double* &pProcRows,
                           int &Size,
                           int &RowNum,
                           double &Eps) {
    if (ProcRank == 0) {
        // Ввід розміру сітки з перевіркою:
        // Size > 2, Size >= ProcNum, (Size-2) кратне ProcNum
        do {
            std::printf("\nEnter the grid size: ");
            if (std::scanf("%d", &Size) != 1) {
                std::printf("Invalid input, exiting.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::printf("Chosen grid size = %d\n", Size);

            if (Size <= 2) {
                std::printf("Grid size must be greater than 2!\n");
            }
            if (Size < ProcNum) {
                std::printf("Grid size must be >= number of processes!\n");
            }
            if ((Size - 2) % ProcNum != 0) {
                std::printf("Number of inner rows (Size-2) must be divisible "
                            "by number of processes!\n");
            }
        } while ((Size <= 2) || (Size < ProcNum) || ((Size - 2) % ProcNum != 0));

        // Ввід точності
        do {
            std::printf("\nEnter the required accuracy: ");
            if (std::scanf("%lf", &Eps) != 1) {
                std::printf("Invalid input, exiting.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::printf("Chosen accuracy = %lf\n", Eps);
            if (Eps <= 0.0) {
                std::printf("Accuracy must be greater than 0!\n");
            }
        } while (Eps <= 0.0);
    }

    // Розсилка Size та Eps усім процесам
    MPI_Bcast(&Size, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&Eps,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Кількість рядків у смузі (включно з двома граничними)
    RowNum = (Size - 2) / ProcNum + 2;

    // Виділення пам'яті під смугу на кожному процесі
    pProcRows = new double[RowNum * Size];

    // Повна матриця лише на процесі 0
    if (ProcRank == 0) {
        pMatrix = new double[Size * Size];
        DummyDataInitialization(pMatrix, Size);
    } else {
        pMatrix = nullptr;
    }
}

// Завершення: звільнення пам'яті
void ProcessTermination(double* pMatrix, double* pProcRows) {
    if (ProcRank == 0 && pMatrix != nullptr) {
        delete[] pMatrix;
    }
    delete[] pProcRows;
}

// Розподіл смуг між процесами: MPI_Scatter + пересилка нижнього граничного рядка
void DataDistribution(double* pMatrix,
                      double* pProcRows,
                      int Size,
                      int RowNum) {
    MPI_Status status;

    // Розсилаємо внутрішні рядки (без верхнього та нижнього граничних)
    // з pMatrix[Size] (другий рядок повної матриці)
    MPI_Scatter(
        pMatrix + Size,              // початок внутрішніх рядків
        (RowNum - 2) * Size,         // скільки елементів на процес
        MPI_DOUBLE,
        pProcRows + Size,            // запис з другого рядка локальної смуги
        (RowNum - 2) * Size,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Копіюємо верхній граничний рядок у процес 0
    if (ProcRank == 0) {
        for (int j = 0; j < Size; ++j) {
            pProcRows[j] = pMatrix[j];
        }
    }

    // Надсилаємо нижній граничний рядок (останній рядок матриці) процесу ProcNum-1
    if (ProcRank == 0) {
        MPI_Send(
            pMatrix + Size * (Size - 1),
            Size,
            MPI_DOUBLE,
            ProcNum - 1,
            5,
            MPI_COMM_WORLD
        );
    }
    if (ProcRank == ProcNum - 1) {
        MPI_Recv(
            pProcRows + (RowNum - 1) * Size,
            Size,
            MPI_DOUBLE,
            0,
            5,
            MPI_COMM_WORLD,
            &status
        );
    }
}

// (опційно) перевірка розподілу – можна не викликати в фінальній версії
void TestDistribution(double* pMatrix,
                      double* pProcRows,
                      int Size,
                      int RowNum) {
    if (ProcRank == 0 && pMatrix != nullptr) {
        std::printf("Initial matrix:\n");
        PrintMatrix(pMatrix, Size, Size);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int p = 0; p < ProcNum; ++p) {
        if (ProcRank == p) {
            std::printf("\nProcRank = %d\nMatrix stripe:\n", ProcRank);
            PrintMatrix(pProcRows, RowNum, Size);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// Обмін граничними рядками між сусідніми процесами (Sendrecv)
void ExchangeData(double* pProcRows, int Size, int RowNum) {
    MPI_Status status;
    int NextProcNum = (ProcRank == ProcNum - 1) ? MPI_PROC_NULL : ProcRank + 1;
    int PrevProcNum = (ProcRank == 0)           ? MPI_PROC_NULL : ProcRank - 1;

    // Надсилаємо нижній внутрішній рядок (RowNum-2) вниз, приймаємо верхній граничний у рядок 0
    MPI_Sendrecv(
        pProcRows + Size * (RowNum - 2), // sendbuf
        Size,
        MPI_DOUBLE,
        NextProcNum,
        4,
        pProcRows,                       // recvbuf (рядок 0)
        Size,
        MPI_DOUBLE,
        PrevProcNum,
        4,
        MPI_COMM_WORLD,
        &status
    );

    // Надсилаємо перший внутрішній рядок (1) вгору, приймаємо нижній граничний у рядок RowNum-1
    MPI_Sendrecv(
        pProcRows + Size,                 // sendbuf (рядок 1)
        Size,
        MPI_DOUBLE,
        PrevProcNum,
        5,
        pProcRows + Size * (RowNum - 1),  // recvbuf (останній рядок)
        Size,
        MPI_DOUBLE,
        NextProcNum,
        5,
        MPI_COMM_WORLD,
        &status
    );
}

// Одна ітерація Гаусса–Зейделя над локальною смугою, повертає локальний dmax
double IterationCalculation(double* pProcRows, int Size, int RowNum) {
    int i, j;
    double dm, dmax, temp;
    dmax = 0.0;

    // Оновлюємо лише "власні" внутрішні рядки 1..RowNum-2
    for (i = 1; i < RowNum - 1; ++i) {
        for (j = 1; j < Size - 1; ++j) {
            int idx = Size * i + j;
            temp = pProcRows[idx];
            pProcRows[idx] = 0.25 * (
                pProcRows[Size * i + (j + 1)] +
                pProcRows[Size * i + (j - 1)] +
                pProcRows[Size * (i + 1) + j] +
                pProcRows[Size * (i - 1) + j]
            );
            dm = std::fabs(pProcRows[idx] - temp);
            if (dmax < dm) dmax = dm;
        }
    }

    return dmax;
}

// Повний паралельний метод Гаусса–Зейделя з обміном рядків і MPI_Allreduce
void ParallelResultCalculation(double* pProcRows,
                               int Size,
                               int RowNum,
                               double Eps,
                               int &Iterations) {
    double ProcDelta, Delta;
    Iterations = 0;

    do {
        ++Iterations;

        // Обмін граничними рядками з сусідами
        ExchangeData(pProcRows, Size, RowNum);

        // Локальна ітерація ГЗ + локальний максимум відхилення
        ProcDelta = IterationCalculation(pProcRows, Size, RowNum);

        // Глобальний максимум відхилення
        MPI_Allreduce(
            &ProcDelta,
            &Delta,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            MPI_COMM_WORLD
        );
    } while (Delta > Eps);
}

// Збирання внутрішніх рядків назад у повну матрицю (на процесі 0)
void ResultCollection(double* pMatrix,
                      double* pProcRows,
                      int Size,
                      int RowNum) {
    MPI_Gather(
        pProcRows + Size,          // пропускаємо верхній граничний рядок 0
        (RowNum - 2) * Size,
        MPI_DOUBLE,
        pMatrix + Size,            // починаємо з другого рядка повної матриці
        (RowNum - 2) * Size,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );
}
