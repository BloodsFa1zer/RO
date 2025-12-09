// SerialGS.cpp
// Послідовний алгоритм Гаусса–Зейделя для задачі Діріхле (рівняння Пуассона)
// Методичка: Learning Lab 6 (Serial Gauss-Seidel)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// Функції
void ProcessInitialization(double* &pMatrix, int &Size, double &Eps);
void DummyDataInitialization(double* pMatrix, int Size);
void PrintMatrix(double* pMatrix, int RowCount, int ColCount);
void ResultCalculation(double* pMatrix, int Size, double Eps, int &Iterations);
void ProcessTermination(double* pMatrix);

int main() {
    double* pMatrix = nullptr; // Матриця вузлів сітки
    int Size = 0;              // Розмір матриці (Size x Size)
    double Eps = 0.0;          // Вимагана точність
    int Iterations = 0;        // Кількість ітерацій
    std::printf("Serial Gauss - Seidel algorithm\n");

    // Ініціалізація (ввід розміру сітки, точності, виділення пам'яті, граничні умови)
    ProcessInitialization(pMatrix, Size, Eps);

    // Вивід початкової матриці (лише для невеликих розмірів, щоб не залити консоль)
    if (Size <= 10) {
        std::printf("\nInitial matrix:\n");
        PrintMatrix(pMatrix, Size, Size);
    }

    // Обчислення
    std::clock_t start = std::clock();
    ResultCalculation(pMatrix, Size, Eps, Iterations);
    std::clock_t finish = std::clock();
    double duration = double(finish - start) / double(CLOCKS_PER_SEC);

    // Результати
    std::printf("\nNumber of iterations: %d\n", Iterations);
    if (Size <= 10) {
        std::printf("\nResult matrix:\n");
        PrintMatrix(pMatrix, Size, Size);
    }
    std::printf("\nExecution time: %f s\n", duration);

    // Завершення
    ProcessTermination(pMatrix);
    return 0;
}

// ----------------------- Реалізація функцій -----------------------------

// Ввід розміру сітки, точності, виділення пам'яті та ініціалізація вузлів
void ProcessInitialization(double* &pMatrix, int &Size, double &Eps) {
    // Ввід розміру сітки
    do {
        std::printf("\nEnter the grid size (Size > 2): ");
        if (std::scanf("%d", &Size) != 1) {
            std::printf("Invalid input, exiting.\n");
            std::exit(1);
        }
        std::printf("Chosen grid size = %d\n", Size);
        if (Size <= 2) {
            std::printf("Size of the grid must be greater than 2!\n");
        }
    } while (Size <= 2);

    // Ввід точності
    do {
        std::printf("\nEnter the required accuracy (Eps > 0): ");
        if (std::scanf("%lf", &Eps) != 1) {
            std::printf("Invalid input, exiting.\n");
            std::exit(1);
        }
        std::printf("Chosen accuracy = %lf\n", Eps);
        if (Eps <= 0.0) {
            std::printf("Accuracy must be greater than 0!\n");
        }
    } while (Eps <= 0.0);

    // Виділення пам'яті
    pMatrix = new double[Size * Size];

    // Задання початкових значень вузлів (g ≡ 100 на границі, 0 всередині)
    DummyDataInitialization(pMatrix, Size);
}

// Проста ініціалізація вузлів сітки: 100 на границі, 0 у внутрішніх
void DummyDataInitialization(double* pMatrix, int Size) {
    for (int i = 0; i < Size; ++i) {
        for (int j = 0; j < Size; ++j) {
            bool isBoundary = (i == 0) || (i == Size - 1) || (j == 0) || (j == Size - 1);
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

// Реалізація методу Гаусса–Зейделя для задачі Діріхле
void ResultCalculation(double* pMatrix, int Size, double Eps, int &Iterations) {
    int i, j;
    double dm, dmax, temp;

    Iterations = 0;
    do {
        dmax = 0.0;
        // Оновлюємо лише внутрішні вузли (1..Size-2) – границю не чіпаємо
        for (i = 1; i < Size - 1; ++i) {
            for (j = 1; j < Size - 1; ++j) {
                int idx = Size * i + j;
                temp = pMatrix[idx];
                pMatrix[idx] = 0.25 * (
                    pMatrix[Size * i + (j + 1)] +   // праворуч
                    pMatrix[Size * i + (j - 1)] +   // ліворуч
                    pMatrix[Size * (i + 1) + j] +   // знизу
                    pMatrix[Size * (i - 1) + j]     // зверху
                );
                dm = std::fabs(pMatrix[idx] - temp);
                if (dmax < dm) dmax = dm;
            }
        }
        ++Iterations;
    } while (dmax > Eps);
}

// Коректне завершення: звільнення пам'яті
void ProcessTermination(double* pMatrix) {
    delete[] pMatrix;
}
