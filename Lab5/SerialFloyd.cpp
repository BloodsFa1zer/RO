// SerialFloyd.cpp
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

const int NO_EDGE = -1;
const double INFINITY_PERCENT = 50.0; // для випадкової ініціалізації

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

void DummyDataInitialization(int *matrix, int size) {
    // Верхній трикутник (включно з діагоналлю)
    for (int i = 0; i < size; ++i) {
        for (int j = i; j < size; ++j) {
            if (i == j) {
                matrix[i * size + j] = 0;
            } else if (i == 0) {
                // Від вершини 0 до j вага = j
                matrix[i * size + j] = j;
            } else {
                // Немає ребра
                matrix[i * size + j] = NO_EDGE;
            }
        }
    }

    // Дзеркалимо верхній трикутник у нижній, щоб граф був неорієнтованим
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
                    matrix[i * size + j] = NO_EDGE; // немає ребра
                } else {
                    matrix[i * size + j] = (std::rand() % 10) + 1; // довільна позитивна вага
                }
            }
        }
    }
}

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

void ProcessInitialization(int *&matrix, int &size, bool quiet = false) {
    matrix = new int[size * size];

    // Фіксований приклад з методички
    DummyDataInitialization(matrix, size);

    // Якщо треба випадкова матриця — закоментуй DummyDataInitialization
    // і розкоментуй наступний виклик:
    // RandomDataInitialization(matrix, size);
    
    if (!quiet) {
        std::printf("Using graph with %d vertices\n", size);
    }
}

void ProcessTermination(int *matrix) {
    delete [] matrix;
}

int main(int argc, char **argv) {
    int *matrix = nullptr;
    int size = 0;
    bool quiet = false;

    // Перевірка аргументів командного рядка
    if (argc >= 2) {
        size = std::atoi(argv[1]);
        if (size <= 0) {
            std::fprintf(stderr, "Error: Invalid size. Must be positive.\n");
            return 1;
        }
        quiet = (argc >= 3 && std::string(argv[2]) == "quiet");
    } else {
        std::printf("Serial Floyd algorithm\n");
        do {
            std::printf("Enter the number of vertices: ");
            std::scanf("%d", &size);
            if (size <= 0) {
                std::printf("The number of vertices should be greater than zero\n");
            }
        } while (size <= 0);
    }

    // Ініціалізація
    ProcessInitialization(matrix, size, quiet);

    if (!quiet && size <= 10) {
        std::printf("The matrix before Floyd algorithm\n");
        PrintMatrix(matrix, size, size);
    }

    std::clock_t start = std::clock();
    SerialFloyd(matrix, size);
    std::clock_t finish = std::clock();

    if (!quiet && size <= 10) {
        std::printf("The matrix after Floyd algorithm\n");
        PrintMatrix(matrix, size, size);
    }

    double duration = double(finish - start) / double(CLOCKS_PER_SEC);
    if (quiet) {
        std::printf("%d %f\n", size, duration);
    } else {
        std::printf("Time of execution: %f\n", duration);
    }

    ProcessTermination(matrix);
    return 0;
}
