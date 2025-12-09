// TestGS.cpp
// Автоматичні тести для методу Гаусса-Зейделя
// Перевіряє коректність роботи та порівнює результати

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

// Структура для зберігання результатів
struct TestResult {
    int size;
    double eps;
    int iterations;
    double execution_time;
    bool passed;
    std::string error_message;
};

// Функція для читання матриці з виводу програми
bool readMatrixFromOutput(const char* filename, std::vector<double>& matrix, int& size) {
    FILE* file = std::fopen(filename, "r");
    if (!file) {
        return false;
    }
    
    char line[1024];
    bool inMatrix = false;
    matrix.clear();
    
    while (std::fgets(line, sizeof(line), file)) {
        if (std::strstr(line, "Result matrix:")) {
            inMatrix = true;
            continue;
        }
        
        if (inMatrix) {
            // Парсимо рядок матриці
            double value;
            int count = 0;
            char* token = std::strtok(line, " \n");
            while (token != nullptr) {
                if (std::sscanf(token, "%lf", &value) == 1) {
                    matrix.push_back(value);
                    count++;
                }
                token = std::strtok(nullptr, " \n");
            }
            
            if (count > 0 && size == 0) {
                size = count;
            }
        }
    }
    
    std::fclose(file);
    return !matrix.empty();
}

// Порівняння двох матриць з урахуванням точності
bool compareMatrices(const std::vector<double>& m1, const std::vector<double>& m2, 
                     int size, double tolerance) {
    if (m1.size() != m2.size() || m1.size() != size * size) {
        return false;
    }
    
    for (size_t i = 0; i < m1.size(); ++i) {
        if (std::fabs(m1[i] - m2[i]) > tolerance) {
            std::printf("Різниця в елементі [%zu]: %.6f vs %.6f (різниця: %.6f)\n", 
                       i, m1[i], m2[i], std::fabs(m1[i] - m2[i]));
            return false;
        }
    }
    
    return true;
}

// Тест 1: Перевірка граничних умов
bool testBoundaryConditions(int size) {
    std::printf("Тест 1: Перевірка граничних умов (Size=%d)\n", size);
    
    // Запускаємо послідовну версію
    char command[256];
    std::sprintf(command, "echo -e \"%d\\n0.01\" | ./SerialGaussSeidel > serial_test1.txt 2>&1", size);
    
    if (std::system(command) != 0) {
        std::printf("  ✗ Помилка запуску послідовної версії\n");
        return false;
    }
    
    // Читаємо матрицю
    std::vector<double> matrix;
    int readSize = 0;
    if (!readMatrixFromOutput("serial_test1.txt", matrix, readSize)) {
        std::printf("  ✗ Не вдалося прочитати матрицю\n");
        return false;
    }
    
    // Перевіряємо граничні умови (мають бути 100.0)
    bool passed = true;
    for (int i = 0; i < size; ++i) {
        // Верхній рядок
        if (std::fabs(matrix[i] - 100.0) > 1e-6) {
            std::printf("  ✗ Гранична умова не виконана: matrix[%d] = %.6f (очікується 100.0)\n", 
                       i, matrix[i]);
            passed = false;
        }
        // Нижній рядок
        int bottomIdx = (size - 1) * size + i;
        if (std::fabs(matrix[bottomIdx] - 100.0) > 1e-6) {
            std::printf("  ✗ Гранична умова не виконана: matrix[%d] = %.6f (очікується 100.0)\n", 
                       bottomIdx, matrix[bottomIdx]);
            passed = false;
        }
        // Лівий стовпець
        int leftIdx = i * size;
        if (std::fabs(matrix[leftIdx] - 100.0) > 1e-6) {
            std::printf("  ✗ Гранична умова не виконана: matrix[%d] = %.6f (очікується 100.0)\n", 
                       leftIdx, matrix[leftIdx]);
            passed = false;
        }
        // Правий стовпець
        int rightIdx = i * size + (size - 1);
        if (std::fabs(matrix[rightIdx] - 100.0) > 1e-6) {
            std::printf("  ✗ Гранична умова не виконана: matrix[%d] = %.6f (очікується 100.0)\n", 
                       rightIdx, matrix[rightIdx]);
            passed = false;
        }
    }
    
    if (passed) {
        std::printf("  ✓ Всі граничні умови виконані правильно\n");
    }
    
    return passed;
}

// Тест 2: Порівняння послідовної та паралельної версій
bool testSerialVsParallel(int size, int numProcs) {
    std::printf("Тест 2: Порівняння послідовної та паралельної версій (Size=%d, Procs=%d)\n", 
               size, numProcs);
    
    // Перевірка, чи можна запустити паралельну версію
    if (size < numProcs || (size - 2) % numProcs != 0) {
        std::printf("  ⊘ Пропущено: Size не відповідає вимогам для %d процесів\n", numProcs);
        return true; // Не помилка, просто не можна протестувати
    }
    
    double eps = 0.01;
    
    // Запускаємо послідовну версію
    char command[512];
    std::sprintf(command, "echo -e \"%d\\n%.6f\" | ./SerialGaussSeidel > serial_test2.txt 2>&1", 
                size, eps);
    
    if (std::system(command) != 0) {
        std::printf("  ✗ Помилка запуску послідовної версії\n");
        return false;
    }
    
    // Запускаємо паралельну версію
    std::sprintf(command, "echo -e \"%d\\n%.6f\" | mpirun -n %d ./ParallelGaussSeidel > parallel_test2.txt 2>&1", 
                size, eps, numProcs);
    
    if (std::system(command) != 0) {
        std::printf("  ✗ Помилка запуску паралельної версії\n");
        return false;
    }
    
    // Читаємо кількість ітерацій
    FILE* file = std::fopen("serial_test2.txt", "r");
    int serialIter = -1;
    if (file) {
        char line[256];
        while (std::fgets(line, sizeof(line), file)) {
            if (std::sscanf(line, "Number of iterations: %d", &serialIter) == 1) {
                break;
            }
        }
        std::fclose(file);
    }
    
    file = std::fopen("parallel_test2.txt", "r");
    int parallelIter = -1;
    if (file) {
        char line[256];
        while (std::fgets(line, sizeof(line), file)) {
            if (std::sscanf(line, "Number of iterations: %d", &parallelIter) == 1) {
                break;
            }
        }
        std::fclose(file);
    }
    
    // Порівнюємо кількість ітерацій
    if (serialIter == -1 || parallelIter == -1) {
        std::printf("  ✗ Не вдалося прочитати кількість ітерацій\n");
        return false;
    }
    
    if (serialIter != parallelIter) {
        std::printf("  ✗ Кількість ітерацій не збігається: Serial=%d, Parallel=%d\n", 
                   serialIter, parallelIter);
        return false;
    }
    
    std::printf("  ✓ Кількість ітерацій збігається: %d\n", serialIter);
    
    // Для малих розмірів порівнюємо матриці
    if (size <= 10) {
        std::vector<double> serialMatrix, parallelMatrix;
        int serialSize = 0, parallelSize = 0;
        
        if (readMatrixFromOutput("serial_test2.txt", serialMatrix, serialSize) &&
            readMatrixFromOutput("parallel_test2.txt", parallelMatrix, parallelSize)) {
            
            if (compareMatrices(serialMatrix, parallelMatrix, size, 1e-4)) {
                std::printf("  ✓ Матриці збігаються з точністю 1e-4\n");
            } else {
                std::printf("  ✗ Матриці не збігаються\n");
                return false;
            }
        }
    }
    
    return true;
}

// Тест 3: Перевірка збіжності
bool testConvergence(int size, double eps) {
    std::printf("Тест 3: Перевірка збіжності (Size=%d, Eps=%.6f)\n", size, eps);
    
    char command[256];
    std::sprintf(command, "echo -e \"%d\\n%.6f\" | ./SerialGaussSeidel > convergence_test.txt 2>&1", 
                size, eps);
    
    if (std::system(command) != 0) {
        std::printf("  ✗ Помилка запуску\n");
        return false;
    }
    
    // Перевіряємо, що програма завершилася успішно
    FILE* file = std::fopen("convergence_test.txt", "r");
    if (!file) {
        std::printf("  ✗ Не вдалося відкрити файл результатів\n");
        return false;
    }
    
    char line[256];
    int iterations = -1;
    while (std::fgets(line, sizeof(line), file)) {
        if (std::sscanf(line, "Number of iterations: %d", &iterations) == 1) {
            break;
        }
    }
    std::fclose(file);
    
    if (iterations == -1) {
        std::printf("  ✗ Не вдалося знайти кількість ітерацій\n");
        return false;
    }
    
    if (iterations <= 0) {
        std::printf("  ✗ Невірна кількість ітерацій: %d\n", iterations);
        return false;
    }
    
    std::printf("  ✓ Алгоритм зійшовся за %d ітерацій\n", iterations);
    return true;
}

int main() {
    std::printf("========================================\n");
    std::printf("Автоматичні тести для методу Гаусса-Зейделя\n");
    std::printf("========================================\n\n");
    
    int passed = 0;
    int total = 0;
    
    // Тест 1: Граничні умови
    total++;
    if (testBoundaryConditions(10)) {
        passed++;
    }
    std::printf("\n");
    
    // Тест 2: Порівняння версій
    total++;
    if (testSerialVsParallel(10, 2)) {
        passed++;
    }
    std::printf("\n");
    
    total++;
    if (testSerialVsParallel(14, 4)) {
        passed++;
    }
    std::printf("\n");
    
    // Тест 3: Збіжність
    total++;
    if (testConvergence(20, 0.01)) {
        passed++;
    }
    std::printf("\n");
    
    total++;
    if (testConvergence(20, 0.001)) {
        passed++;
    }
    std::printf("\n");
    
    // Очищення тимчасових файлів
    std::system("rm -f serial_test1.txt serial_test2.txt parallel_test2.txt convergence_test.txt");
    
    // Підсумок
    std::printf("========================================\n");
    std::printf("Результати: %d/%d тестів пройдено\n", passed, total);
    std::printf("========================================\n");
    
    return (passed == total) ? 0 : 1;
}
