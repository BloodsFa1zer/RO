#!/bin/bash

# Тестовий скрипт для лабораторної роботи №6
# Тестує послідовну та паралельну версії методу Гаусса-Зейделя

set -e  # Зупинитися при помилці

# Кольори для виводу
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функції для виводу
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Перевірка наявності необхідних програм
check_dependencies() {
    print_info "Перевірка залежностей..."
    
    if ! command -v g++ &> /dev/null; then
        print_error "g++ не знайдено. Встановіть компілятор C++."
        exit 1
    fi
    print_success "g++ знайдено"
    
    if ! command -v mpic++ &> /dev/null; then
        print_error "mpic++ не знайдено. Встановіть OpenMPI: brew install open-mpi"
        exit 1
    fi
    print_success "mpic++ знайдено"
    
    if ! command -v mpirun &> /dev/null; then
        print_error "mpirun не знайдено. Встановіть OpenMPI: brew install open-mpi"
        exit 1
    fi
    print_success "mpirun знайдено"
}

# Компіляція програм
compile_programs() {
    print_info "Компіляція програм..."
    
    # Компіляція послідовної версії
    if g++ -O2 SerialGS.cpp -o SerialGaussSeidel 2>&1; then
        print_success "SerialGS.cpp скомпільовано"
    else
        print_error "Помилка компіляції SerialGS.cpp"
        exit 1
    fi
    
    # Компіляція паралельної версії
    if mpic++ -O2 ParallelGS.cpp -o ParallelGaussSeidel 2>&1; then
        print_success "ParallelGS.cpp скомпільовано"
    else
        print_error "Помилка компіляції ParallelGS.cpp"
        exit 1
    fi
}

# Тест 1: Малий розмір сітки (перевірка коректності)
test_small_size() {
    print_info "Тест 1: Малий розмір сітки (Size=10, Eps=0.01)"
    
    SIZE=10
    EPS=0.01
    
    # Запуск послідовної версії
    echo -e "${SIZE}\n${EPS}" | ./SerialGaussSeidel > serial_output.txt 2>&1
    SERIAL_ITER=$(grep "Number of iterations" serial_output.txt | awk '{print $4}')
    SERIAL_TIME=$(grep "Execution time" serial_output.txt | awk '{print $3}')
    
    if [ -z "$SERIAL_ITER" ]; then
        print_error "Не вдалося отримати кількість ітерацій з послідовної версії"
        return 1
    fi
    
    print_success "Послідовна версія: $SERIAL_ITER ітерацій, час: $SERIAL_TIME сек"
    
    # Запуск паралельної версії на 2 процесах
    if (( SIZE >= 2 && (SIZE - 2) % 2 == 0 )); then
        echo -e "${SIZE}\n${EPS}" | mpirun -n 2 ./ParallelGaussSeidel > parallel_output.txt 2>&1
        PARALLEL_ITER=$(grep "Number of iterations" parallel_output.txt | awk '{print $4}')
        PARALLEL_TIME=$(grep "Parallel execution time" parallel_output.txt | awk '{print $4}')
        
        if [ -z "$PARALLEL_ITER" ]; then
            print_error "Не вдалося отримати кількість ітерацій з паралельної версії"
            return 1
        fi
        
        print_success "Паралельна версія (2 proc): $PARALLEL_ITER ітерацій, час: $PARALLEL_TIME сек"
        
        # Перевірка, що кількість ітерацій збігається
        if [ "$SERIAL_ITER" = "$PARALLEL_ITER" ]; then
            print_success "Кількість ітерацій збігається: $SERIAL_ITER"
        else
            print_error "Кількість ітерацій не збігається: Serial=$SERIAL_ITER, Parallel=$PARALLEL_ITER"
            return 1
        fi
    fi
}

# Тест 2: Середній розмір сітки
test_medium_size() {
    print_info "Тест 2: Середній розмір сітки (Size=50, Eps=0.01)"
    
    SIZE=50
    EPS=0.01
    
    # Запуск послідовної версії
    echo -e "${SIZE}\n${EPS}" | ./SerialGaussSeidel > serial_output.txt 2>&1
    SERIAL_ITER=$(grep "Number of iterations" serial_output.txt | awk '{print $4}')
    SERIAL_TIME=$(grep "Execution time" serial_output.txt | awk '{print $3}')
    
    print_success "Послідовна версія: $SERIAL_ITER ітерацій, час: $SERIAL_TIME сек"
    
    # Запуск паралельної версії на 2 процесах
    if (( SIZE >= 2 && (SIZE - 2) % 2 == 0 )); then
        echo -e "${SIZE}\n${EPS}" | mpirun -n 2 ./ParallelGaussSeidel > parallel_output.txt 2>&1
        PARALLEL_ITER=$(grep "Number of iterations" parallel_output.txt | awk '{print $4}')
        PARALLEL_TIME=$(grep "Parallel execution time" parallel_output.txt | awk '{print $4}')
        
        print_success "Паралельна версія (2 proc): $PARALLEL_ITER ітерацій, час: $PARALLEL_TIME сек"
        
        if [ "$SERIAL_ITER" = "$PARALLEL_ITER" ]; then
            print_success "Кількість ітерацій збігається: $SERIAL_ITER"
        else
            print_error "Кількість ітерацій не збігається"
        fi
    fi
}

# Тест 3: Великий розмір сітки (якщо можливо)
test_large_size() {
    print_info "Тест 3: Великий розмір сітки (Size=100, Eps=0.01)"
    
    SIZE=100
    EPS=0.01
    
    # Запуск послідовної версії
    echo -e "${SIZE}\n${EPS}" | timeout 60 ./SerialGaussSeidel > serial_output.txt 2>&1 || true
    SERIAL_ITER=$(grep "Number of iterations" serial_output.txt | awk '{print $4}' || echo "N/A")
    SERIAL_TIME=$(grep "Execution time" serial_output.txt | awk '{print $3}' || echo "N/A")
    
    if [ "$SERIAL_ITER" != "N/A" ]; then
        print_success "Послідовна версія: $SERIAL_ITER ітерацій, час: $SERIAL_TIME сек"
    else
        print_error "Послідовна версія не завершилася за 60 секунд або сталася помилка"
        return 1
    fi
    
    # Запуск паралельної версії на 4 процесах
    if (( SIZE >= 4 && (SIZE - 2) % 4 == 0 )); then
        echo -e "${SIZE}\n${EPS}" | timeout 60 mpirun -n 4 ./ParallelGaussSeidel > parallel_output.txt 2>&1 || true
        PARALLEL_ITER=$(grep "Number of iterations" parallel_output.txt | awk '{print $4}' || echo "N/A")
        PARALLEL_TIME=$(grep "Parallel execution time" parallel_output.txt | awk '{print $4}' || echo "N/A")
        
        if [ "$PARALLEL_ITER" != "N/A" ]; then
            print_success "Паралельна версія (4 proc): $PARALLEL_ITER ітерацій, час: $PARALLEL_TIME сек"
            
            if [ "$SERIAL_ITER" = "$PARALLEL_ITER" ]; then
                print_success "Кількість ітерацій збігається: $SERIAL_ITER"
            else
                print_error "Кількість ітерацій не збігається"
            fi
        else
            print_error "Паралельна версія не завершилася за 60 секунд або сталася помилка"
        fi
    fi
}

# Тест 4: Перевірка різних значень точності
test_different_epsilon() {
    print_info "Тест 4: Різні значення точності (Size=20)"
    
    SIZE=20
    
    for EPS in 0.1 0.01 0.001; do
        print_info "  Тестування з Eps=$EPS"
        
        # Послідовна версія
        echo -e "${SIZE}\n${EPS}" | ./SerialGaussSeidel > serial_output.txt 2>&1
        SERIAL_ITER=$(grep "Number of iterations" serial_output.txt | awk '{print $4}')
        
        # Паралельна версія на 2 процесах
        if (( SIZE >= 2 && (SIZE - 2) % 2 == 0 )); then
            echo -e "${SIZE}\n${EPS}" | mpirun -n 2 ./ParallelGaussSeidel > parallel_output.txt 2>&1
            PARALLEL_ITER=$(grep "Number of iterations" parallel_output.txt | awk '{print $4}')
            
            if [ "$SERIAL_ITER" = "$PARALLEL_ITER" ]; then
                print_success "    Eps=$EPS: обидві версії збігаються ($SERIAL_ITER ітерацій)"
            else
                print_error "    Eps=$EPS: не збігаються (Serial=$SERIAL_ITER, Parallel=$PARALLEL_ITER)"
            fi
        fi
    done
}

# Тест 5: Перевірка валідації вводу
test_input_validation() {
    print_info "Тест 5: Перевірка валідації вводу"
    
    # Тест з невалідним розміром (Size <= 2)
    echo -e "2\n0.01" | ./SerialGaussSeidel > /dev/null 2>&1 && {
        print_error "Послідовна версія прийняла Size=2 (має відхилити)"
    } || {
        print_success "Послідовна версія правильно відхилила Size=2"
    }
    
    # Тест з невалідною точністю (Eps <= 0)
    echo -e "10\n-0.01" | ./SerialGaussSeidel > /dev/null 2>&1 && {
        print_error "Послідовна версія прийняла Eps=-0.01 (має відхилити)"
    } || {
        print_success "Послідовна версія правильно відхилила Eps=-0.01"
    }
}

# Тест 6: Порівняння результатів (якщо можливо)
test_result_comparison() {
    print_info "Тест 6: Порівняння числових результатів"
    
    SIZE=10
    EPS=0.01
    
    # Запускаємо обидві версії та зберігаємо результати
    echo -e "${SIZE}\n${EPS}" | ./SerialGaussSeidel > serial_output.txt 2>&1
    echo -e "${SIZE}\n${EPS}" | mpirun -n 2 ./ParallelGaussSeidel > parallel_output.txt 2>&1
    
    # Перевіряємо, що обидві версії виводять матрицю для малих розмірів
    if grep -q "Result matrix" serial_output.txt && grep -q "Result matrix" parallel_output.txt; then
        print_success "Обидві версії вивели результати"
        print_info "Перевірте вручну, що значення матриць збігаються"
    else
        print_info "Матриці не виведені (Size > 10 або інша причина)"
    fi
}

# Очищення тимчасових файлів
cleanup() {
    print_info "Очищення тимчасових файлів..."
    rm -f serial_output.txt parallel_output.txt
    print_success "Очищення завершено"
}

# Головна функція
main() {
    echo "=========================================="
    echo "Тестування лабораторної роботи №6"
    echo "Метод Гаусса-Зейделя"
    echo "=========================================="
    echo ""
    
    check_dependencies
    echo ""
    
    compile_programs
    echo ""
    
    # Запуск тестів
    test_small_size
    echo ""
    
    test_medium_size
    echo ""
    
    test_different_epsilon
    echo ""
    
    test_input_validation
    echo ""
    
    test_result_comparison
    echo ""
    
    # Великий розмір (опційно, може зайняти багато часу)
    read -p "Запустити тест з великим розміром сітки (Size=100)? Це може зайняти багато часу. (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_large_size
        echo ""
    fi
    
    cleanup
    
    echo "=========================================="
    print_success "Всі тести завершено!"
    echo "=========================================="
}

# Запуск головної функції
main
