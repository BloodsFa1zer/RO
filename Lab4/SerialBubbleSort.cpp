#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <algorithm>
#include "SerialBubbleSortTest.h"

using namespace std;

const double RandomDataMultiplier = 1000.0;

// Test sizes from the lab description
const int TestSizes[] = {10, 100, 10000, 20000, 30000, 40000, 50000};
const int NumTests = sizeof(TestSizes) / sizeof(TestSizes[0]);

int main(int argc, char *argv[]) {
    printf("=== Serial Bubble Sort Program ===\n\n");
    printf("Testing different data sizes and comparing with std::sort\n\n");
    printf("%-10s | %-25s | %-25s\n", "Data Size", "Bubble Sort (sec)", "std::sort (sec)");
    printf("----------|--------------------------|--------------------------\n");

    for (int test = 0; test < NumTests; test++) {
        int DataSize = TestSizes[test];
        double *pData = new double[DataSize];
        double *pDataCopy = new double[DataSize];
        
        // Initialize data with random values
        RandomDataInitialization(pData, DataSize);
        CopyData(pData, DataSize, pDataCopy);

        // Test Bubble Sort
        clock_t start = clock();
        SerialBubble(pData, DataSize);
        clock_t finish = clock();
        double bubbleTime = (finish - start) / double(CLOCKS_PER_SEC);

        // Test std::sort
        start = clock();
        SerialStdSort(pDataCopy, DataSize);
        finish = clock();
        double stdTime = (finish - start) / double(CLOCKS_PER_SEC);

        printf("%-10d | %-25.9f | %-25.9f\n", DataSize, bubbleTime, stdTime);

        delete [] pData;
        delete [] pDataCopy;
    }

    printf("\n=== Testing completed ===\n");
    return 0;
}

// Function for allocating the memory and setting the initial values
void ProcessInitialization(double *&pData, int& DataSize) {
    do {
        printf("Enter the size of data to be sorted: ");
        scanf("%d", &DataSize);
        if (DataSize <= 0)
            printf("Data size should be greater than zero\n");
    } while (DataSize <= 0);

    printf("Sorting %d data items\n", DataSize);

    pData = new double[DataSize];

    // Setting the data by the random generator
    RandomDataInitialization(pData, DataSize);
}

// Function for computational process termination
void ProcessTermination(double *pData) {
    delete [] pData;
}

// Function for initializing the data by the random generator
void RandomDataInitialization(double *pData, int DataSize) {
    srand((unsigned)time(0));
    for (int i = 0; i < DataSize; i++)
        pData[i] = double(rand()) / RAND_MAX * RandomDataMultiplier;
}

// Function for the serial bubble sort algorithm
void SerialBubble(double *pData, int DataSize) {
    double Tmp;
    for (int i = 1; i < DataSize; i++)
        for (int j = 0; j < DataSize - i; j++)
            if (pData[j] > pData[j + 1]) {
                Tmp = pData[j];
                pData[j] = pData[j + 1];
                pData[j + 1] = Tmp;
            }
}

// Function for copying data
void CopyData(double *pData, int DataSize, double *pDataCopy) {
    for (int i = 0; i < DataSize; i++)
        pDataCopy[i] = pData[i];
}
