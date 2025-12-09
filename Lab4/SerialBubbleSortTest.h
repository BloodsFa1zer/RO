#ifndef SERIALBUBBLESORTTEST_H
#define SERIALBUBBLESORTTEST_H

// Function for formatted data output
void PrintData(double *pData, int DataSize);

// Sorting by the standard library algorithm
void SerialStdSort(double *pData, int DataSize);

// Function for allocating the memory and setting the initial values
void ProcessInitialization(double *&pData, int& DataSize);

// Function for computational process termination
void ProcessTermination(double *pData);

// Function for initializing the data by the random generator
void RandomDataInitialization(double *pData, int DataSize);

// Function for the serial bubble sort algorithm
void SerialBubble(double *pData, int DataSize);

// Function for copying data
void CopyData(double *pData, int DataSize, double *pDataCopy);

#endif // SERIALBUBBLESORTTEST_H
