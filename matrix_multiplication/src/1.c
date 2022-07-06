#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void fillMatrix(double *A, int N, int M) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            A[i * M + j] = 1 + (i*j) % 2;
        }
    }
}

void printMatrix(double *A, int N, int M) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            printf(" %lf", A[i * M + j]);
        }
        printf("\n");
    }
}

int main() {

    int N1 = 4000;
    int N2 = 600;
    int N3 = 1200;

    // double A[N1 * N2];
    double *A = (double*)malloc(sizeof(double) * N1 * N2);
    memset(A, 0, N1 * N2 * sizeof(double));
    // double B[N2 * N3];
    double *B = (double*)malloc(sizeof(double) * N3 * N2);
    memset(B, 0, N2 * N3 * sizeof(double));
    // double C[N1 * N3];
    double *C = (double*)malloc(sizeof(double) * N1 * N3);
    memset(C, 0, N1 * N3 * sizeof(double));

    fillMatrix(A, N1, N2);

    // printMatrix(A, N1, N2);
    printf("1\n");

    fillMatrix(B, N2, N3);

    // printMatrix(B, N2, N3);
    printf("2\n");

    struct timespec start;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    for (size_t i = 0; i < N1; ++i) {
        for (size_t j = 0; j < N2; ++j) {
            for (size_t k = 0; k < N3; ++k) {
                C[i*N3 + k] += A[i*N2 + j] * B[j*N3 + k];
            }
        }
    }

    struct timespec end;

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    printf("Time taken: %lf sec\n", end.tv_sec-start.tv_sec + 10e-9*(end.tv_nsec-start.tv_nsec));

    return 0;
}