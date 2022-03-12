#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #===# Vector #===#
typedef struct vector {
    long double* data;
    size_t N;
} vector_t;

void fillVector(vector_t b, size_t N) {
    for(size_t i = 0; i < N; ++i) {
        if(!scanf("%Lf", b.data+i)) return;
    }
}

// #===# Matrix #===#
typedef struct matrix {
    long double* data;
    size_t N, M;
} matrix_t;

void fillMatrix(matrix_t A, size_t N) {
    for(size_t i = 0; i < N*N; ++i) {
        if(!scanf("%Lf", A.data+i)) return;
    }
}


long double* multMatrixVector(long double* A, long double* b, size_t N) {
    long double* c = (long double*)malloc(N*sizeof(long double));
    memset(c, 0, N*sizeof(long double));

    for(size_t i = 0; i < N; ++i) {
        for(size_t j = 0; j < N; ++j) {
            c[i] += A[i*N + j] * b[j];
        }
    }

    return c;
}

long double* substractVectors(long double* a, long double* b, size_t N) {
    long double* c = (long double*)malloc(N*sizeof(long double));
    for(size_t i = 0; i < N; ++i) {
        c[i] = a[i] - b[i];
    }
    return c;
}

double* calculateYn(double* A, double* xn, double* b, int N) {

}

void printMatrix(long double* A, size_t N) {
    for(size_t i = 0; i < N*N; ++i) {
        printf("%Lf ", *(A+i));
        i%N==N-1?printf("\n"):0;
    }
}
void printVector(long double* b, size_t N) {
    for(size_t i = 0; i < N; ++i) {
        printf("%Lf ", *(b+i));
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    size_t N;
    if(!scanf("%lu", &N)) return 0;
    printf("%lu\n", N);

    matrix_t A;
    vector_t b;
    A.data = (long double*)malloc(sizeof(long double) * N * N);
    b.data = (long double*)malloc(sizeof(long double) * N);
    long double* x = (long double*)malloc(sizeof(long double) * N);
    memset(x, 0, N*sizeof(long double));


    fillMatrix(A, N);
    fillVector(b, N);

    printMatrix(A.data, N);
    printf("\n");
    printVector(b.data, N);

    free(A.data);
    free(b.data);
    
    return 0;
}