#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define min(a, b) (a>b?b:a)

enum {
    SUCCESS,
    MEMORY_ERROR,
    BAD_INPUT
};

// #===# Vector #===#
typedef struct vector {
    long double* data;
    size_t N;
} vector_t;

int vector_init(vector_t* v, size_t N) {
    v->data = (long double*)malloc(sizeof(long double) * N);
    if (v->data == NULL) {
        return MEMORY_ERROR;
    }
    memset(v->data, 0, sizeof(long double) * N);

    v->N = N;

    return SUCCESS;
}

int vector_free(vector_t* v) {
    if(v->data > (long double*)&v) {
        free(v->data);
    }

    return SUCCESS;
}

int vector_fill(vector_t* b) {
    for(size_t i = 0; i < b->N; ++i) {
        if(!scanf("%Lf", b->data+i)) return BAD_INPUT;
    }

    return SUCCESS;
}

void vector_print(vector_t b) {
    for(size_t i = 0; i < b.N; ++i) {
        printf("%Lf ", *(b.data+i));
    }
    printf("\n");
}


// #===# Matrix #===#
typedef struct matrix {
    long double* data;
    size_t N, M;
} matrix_t;

int matrix_init(matrix_t* m, size_t N, size_t M) {
    m->data = (long double*)malloc(sizeof(long double) * N * N);
    if (m->data == NULL) {
        return MEMORY_ERROR;
    }
    memset(m->data, 0, sizeof(long double) * N * M);

    m->N = N;
    m->M = M;

    return SUCCESS;
}

int matrix_free(matrix_t* m) {
    if(m->data > (long double*)&m) {
        free(m->data);
    }

    return SUCCESS;
}

int matrix_fill(matrix_t* A) {
    for(size_t i = 0; i < A->N*A->M; ++i) {
        if(!scanf("%Lf", A->data+i)) return BAD_INPUT;
    }

    return SUCCESS;
}

void matrix_print(matrix_t A) {
    size_t N = A.N * A.M;
    for(size_t i = 0; i < N; ++i) {
        printf("%Lf ", *(A.data+i));
        i%N==N-1?printf("\n"):0;
    }
}

// #======# Utils #======#

int multMatrixVector(matrix_t* A, vector_t* b, vector_t* dst) {
    free_vector(dst);
    if(vector_init(dst, b->N)) {
        return MEMORY_ERROR;
    }

    for(size_t i = 0; i < A->N; ++i) {
        for(size_t j = 0; j < A->M; ++j) {
            dst->data[i] += A->data[i*A->N + j] * b->data[j];
        }
    }

    return SUCCESS;
}

int substractVectors(vector_t* a, vector_t* b, vector_t* dst) {
    size_t N = min(a->N, b->N);
    
    free_vector(dst);

    if(vector_init(dst, N)) {
        return MEMORY_ERROR;
    }

    for(size_t i = 0; i < N; ++i) {
        dst->data[i] = a->data[i] - b->data[i];
    }

    return SUCCESS;
}

int calculateYn(matrix_t* A, vector_t* xn, vector_t* b, vector_t* yn) {
    vector_t tmp;
    multMatrixVector(A, xn, &tmp);

    if (substractVectors(&tmp, b, yn))
    {
        return MEMORY_ERROR;
    }
    

    return SUCCESS;
}

int main(int argc, char* argv[]) {
    size_t N;
    if(!scanf("%lu", &N)) return BAD_INPUT;
    printf("%lu\n", N);

    matrix_t A;
    if (matrix_init(&A, N, N)) {
        return MEMORY_ERROR;
    }
    vector_t b;
    if (vector_init(&b, N) == -1) {
        return MEMORY_ERROR;
    }
    
    fillMatrix(&A, N);
    fillVector(&b, N);
    
    vector_t x;
    if (vector_init(&x, N)) {
        return MEMORY_ERROR;
    }

    printMatrix(A);
    printf("\n");
    printVector(b);

    matrix_free(&A);
    vector_free(&b);
    vector_free(&x);
    
    return SUCCESS;
}
