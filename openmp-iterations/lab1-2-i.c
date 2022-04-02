#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>


#define ITERATION_COUNT 2000000
#define DOUBLE_EPS 1e-6
#define SOLUTION_EPS 1e-14

#define MIN(a,b) (a>b?b:a)
#define EXIT(e) matrix_free(&A);vector_free(&b);vector_free(&x);vector_free(&yn);return e
#define CHANGE_STATE(s,ns) s=(s==SUCCESS?ns:s)

typedef enum STATE {
    SUCCESS = 0,
    MEMORY_ERROR,
    BAD_INPUT
} STATE;

// #===# Vector #===#
typedef struct vector {
    double* data;
    uint64_t N;
} vector_t;

STATE vector_init(vector_t* v, uint64_t N) {
    v->data = (double*)malloc(sizeof(double) * N);
    if (v->data == NULL) {
        return MEMORY_ERROR;
    }
    memset(v->data, 0, sizeof(double) * N);

    v->N = N;

    return SUCCESS;
}
STATE vector_free(vector_t* v) {
    free(v->data);

    return SUCCESS;
}
STATE vector_fill(vector_t* b) {
    for(uint64_t i = 0; i < b->N; ++i) {
        if(!scanf("%lf", b->data+i)) return BAD_INPUT;
    }

    return SUCCESS;
}
void vector_print(vector_t b) {
    for(uint64_t i = 0; i < b.N; ++i) {
        printf("%lf ", *(b.data+i));
    }
    printf("\n");
}
int vector_isEqual(vector_t a, vector_t b) {
    if(a.N != b.N) {
        return 0;
    }
    for(uint64_t i = 0; i < a.N; ++i) {
        if(fabs(a.data[i] - b.data[i]) > DOUBLE_EPS) {
            return 0;
        }
    }

    return 1;
}

// #===# Matrix #===#
typedef struct matrix {
    double* data;
    uint64_t N, M;
} matrix_t;

STATE matrix_init(matrix_t* m, uint64_t N, uint64_t M) {
    double* tmp = (double*)malloc(sizeof(double) * N * M);
    m->data = tmp;
    if (m->data == NULL) {
        return MEMORY_ERROR;
    }
    memset(m->data, 0, sizeof(double) * N * M);

    m->N = N;
    m->M = M;

    return SUCCESS;
}
STATE matrix_free(matrix_t* m) {
    free(m->data);
    
    return SUCCESS;
}
STATE matrix_fill(matrix_t* A) {
    for(uint64_t i = 0; i < A->N*A->M; ++i) {
        if(!scanf("%lf", A->data+i)) return BAD_INPUT;
    }

    return SUCCESS;
}
void matrix_print(matrix_t A) {
    uint64_t N = A.N * A.M;
    uint64_t tr = MIN(A.N, A.M);

    for(uint64_t i = 0; i < N; ++i) {
        printf("%lf ", *(A.data+i));
        i%tr==tr-1?printf("\n"):0;
    }
}

// #======# Utils #======#

STATE scalarDotVectors(vector_t* a, vector_t* b, double* dst) {
    uint64_t N = MIN(a->N, b->N);

    double sum = 0;
    #pragma omp for reduction(+:sum) schedule(static, N / omp_get_num_threads)
    for(uint64_t i = 0; i < N; ++i) {
        sum += a->data[i]*b->data[i];
    }

    *dst = sum;

    return SUCCESS;
}
STATE multMatrixVector(matrix_t* A, vector_t* b, vector_t* dst) {
    #pragma omp for
    for(uint64_t i = 0; i < A->N; ++i) {
        for(uint64_t j = 0; j < A->M; ++j) {
            dst->data[i] += A->data[i*A->N + j] * b->data[j];
        }
    }

    return SUCCESS;
}
STATE substractVectors(vector_t* a, double a_k, vector_t* b, double b_k, vector_t* dst) {
    uint64_t N = MIN(a->N, b->N);

    #pragma omp for
    for(uint64_t i = 0; i < N; ++i) {
        dst->data[i] = a_k*a->data[i] - b_k*b->data[i];
    }

    return SUCCESS;
}

// #======# Least Residual Method #======#

STATE calculateYn(matrix_t* A, vector_t* xn, vector_t* b, vector_t* yn) {
    vector_t tmp;
    if(vector_init(&tmp, xn->N) == MEMORY_ERROR) {
        return MEMORY_ERROR;
    }

    multMatrixVector(A, xn, &tmp);

    substractVectors(&tmp, 1.0, b, 1.0, yn);

    vector_free(&tmp);

    return SUCCESS;
}
STATE calculateTn(vector_t* yn, matrix_t* A, double* dst) {
    vector_t Ayn;
    if(vector_init(&Ayn, yn->N) == MEMORY_ERROR) {
        return MEMORY_ERROR;
    }

    multMatrixVector(A, yn, &Ayn);

    double num, den;
    scalarDotVectors(yn, &Ayn, &num);
    scalarDotVectors(&Ayn, &Ayn, &den);

    *dst = num / den;

    vector_free(&Ayn);

    return SUCCESS;
}
STATE calculateXnNext(vector_t* xn, double tn, vector_t* yn) {
    vector_t xn_next;
    if(vector_init(&xn_next, xn->N) == MEMORY_ERROR) {
        return MEMORY_ERROR;
    }

    substractVectors(xn, 1, yn, tn, &xn_next);

    vector_free(xn);

    xn->data = xn_next.data;
    xn->N    = xn_next.N;

    return SUCCESS;
}
int hasSolution(vector_t* yn, vector_t* b) {
    double num, den;
    scalarDotVectors(yn, yn, &num);
    scalarDotVectors(b, b, &den);

    return num/den < SOLUTION_EPS;
}

// #======# Main #======#

int main(int argc, char* argv[]) {
    uint64_t N;
    if(!scanf("%I64u", &N)) return BAD_INPUT;

    STATE state = SUCCESS;

    matrix_t A;
    vector_t b;
    vector_t x;
    vector_t yn;

    CHANGE_STATE(state, matrix_init(&A, N, N));
    CHANGE_STATE(state, vector_init(&b, N));
    CHANGE_STATE(state, vector_init(&x, N));
    CHANGE_STATE(state, vector_init(&yn, N));

    if(state != SUCCESS) {
        perror("");
        EXIT(state);
    }

    CHANGE_STATE(state, matrix_fill(&A));
    CHANGE_STATE(state, vector_fill(&b));

    if(state != SUCCESS) {
        perror("");
        EXIT(state);
    }

    double tn;
    int i = 0;

    double start = omp_get_wtime();
    #pragma omp parallel num_threads(24) shared(A, x, b, yn, tn, state, i)
    {
    do {
        CHANGE_STATE(state, calculateYn(&A, &x, &b, &yn));
        CHANGE_STATE(state, calculateTn(&yn, &A, &tn));
        CHANGE_STATE(state, calculateXnNext(&x, tn, &yn));
    } while( state == SUCCESS && i++ < ITERATION_COUNT && !hasSolution(&yn, &b) );
    }

    double end = omp_get_wtime();

    if(state == SUCCESS && i < ITERATION_COUNT) {
        printf("Found solution:\n");
        vector_print(x);

        multMatrixVector(&A, &x, &yn);

        if(vector_isEqual(yn, b)) {
            printf("Solution is correct!\n");
        } else {
            printf("Solution is incorrect :c\n");
            printf("initial constants:\n");
            vector_print(b);
            printf("testing result:\n");
            vector_print(yn);
        }
    } else {
        printf("Hadn't find solution: in %I64u iterations :c\n", ITERATION_COUNT);
    }

    printf("Time taken: %lf sec\n", end-start);

    EXIT(state);
}
