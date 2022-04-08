#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>

#ifdef _WIN64
    #define LLU "%I64u"
#else
    #define LLU "%lu"
#endif

#define ITERATION_COUNT 1000000
#define DOUBLE_EPS 1e-5
#define SOLUTION_EPS 1e-14

#define MIN(a,b) (a>b?b:a)
#define EXIT(e) matrix_free(&A);matrix_free(&A_part);vector_free(&b);vector_free(&b_part);vector_free(&x);vector_free(&yn);return e
#define CHANGE_STATE(s,ns) s=(s==SUCCESS?ns:s)

typedef enum STATE {
    SUCCESS,
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
void  vector_print(vector_t b) {
    for(uint64_t i = 0; i < b.N; ++i) {
        printf("%lf ", *(b.data+i));
    }
    printf("\n");
}
bool  vector_isEqual(vector_t a, vector_t b) {
    if(a.N != b.N) {
        return false;
    }
    for(uint64_t i = 0; i < a.N; ++i) {
        if(fabs(a.data[i] - b.data[i]) > DOUBLE_EPS) {
            return false;
        }
    }

    return true;
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
void  matrix_print(matrix_t A) {
    uint64_t N = A.N * A.M;
    uint64_t tr = MIN(A.N, A.M);

    for(uint64_t i = 0; i < N; ++i) {
        printf("%lf ", *(A.data+i));
        i%tr==tr-1?printf("\n"):0;
    }
}

// #======# Utils #======#

STATE scalarDotVectors(vector_t* a, vector_t* b, int* lines_in_task, int* shift_vector, int rank, double*   dst) {
    double sum = 0;
    
    int N = lines_in_task[rank];
    double sum_part = 0.;
    for(uint64_t i = 0; i < N; ++i) {
        sum_part += a->data[i]*b->data[i];
    }
    MPI_Allreduce(&sum_part, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    *dst = sum;

    return SUCCESS;
}
STATE multMatrixVector(matrix_t* A, vector_t* b, int* lines_in_task, int* shift_vector, int rank, vector_t* dst) {
    vector_t tmp;
    if(vector_init(&tmp, b->N) == MEMORY_ERROR) {
        return MEMORY_ERROR;
    }

    double* dst_part = dst->data;
    for(uint64_t i = 0; i < A->M; ++i) {
        memcpy(tmp.data, A->data + i*A->N, sizeof(double) * A->N);
        scalarDotVectors(&tmp, b, lines_in_task, shift_vector, rank, dst_part+i);
    }

    return SUCCESS;
}
STATE substractVectors(vector_t* a, double a_k, vector_t* b, double b_k, int* lines_in_task, int* shift_vector, int rank, vector_t* dst) {
    uint64_t N = lines_in_task[rank];

    for(uint64_t i = 0; i < N; ++i) {
        dst->data[i] = a_k*a->data[i] - b_k*b->data[i];
    }

    return SUCCESS;
}

// #======# Least Residual Method #======#

STATE calculateYn(matrix_t* A, vector_t* xn, vector_t* b, vector_t* yn, int* lines_in_task, int* shift_vector, int rank) {
    vector_t tmp;
    if(vector_init(&tmp, xn->N) == MEMORY_ERROR) {
        return MEMORY_ERROR;
    }

    multMatrixVector(A, xn, lines_in_task, shift_vector, rank, &tmp);
    substractVectors(&tmp, 1.0, b, 1.0, lines_in_task, shift_vector, rank, yn);

    vector_free(&tmp);

    return SUCCESS;
}
STATE calculateTn(vector_t* yn, matrix_t* A, double* dst,               int* lines_in_task, int* shift_vector, int rank) {
    vector_t Ayn;
    if(vector_init(&Ayn, yn->N) == MEMORY_ERROR) {
        return MEMORY_ERROR;
    }

    multMatrixVector(A, yn, lines_in_task, shift_vector, rank, &Ayn);
    //vector_print(Ayn);

    double num, den;
    scalarDotVectors(yn,   &Ayn, lines_in_task, shift_vector, rank, &num);
    scalarDotVectors(&Ayn, &Ayn, lines_in_task, shift_vector, rank, &den);
    //printf("num=%lf den=%lf\n", num, den);

    dst[0] = num;
    dst[1] = den;

    vector_free(&Ayn);

    return SUCCESS;
}
STATE calculateXnNext(vector_t* xn, double tn, vector_t* yn,            int* lines_in_task, int* shift_vector, int rank) {
    vector_t xn_next;
    if(vector_init(&xn_next, xn->N) == MEMORY_ERROR) {
        return MEMORY_ERROR;
    }

    substractVectors(xn, 1, yn, tn, lines_in_task, shift_vector, rank, &xn_next);

    vector_free(xn);

    xn->data = xn_next.data;
    xn->N    = xn_next.N;

    return SUCCESS;
}
bool  hasSolution(vector_t* yn, vector_t* b,                            int* lines_in_task, int* shift_vector, int rank) {
    double num, den;
    scalarDotVectors(yn, yn, lines_in_task, shift_vector, rank, &num);
    scalarDotVectors(b,  b,  lines_in_task, shift_vector, rank, &den);

    return num/den < SOLUTION_EPS;
}

// #======# Main #======#

int main(int argc, char* argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    uint64_t N;
    if(!scanf(LLU, &N)) return BAD_INPUT;

    MPI_Bcast(&N, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    int* lines_in_task = (int*)malloc(size * sizeof(int));
    memset(lines_in_task, 0, sizeof(int)*size);

    int lines_count = N / size;
    int untracked   = N % size;
    for(int i = 0; i < size; ++i) {
        lines_in_task[i] = lines_count + (i < untracked ? 1 : 0);
    }

    int* shift_matrix = (int*)malloc(size * sizeof(int));
    int* shift_vector = (int*)malloc(size * sizeof(int));

    shift_matrix[0] = shift_vector[0] = 0;
    for(int i = 1; i < size; ++i) {
        int lines_in_prev = lines_in_task[i-1];
        shift_vector[i] = shift_vector[i-1] + lines_in_prev;
        shift_matrix[i] = shift_matrix[i-1] + lines_in_prev * N;
    }

    STATE state = SUCCESS;

    matrix_t A;
    matrix_t A_part;
    vector_t b;
    vector_t b_part;
    vector_t x;
    vector_t x_part;
    vector_t yn;
    vector_t yn_part;

    CHANGE_STATE(state, matrix_init(&A_part, N, lines_in_task[rank]));
    CHANGE_STATE(state, matrix_init(&A, N, N));
    CHANGE_STATE(state, vector_init(&b_part, lines_in_task[rank]));
    CHANGE_STATE(state, vector_init(&b, N));
    CHANGE_STATE(state, vector_init(&x_part, lines_in_task[rank]));
    CHANGE_STATE(state, vector_init(&x, N));
    CHANGE_STATE(state, vector_init(&yn_part, lines_in_task[rank]));
    CHANGE_STATE(state, vector_init(&yn, N));

    if(state != SUCCESS) {
        perror("");
        MPI_Finalize();
        EXIT(state);
    }

    if(rank == 0) {
        CHANGE_STATE(state, matrix_fill(&A));
        CHANGE_STATE(state, vector_fill(&b));

        if(state != SUCCESS) {
            perror("");
            MPI_Finalize();
            EXIT(state);
        }
    }

    int* elements_count = malloc(size * sizeof(int));
    for(int i = 0; i < size; ++i) {
        elements_count[i] = N * lines_in_task[i];
    }

    MPI_Scatterv(A.data, elements_count, shift_matrix, MPI_DOUBLE, 
                A_part.data, N * lines_in_task[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b.data, lines_in_task, shift_vector, MPI_DOUBLE, 
                b_part.data, lines_in_task[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(elements_count);

    double tn_part[2] = {0., 0.};
    double tn_tmp[2]  = {0., 0.};
    double tn         = 0.;
    int i = 0;

    double start = MPI_Wtime();
    do {
        CHANGE_STATE(state, calculateYn(&A_part, &x_part, &b_part, &yn_part, lines_in_task, shift_vector, rank));
        CHANGE_STATE(state, calculateTn(&yn_part, &A_part, tn_part, lines_in_task, shift_vector, rank));
        MPI_Allreduce(tn_part, tn_tmp, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tn = tn_tmp[0] / tn_tmp[1];
        CHANGE_STATE(state, calculateXnNext(&x_part, tn, &yn_part, lines_in_task, shift_vector, rank));
        MPI_Allgatherv(x_part.data, lines_in_task[rank], MPI_DOUBLE, x.data, lines_in_task, shift_vector, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(b_part.data, lines_in_task[rank], MPI_DOUBLE, b.data, lines_in_task, shift_vector, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(yn_part.data, lines_in_task[rank], MPI_DOUBLE, yn.data, lines_in_task, shift_vector, MPI_DOUBLE, MPI_COMM_WORLD);
    } while( state == SUCCESS && i++ < ITERATION_COUNT && !hasSolution(&yn, &b, lines_in_task, shift_vector, rank) );

    double end = MPI_Wtime();

    if(rank == 0) {
        if(i < ITERATION_COUNT) {
            printf("Found solution:\n");
            vector_print(x);

            multMatrixVector(&A, &x, lines_in_task, shift_vector, rank, &yn);

            if(vector_isEqual(yn, b)) {
                printf("Solution is correct !\n");
            } else {
                printf("Solution is incorrect :c\n");
                // printf("initial constants:\n");
                // vector_print(b);
                // printf("testing result:\n");
                // vector_print(yn);
            }
        } else {
            printf("Hadn't find solution: in %d iterations :c\n", ITERATION_COUNT);
        }
        FILE* out = fopen("./data/result3-30-30.dat", "wb");
        fwrite(x.data, sizeof(double)*x.N, 1, out);
        fclose(out);

        printf("Time taken: %lf sec\n", end-start);
    }

    MPI_Finalize();

    EXIT(state);
}
