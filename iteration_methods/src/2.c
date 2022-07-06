#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _WIN64
    #define LLU "%I64u"
#else
    #define LLU "%lu"
#endif

#define ITERATION_COUNT 1000000
#define DOUBLE_EPS 1e-5
#define SOLUTION_EPS 1e-14

#define MIN(a,b) (a>b?b:a)

typedef enum STATE {
    SUCCESS,
    MEMORY_ERROR,
    BAD_INPUT
} STATE;

// #===# Vector #===#
STATE vector_fill(double* b, uint64_t N) {
    for(uint64_t i = 0; i < N; ++i) {
        if(!scanf("%lf", b+i)) return BAD_INPUT;
    }

    return SUCCESS;
}
void vector_print(double* b, uint64_t N) {
    for(uint64_t i = 0; i < N; ++i) {
        printf("%lf ", *(b+i));
    }
    printf("\n");
}

// #===# Matrix #===#
STATE matrix_fill(double* A, uint64_t N, uint64_t M) {
    for(uint64_t i = 0; i < N*M; ++i) {
        if(!scanf("%lf", A+i)) return BAD_INPUT;
    }

    return SUCCESS;
}
void matrix_print(double* A, uint64_t N, uint64_t M) {
    uint64_t n = N * M;
    uint64_t tr = N;

    for(uint64_t i = 0; i < n; ++i) {
        printf("%lf ", *(A+i));
        i%tr==tr-1?printf("\n"):0;
    }
}

// #======# Utils #======#

STATE scalarDotVectors(double* a, double* b, uint64_t N, double* dst) {
    double sum = 0;
    for(uint64_t i = 0; i < N; ++i) {
        sum += a[i]*b[i];
    }

    *dst = sum;

    return SUCCESS;
}
STATE multMatrixVector(double* A, uint64_t A_N, uint64_t A_M, double* b, uint64_t b_N, int shift, double* dst) {
    double tmp[A_N];
    memset(tmp, 0, sizeof(double)*A_N);
    for(uint64_t j = 0; j < A_M; ++j) {
        scalarDotVectors(A + j * A_N, b, b_N, tmp + shift + j);
    }
    MPI_Allreduce(tmp, dst, A_N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return SUCCESS;
}
STATE substractVectors(double* a, double a_k, double* b, double b_k, uint64_t N, uint64_t M, int shift, double* dst) {
    double* a_part = a + shift;
    double* b_part = b + shift;
    
    double tmp[N];
    memset(tmp, 0, sizeof(double) * N);

    for(uint64_t i = 0; i < M; ++i) {
        tmp[shift + i] = a_k*a_part[i] - b_k*b_part[i];
    }
    MPI_Allreduce(tmp, dst, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return SUCCESS;
}

// #======# Least Residual Method #======#

STATE calculateYn(double* A, uint64_t A_N, uint64_t A_M, double* xn, double* b, double* yn, uint64_t yn_N, int* shift_vector, int rank) {

    multMatrixVector(A, A_N, A_M, xn, yn_N, shift_vector[rank], yn);

    substractVectors(yn, 1.0, b, 1.0, yn_N, A_M, shift_vector[rank], yn);

    return SUCCESS;
}
STATE calculateTn(double* yn, uint64_t yn_N, double* A, uint64_t A_N, uint64_t A_M, double* dst, int* shift_vector, int rank) {
    double Ayn[yn_N];

    multMatrixVector(A, A_N, A_M, yn, yn_N, shift_vector[rank], Ayn);
    
    double num_part, den_part;
    double num,      den;
    scalarDotVectors(yn, Ayn, yn_N, &num_part);
    MPI_Allreduce(&num_part, &num, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    scalarDotVectors(Ayn, Ayn, yn_N, &den_part);
    MPI_Allreduce(&den_part, &den, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    dst[0] = num/den;

    return SUCCESS;
}
STATE calculateXnNext(double* xn, double tn, double* yn, uint64_t N, uint64_t M, int* shift_vector, int rank) {
    substractVectors(xn, 1, yn, tn, N, M, shift_vector[rank], xn);

    return SUCCESS;
}
bool hasSolution(double* yn, double* b, uint64_t N) {
    double num, den;
    scalarDotVectors(yn, yn, N, &num);
    scalarDotVectors(b,  b,  N, &den);

    // printf("num=%lf den=%lf\n", num ,den);

    return num/den < SOLUTION_EPS;
}

// #======# Main #======#

int main(int argc, char* argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    FILE * fd;

    uint64_t N, M;
    if (rank == 0) {
        fd = freopen(argv[1], "r", stdin);
        if(!scanf(LLU, &N)) return BAD_INPUT;
    }

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
    M = lines_in_task[rank];

    double* A;
    double* A_part = (double*)malloc(sizeof(double) * N * M);
    double b[N];
    double x[N];
    memset(x, 0, sizeof(double)*N);
    double yn[N];

    if(state != SUCCESS) {
        perror("");
        MPI_Finalize();
        free(A_part);
        return state;
    }

    if(rank == 0) {
        A = (double*)malloc(sizeof(double) * N * N);
        matrix_fill(A, N, N);
        vector_fill(b, N);
        fclose(fd);

        if(state != SUCCESS) {
            perror("");
            MPI_Finalize();
            free(A_part);
            return state;
        }
    }

    int* elements_count = malloc(size * sizeof(int));
    for(int i = 0; i < size; ++i) {
        elements_count[i] = N * lines_in_task[i];
    }

    MPI_Scatterv(A, elements_count, shift_matrix, MPI_DOUBLE, 
                A_part, N * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(A);
    }

    free(elements_count);

    double tn = 0.0;
    int i = 0;

    double start = MPI_Wtime();
    do {
        calculateYn(A_part, N, M, x, b, yn, N, shift_vector, rank);

        calculateTn(yn, N, A_part, N, M, &tn, shift_vector, rank);

        calculateXnNext(x, tn, yn, N, M, shift_vector, rank);

        MPI_Allgatherv(x+shift_vector[rank], M, MPI_DOUBLE, x, lines_in_task, 
                    shift_vector, MPI_DOUBLE, MPI_COMM_WORLD);

    } while( !hasSolution(yn, b, N) && i++ < ITERATION_COUNT );
    
    double end = MPI_Wtime();

    if(rank == 0) {
        if(i < ITERATION_COUNT) {
            printf("Found solution:\n");
            vector_print(x, N);
        } else {
            printf("Hadn't find solution: in %d iterations :c\n", ITERATION_COUNT);
        }
        char* res_s = strchr(strchr(argv[1], '/') + 1, '/')+1;
        char* res_e = strchr(res_s, '.');
        int res_len = res_e - res_s;

        char prefix[100] = "/home/tjest/Рабочий стол/OPP/iteration_methods/data/result/";
        memcpy(prefix+strlen(prefix), res_s, res_len);
        char* dir_name = strcat(prefix, "/");

        struct stat st = {0};

        if (stat(dir_name, &st) == -1) {
            mkdir(dir_name, 0700);
        }

        char postfix[100] = "2.dat";
        char* data_name = strcat(dir_name, postfix); 
        printf("%s\n", prefix);
        FILE* out = fopen(data_name, "wb");
        fwrite(x, sizeof(double)*N, 1, out);
        fclose(out);

        printf("Time taken: %lf sec\n", end-start);
    }

    MPI_Finalize();

    free(A_part);

    return state;
}
