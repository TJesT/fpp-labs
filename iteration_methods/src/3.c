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
void  matrix_print(double* A, uint64_t N, uint64_t M) {
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
    // MPI_Allreduce(&sum, dst, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    dst[0] = sum;

    return SUCCESS;
}
STATE multMatrixVector(double* A, double* b, uint64_t N, uint64_t M, double* dst, int shift) {
    double sum[N];
    memset(sum, 0, sizeof(double) * N);

    for(uint64_t i = 0; i < M; ++i) {
        for(uint64_t j = 0; j < N; ++j) {
            sum[j] += (A + i*N)[j] * b[i];
        }
    }

    double tmp[N];

    MPI_Allreduce(sum, tmp, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(dst, tmp + shift, sizeof(double) * M);

    return SUCCESS;
}
STATE substractVectors(double* a, double a_k, double* b, double b_k, uint64_t N, uint64_t M, double* dst, int shift) {
    double* tmp = (double*)malloc(sizeof(double)*N);
    memset(tmp, 0, sizeof(double) * N);

    for(uint64_t i = 0; i < M; ++i) {
        tmp[shift + i] = a_k*a[i] - b_k*b[i];
    }
    double buf[N];
    MPI_Allreduce(tmp, buf, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(dst, buf + shift, sizeof(double)*M);

    return SUCCESS;
}

// #======# Least Residual Method #======#

STATE calculateYn(double* A, double* xn, double* b, double* yn, uint64_t N, uint64_t M, int shift) {

    multMatrixVector(A, xn, N, M, yn, shift);

    substractVectors(yn, 1.0, b, 1.0, N, M, yn, shift);

    return SUCCESS;
}
STATE calculateTn(double* yn, double* A, double* tn, uint64_t N, uint64_t M, int shift) {
    double Ayn[M];

    multMatrixVector(A, yn, N, M, Ayn, shift);

    double num_part, den_part;
    
    scalarDotVectors(yn,  Ayn, M, &num_part);
    scalarDotVectors(Ayn, Ayn, M, &den_part);
    
    double num, den;
    
    MPI_Allreduce(&num_part, &num, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&den_part, &den, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    tn[0] = num / den;

    return SUCCESS;
}
STATE calculateXnNext(double* xn, double tn, double* yn, uint64_t N, uint64_t M, int shift) {

    substractVectors(xn, 1, yn, tn, N, M, xn, shift);

    return SUCCESS;
}
bool  hasSolution(double* yn, double* b, uint64_t N, int shift) {
    
    double num_part, den_part;
    scalarDotVectors(yn, yn, N, &num_part);
    scalarDotVectors(b,  b,  N, &den_part);
    
    double num, den;
    MPI_Allreduce(&num_part, &num, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&den_part, &den, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return fabs(num/den) < SOLUTION_EPS;
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

    double *A;
    double *A_part = (double*)malloc(sizeof(double) * N * M);
    double *b;
    double b_part[M];
    double x[N];
    double x_part[M];
    memset(x_part, 0, sizeof(double)*M);
    double yn[N];
    double yn_part[M];


    if(state != SUCCESS) {
        perror("");
        MPI_Finalize();
        free(A_part);
        free(shift_matrix);
        free(shift_vector);
        free(lines_in_task);
        return state;
    }

    if(rank == 0) {
        A = (double*)malloc(sizeof(double) * N * N);
        b = (double*)malloc(sizeof(double) * N);
        matrix_fill(A, N, N);
        vector_fill(b, N);
        fclose(fd);

        if(state != SUCCESS) {
            perror("");
            MPI_Finalize();
            free(A);
            free(b);
            free(A_part);
            free(shift_matrix);
            free(shift_vector);
            free(lines_in_task);
            return state;
        }
    }

    int* elements_count = (int*)malloc(size * sizeof(int));
    for(int i = 0; i < size; ++i) {
        elements_count[i] = N * lines_in_task[i];
    }

    MPI_Scatterv(A, elements_count, shift_matrix, MPI_DOUBLE, 
                A_part, N * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, lines_in_task, shift_vector, MPI_DOUBLE, 
                b_part, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        free(A);
        free(b);
    }

    free(elements_count);

    double tn = 0.0;
    int i = 0;

    double start = MPI_Wtime();
    do {
        calculateYn(A_part, x_part, b_part, yn_part, N, M, shift_vector[rank]);

        calculateTn(yn_part, A_part, &tn, N, M, shift_vector[rank]);

        calculateXnNext(x_part, tn, yn_part, N, M, shift_vector[rank]);

    } while( !hasSolution(yn_part, b_part, M, shift_vector[rank]) && i++ < ITERATION_COUNT );

    double end = MPI_Wtime();

    MPI_Allgatherv(x_part, M, MPI_DOUBLE, x, lines_in_task, shift_vector, MPI_DOUBLE, MPI_COMM_WORLD);

    if(rank == 0) {
        if(i < ITERATION_COUNT) {
            printf("Found solution in %d iterations\n", i);
            // vector_print(x, N);
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

        char postfix[100] = "3.dat";
        char* data_name = strcat(dir_name, postfix); 
        printf("%s\n", prefix);
        FILE* out = fopen(data_name, "wb");
        fwrite(x, sizeof(double)*N, 1, out);
        fclose(out);

        printf("Time taken: %lf sec\n", end-start);
    }

    MPI_Finalize();
    free(A_part);
    free(shift_matrix);
    free(shift_vector);
    free(lines_in_task);
    
    return state;
}
