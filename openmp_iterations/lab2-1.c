#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _WIN64
    #define LLU "%I64u"
#else
    #define LLU "%lu"
#endif

#define ITERATION_COUNT 10000
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
    uint64_t tr = MIN(N, M);

    for(uint64_t i = 0; i < n; ++i) {
        printf("%lf ", *(A+i));
        i%tr==tr-1?printf("\n"):0;
    }
}

// #======# Utils #======#

STATE scalarDotVectors(double* a, double* b, uint64_t N, double* dst) {
    double sum = 0;
    // #pragma omp shared(sum) for reduction(+:sum) schedule(static)
    for(uint64_t i = 0; i < N; ++i) {
        sum += a[i]*b[i];
    }

    // #pragma omp critical
    *dst = sum;

    return SUCCESS;
}
STATE multMatrixVector(double* A, uint64_t A_N, uint64_t A_M, double* b, uint64_t b_N, double* dst) {
    // #pragma omp for
    for(uint64_t j = 0; j < A_N; ++j) {
        // scalarDotVectors(A + j * A_N, b, b_N, dst + j);
        double sum = 0;
        #pragma omp shared(sum) for reduction(+:sum)
        for(uint64_t i = 0; i < A_N; ++i) {
            sum += A[j * A_N + i]*b[i];
        }

        // #pragma omp critical
        dst[j] = sum;
    }

    return SUCCESS;
}
STATE substractVectors(double* a, double a_k, double* b, double b_k, uint64_t N, double* dst) {
    #pragma omp for schedule(static)
    for(uint64_t i = 0; i < N; ++i) {
        dst[i] = a_k*a[i] - b_k*b[i];
    }

    return SUCCESS;
}

// #======# Least Residual Method #======#

STATE calculateYn(double* A, uint64_t A_N, uint64_t A_M, double* xn, double* b, double* yn, uint64_t yn_N) {

    multMatrixVector(A, A_N, A_M, xn, yn_N, yn);
    substractVectors(yn, 1.0, b, 1.0, yn_N, yn);

    return SUCCESS;
}
STATE calculateTn(double* yn, uint64_t yn_N, double* A, uint64_t A_N, uint64_t A_M, double* dst) {
    double Ayn[yn_N];

    // if(omp_get_thread_num() == 0) {
    //     printf("yn ");
    //     vector_print(yn, yn_N);
    // }

    multMatrixVector(A, A_N, A_M, yn, yn_N, Ayn);

    // if(omp_get_thread_num() == 0) {
    //     printf("Ayn ");
    //     vector_print(Ayn, yn_N);
    // }

    double num, den;
    scalarDotVectors(yn,  Ayn, yn_N, &num);
    scalarDotVectors(Ayn, Ayn, yn_N, &den);

    *dst = num / den;

    return SUCCESS;
}
STATE calculateXnNext(double* xn, double tn, double* yn, uint64_t N) {
    substractVectors(xn, 1, yn, tn, N, xn);

    return SUCCESS;
}
STATE hasSolution(double* yn, double* b, uint64_t N, int* dst) {
    double num, den;
    scalarDotVectors(yn, yn, N, &num);
    scalarDotVectors(b,  b,  N, &den);

    // printf("num=%lf den=%lf\n", num ,den);

    int cur = *dst;

    #pragma omp critical
    *dst = (cur ? cur : num/den < SOLUTION_EPS);

    return SUCCESS;
}

// #======# Main #======#

int main(int argc, char* argv[]) {
    uint64_t N;

    FILE * f = freopen(argv[2], "r", stdin);

    if(!scanf(LLU, &N)) return BAD_INPUT;

    STATE state = SUCCESS;

    double* A = (double*)malloc(sizeof(double) * N * N);
    double b[N];
    double x[N];
    memset(x, 0, sizeof(double)*N);
    double yn[N];

    matrix_fill(A, N, N);
    vector_fill(b, N);

    fclose(f);

    if(state != SUCCESS) {
        perror("");
        return state;
    }

    double tn = 0;
    
    int solution_found = 0;

    omp_set_num_threads(atoi(argv[1]));

    double start = omp_get_wtime();
    #pragma omp parallel shared(A, b, x, yn, solution_found, tn, state)
    {
    
    int i = 0;
    do {
        // printf("%lf ", tn);
        calculateYn(A, N, N, x, b, yn, N);
        // if(i % 1 == 0) {
            // printf("%d yn ", i);
            // vector_print(yn, N);
        // }
        calculateTn(yn, N, A, N, N, &tn);
        // if(omp_get_thread_num() == 0) {
        //     printf("%d tn %lf\n", i, tn);
        // }
        calculateXnNext(x, tn, yn, N);
        // printf("xn ");
        // vector_print(x, N);
        hasSolution(yn, b, N, &solution_found);
        
                // printf("\n%d: ", i);
                // vector_print(x, N);
        // #pragma omp critical
        // solution_found = 1;

        i++;
    } while( !solution_found && i < ITERATION_COUNT );
    }

    double end = omp_get_wtime();

    
    printf("Found solution:\n");
    vector_print(x, N);
    
    char* res_s = strchr(strchr(argv[2], '/') + 1, '/')+1;
    char* res_e = strchr(res_s, '.');
    int res_len = res_e - res_s;

    char prefix[100] = "./data/result/";
    memcpy(prefix+strlen(prefix), res_s, res_len);
    char* dir_name = strcat(prefix, "/");

    struct stat st = {0};

    if (stat(dir_name, &st) == -1) {
        mkdir(dir_name, 0700);
    }

    char postfix[100] = "1.dat";
    char* data_name = strcat(dir_name, postfix); 
    printf("%s\n", prefix);
    FILE* out = fopen(data_name, "wb");
    fwrite(x,sizeof(double)*N,1,out);
    fclose(out);

    printf("Time taken: %lf sec\n", end - start);

    free(A);
    return SUCCESS;
}
