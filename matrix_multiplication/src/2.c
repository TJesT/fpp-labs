#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

void fillMatrix(double *A, int N, int M) {
    for (size_t i = 0; i < N * M; i++) {
        A[i] = (i * (i - 1) + M / 2 - N / 2) % 5 + 1;
    }
}

void printMatrix(double *A, int N, int M) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            printf(" %lf", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int p1 = atoi(argv[1]);
    int p2 = atoi(argv[2]);

    int N1 = 4;
    int N2 = 3;
    int N3 = 4;

    double *A;
    double *B;
    double *C;

    int ndims = 2;
    int dims[2] = {p1, p2};
    int periods[2] = {0, 0};
    int reorder = 0;

    MPI_Comm comm_cartesian2d;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cartesian2d);

    int vertical_dims[2] = {1, 0};
    MPI_Comm comm_vertical;
    MPI_Cart_sub(comm_cartesian2d, vertical_dims, &comm_vertical);

    int horizontal_dims[2] = {0, 1};
    MPI_Comm comm_horizontal;
    MPI_Cart_sub(comm_cartesian2d, horizontal_dims, &comm_horizontal);

    int coords[2];
    MPI_Cart_coords(comm_cartesian2d, rank, ndims, coords);

    if (coords[0] == 0 && coords[1] == 0) {
        A = (double*)malloc(sizeof(double) * N1 * N2);
        fillMatrix(A, N1, N2);

        B = (double*)malloc(sizeof(double) * N3 * N2);
        fillMatrix(B, N2, N3);
        
        C = (double*)malloc(sizeof(double) * N1 * N3);
        memset(C, 0, N1 * N3 * sizeof(double));
    }

    double *Apart = (double*)malloc(sizeof(double) * N1/p1 * N2);
    
    if (coords[1] == 0) {
        MPI_Scatter(A, N1/p1 * N2, MPI_DOUBLE, Apart, N1/p1 * N2, MPI_DOUBLE, 0, comm_vertical);
    }
    
    double *Bpart = (double*)malloc(sizeof(double) * N2 * N3/p2);
    
    if (coords[0] == 0) {
        MPI_Datatype col, type_col;
        MPI_Type_vector(N2, N3/p2, N3, MPI_DOUBLE, &col);
        MPI_Type_commit(&col);

        MPI_Type_create_resized(col, 0, N3/p2 * sizeof(double), &type_col);
        MPI_Type_commit(&type_col);

        MPI_Scatter(B, 1, type_col, Bpart, N3/p2 * N2, MPI_DOUBLE, 0, comm_horizontal);
        
        MPI_Type_free(&col);
        MPI_Type_free(&type_col);
    }
    
    MPI_Bcast(Apart, N1/p1 * N2, MPI_DOUBLE, 0, comm_horizontal);
    MPI_Bcast(Bpart, N3/p2 * N2, MPI_DOUBLE, 0, comm_vertical);

    double *Cpart = (double*)malloc(sizeof(double) * N1/p1 * N3/p2);
    memset(Cpart, 0, sizeof(double) * N1/p1 * N3/p2);

    double start = MPI_Wtime();

    for (size_t i = 0; i < N1/p1; ++i) {
        for (size_t j = 0; j < N2; ++j) {
            for (size_t k = 0; k < N3/p2; ++k) {
                Cpart[i*N3/p2 + k] += Apart[i*N2 + j] * Bpart[j*N3/p2 + k];
            }
        }
    }

    double end = MPI_Wtime();

    MPI_Datatype block, type_block;

    MPI_Type_vector(N1/p1, N3/p2, N3, MPI_DOUBLE, &block);
    MPI_Type_commit(&block);

    MPI_Type_create_resized(block, 0, N3/p2 * sizeof(double), &type_block);
    MPI_Type_commit(&type_block);

    int recv_counts[p1*p2];
    for (size_t i = 0; i < p1*p2; i++) {
        recv_counts[i] = 1;
    }

    int placements[p1*p2];
    for (size_t i = 0; i < p1; ++i) {
        for (size_t j = 0; j < p2; ++j) {
            placements[i*p2 + p1] = i * N1/p1 * p2 + j;
        }
    }

    MPI_Gatherv(Cpart, N1/p1 * N3/p2, MPI_DOUBLE, 
        C, recv_counts, placements, type_block, 0, MPI_COMM_WORLD);

    MPI_Type_free(&block);
    MPI_Type_free(&type_block);

    if (coords[0] == 0 && coords[1] == 0) {
        // printMatrix(C, N1, N3);

        printf("Time taken: %lf sec\n", end-start);

        free(A);
        free(B);
        free(C);
    }
    
    free(Apart);
    free(Bpart);
    free(Cpart);

    MPI_Comm_free(&comm_vertical);
    MPI_Comm_free(&comm_horizontal);
    MPI_Comm_free(&comm_cartesian2d);

    MPI_Finalize();
    
    return 0;
}