#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define HEAT_MAX 50
int STEP = 10;

typedef enum STATE {
    SUCCESS,
    ERROR
} STATE;

void matrix_print(double* A, int N) {
    for(int i = 0; i < N*N; ++i) {
        printf("%lf", *(A+i));
        printf(i%N==N-1?"\n":" ");
    }
}
void vector_print(double* b, int N) {
    for(int i = 0; i < N; ++i) {
        printf("%lf", *(b+i));
        printf(i%N==N-1?"\n":" ");
    }
}

int  matrix_inner_shift(int x, int y, int Nx, int Ny) {
    return (y*Nx*Ny + x)*Nx;
}
void matrix_inner_fillIdentical(double* m, int x, int y, int Nx, int Ny) {
    int shift = matrix_inner_shift(x, y, Nx, Ny);
    
    for(int i = 0; i < Nx; ++i) {
        m[(i*Nx*Ny + i) + shift] = 1.0;
    }
}
void matrix_inner_fillThermical(double* m, int x, int y, int Nx, int Ny) {
    int shift = matrix_inner_shift(x, y, Nx, Ny);

    m[shift] = -4.0;

    if(Nx < 2) return;

    m[shift + 1] = 1.0;
    for(int i = 1; i < Nx-1; ++i) {
        m[(i*Nx*Ny + i) - 1 + shift] =  1.0;
        m[(i*Nx*Ny + i)     + shift] = -4.0;
        m[(i*Nx*Ny + i) + 1 + shift] =  1.0;
    }
    m[(Nx-1)*Nx*Ny + shift + Nx-2] =  1.0;
    m[(Nx-1)*Nx*Ny + shift + Nx-1] = -4.0;
}

void matrix_fill(double* m, int Nx, int Ny) {
    if (Nx < 1 || Ny < 1) return;

    matrix_inner_fillThermical(m, 0, 0, Nx, Ny);

    if(Ny < 2) return;

    matrix_inner_fillIdentical(m, 1, 0, Nx, Ny);
    for(int j = 1; j < Ny-1; ++j) {
        matrix_inner_fillIdentical(m, j-1, j, Nx, Ny);
        matrix_inner_fillThermical(m, j,   j, Nx, Ny);
        matrix_inner_fillIdentical(m, j+1, j, Nx, Ny);
    }
    matrix_inner_fillIdentical(m, Ny-2, Ny-1, Nx, Ny);
    matrix_inner_fillThermical(m, Ny-1, Ny-1, Nx, Ny);
}
void vector_fill(double* v, int Nx, int Ny, int count) {
    for(; count > 0; --count) {
        int    index = rand() % (Nx * Ny);
        int    ratio = HEAT_MAX / STEP;
        double heat  = rand() % (2*ratio) - ratio;
        v[index] = heat * STEP;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("Wrong number of arguments, should be 4");
        return ERROR;
    }

    srand(time(NULL));

    int Nx      = atoi(argv[1]);
    int Ny      = atoi(argv[2]);
    int sources = atoi(argv[3]);
    int STEP    = atoi(argv[4]);

    int N = Nx*Ny;

    printf("Nx=%d Ny=%d N=%d\nsources=%d heat_step=%d\n", Nx, Ny, N, sources, STEP);

    double* A = (double*)malloc(sizeof(double) * N * N);
    memset(A, 0, sizeof(double) * N * N);

    matrix_fill(A, Nx, Ny);

    FILE *out = freopen("./data/input.txt", "w", stdout);
    printf("%d\n", N);
    matrix_print(A, N);

    double* b = (double*)malloc(sizeof(double) * N);
    memset(b, 0, sizeof(double) * N);

    vector_fill(b, Nx, Ny, sources);

    vector_print(b, N);

    free(A);
    free(b);

    fclose(out);

    return SUCCESS;
}