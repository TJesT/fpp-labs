#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define max(a, b) (a>b?a:b)

#define SEED (0x3a7eb429)
#define ITERATIONS_COUNT (100)

size_t goi_hash(bool *field, int cols, int rows, size_t seed) {
    for (int y = 0; y != 8; ++y) {
        for (int x = 0; x != 8; ++x) {
            seed = (seed >> 1) | (seed << (sizeof(size_t) * 8 - 1));
            seed ^= field[y*cols + x] * 0xee6b2807;
        }
    }
    seed *= 0xee6b2807;
    seed ^= seed >> 16;

    fprintf(stdout, "hash: %zu\n", seed);

    return seed;
}

void goi_show(bool *field, int rows, int cols) {
    for(int y = 0 ; y < rows; ++y) {
        for(int x = 0; x < cols; ++x) {
            fprintf(stdout, "%c", field[y*cols + x]?'#':'.');
        }
        fprintf(stdout, "%c", '\n');
    }
}

void goi_init(bool *field, int rows, int cols) {
    if(!field) {
        fprintf(stderr, "Must allocate field first\n");
        return;
    }

    field[0*cols + 1] = true;
    field[1*cols + 2] = true;
    field[2*cols + 0] = true;
    field[2*cols + 1] = true;
    field[2*cols + 2] = true;
}

bool goi_rule(bool cell, int count) {
    bool a = cell;
    // Condition if alive
    bool b = count < 2 || count > 3;
    // Condition if dead
    bool c = count == 3;

    return (a & ~b) | (~a & c);
}

void goi_start(bool *field, int rows, int cols) {
    fprintf(stdout, "field ptr: %p, rows: %i, cols: %i\n", field, rows, cols);
    
    goi_init(field, rows, cols);

    size_t states[ITERATIONS_COUNT];
    memset(states, 0, sizeof(size_t)*ITERATIONS_COUNT);

    int iteration = 0;
    
    bool stop = false;
    while(!stop) {
        states[iteration] = goi_hash(field, cols, rows, SEED);
        goi_show(field, rows, cols);

        free(field);

        field = (bool*)malloc(sizeof(bool) * rows * cols);
        memset(field, 0, sizeof(bool) * rows * cols);

        ++iteration;
        if (iteration >= ITERATIONS_COUNT) stop |= true;
    }

    free(field);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Must specify exactly 2 arguments\n");
        return 0;
    }

    int rows = strtol(argv[1], NULL, 10);
    int cols = strtol(argv[2], NULL, 10);

    rows = max(3, rows);
    cols = max(3, cols);

    bool *field = (bool*)malloc(sizeof(bool) * rows * cols);
    memset(field, 0, rows*cols*sizeof(bool));

    goi_start(field, rows, cols);

    return EXIT_SUCCESS;
}