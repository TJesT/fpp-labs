#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#define max(a, b) (a>b?a:b)

#define SEED (0x3a7eb429)
#define ITERATIONS_COUNT (100000)
#define SWAP(a,b,type) {type temporary_variable=a;a=b;b=temporary_variable;}

size_t goi_hash(bool *field, int cols, int rows, size_t seed) {
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
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

    // field[0*cols + 0] = true;
    // field[(cols-1)*cols + rows-1] = true;
    // field[0*cols + rows-1] = true;
    // field[(cols-1)*cols + 0] = true;

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

bool goi_isrepeated(size_t *states, int len) {
    size_t hash = states[len-1];
    for (int i = 0; i < len-1; ++i) {
        if (hash == states[i]) return true;
    }

    return false;
}

void goi_count_neighbors(int x, int y, bool *field, int rows, int cols, int *count) {
    count[0] = 0;
    for (int dy = -1; dy < 2; ++dy) {
        for (int dx = -1; dx < 2; ++dx) {
            count[0] += field[(y + dy)*cols + (x + dx + cols) % cols];
        }
    }
    count[0] -= field[y*cols + x];
}

void goi_step(bool *field, int rows, int cols, bool *next, 
    void (*count_neighbors)(int x, int y, bool *field, int rows, int cols, int *count)) {
    
    // fprintf(stdout, "=======\n");
    int count;
    for(int y = 1; y < rows - 1; ++y) {
        for (int x = 0; x < cols; ++x) {
            count_neighbors(x, y, field, rows, cols, &count);
            // fprintf(stdout, "%d", count);
            next[y*cols + x] = goi_rule(field[y*cols + x], count);
            count = 0;
        }
        // fprintf(stdout, "\n");
    }
}

void goi_std_step(bool *field, int rows, int cols, bool *next) {
    goi_step(field, rows, cols, next, goi_count_neighbors);
}

void goi_start(bool *field, int rows, int cols) {
    fprintf(stdout, "field ptr: %p, rows: %i, cols: %i\n", field, rows, cols);
    
    goi_init(field, rows, cols);

    bool *matrix = (bool*)malloc(sizeof(bool) * (rows + 2) * cols);
    memcpy(matrix + cols, field, sizeof(bool) * rows * cols);
    field = matrix + cols;
    memcpy(matrix,                 field + (rows-1)*cols, sizeof(bool) * cols);
    memcpy(matrix + (rows+1)*cols, field,                 sizeof(bool) * cols);    

    bool *next_matrix = (bool*)malloc(sizeof(bool) * (rows + 2) * cols);
    memset(next_matrix, 0, sizeof(bool) * (rows + 2) * cols);
    bool *next = next_matrix + cols;

    size_t states[ITERATIONS_COUNT];
    memset(states, 0, sizeof(size_t)*ITERATIONS_COUNT);

    int iteration = 0;
    
    // for parallel replace with stop_vector
    bool stop = false;
    while(!stop) {
        states[iteration] = goi_hash(field, cols, rows, SEED);
        goi_show(field, rows, cols);

        if (goi_isrepeated(states, iteration+1)) stop |= true;

        if (stop) break;
        
        // fprintf(stdout, "===========\n");
        // goi_show(matrix, rows+2, cols);

        goi_std_step(matrix, rows+2, cols, next_matrix);
        // goi_std_step(matrix, 3, cols, next_matrix);
        // goi_std_step(matrix + (rows+1)*cols, 3, cols, next_matrix + (rows+1)*cols);

        SWAP(field, next, bool*);
        SWAP(matrix, next_matrix, bool*);

        memcpy(matrix,                 field + (rows-1)*cols, sizeof(bool) * cols);
        memcpy(matrix + (rows+1)*cols, field,                 sizeof(bool) * cols);
        // printf("\n\n\n\n\n\n\n\n\n\n\n\n");
        usleep(10000);

        ++iteration;
        if (iteration >= ITERATIONS_COUNT) stop |= true;
    }

    fprintf(stdout, "Ended after %i interations\n", iteration);

    free(matrix);
    free(next_matrix);
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

    free(field);

    return EXIT_SUCCESS;
}