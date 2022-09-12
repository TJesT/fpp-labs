#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>

#define max(a, b) (a>b?a:b)

#define SEED (0x3a7eb429)
#define ITERATIONS_COUNT (1000000)
#define SWAP(a,b,type) {type temporary_variable=a;a=b;b=temporary_variable;}

typedef enum fseed {
    BLOCK,
    GLIDER
} field_seed_t;

int rank = -1, size = -1;

void goi_parallel_configure(int *shifts, int *counts, 
    int *neighbor_bot, int *neighbor_top, int rows, int cols) {
    
    neighbor_bot[0] = (rank + 1) % size;
    neighbor_top[0] = (rank + size - 1) % size;

    if (rank == -1 || size == -1) {
        fprintf(stderr, "Run MPI_Init first\n");
        return;
    }

    int default_rows_count = rows / size;
    for(int i = 0; i < size; ++i) {
        counts[i] = default_rows_count * cols;
    }
    
    int additional_rows    = rows % size;
    for(int i = 0; i < additional_rows; ++i) {
        counts[i] += cols;
    }

    shifts[0] = 0;
    for(int i = 1; i < size; ++i) {
        shifts[i] = shifts[i-1] + counts[i-i];
    }
}

size_t goi_hash(bool *field, int cols, int rows, size_t seed) {
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            seed = (seed >> 1) | (seed << (sizeof(size_t) * 8 - 1));
            seed ^= field[y*cols + x] * 0xee6b2807;
        }
    }
    seed *= 0xee6b2807;
    seed ^= seed >> 16;

    // usleep(rank * 1000);
    // fprintf(stdout, "hash: %zu\n", seed);
    // usleep((size-rank-1) * 1000);

    return seed;
}

void goi_show(bool *field, int rows, int cols) {
    usleep(rank * 1000);
    fprintf(stdout, "rank: %d\n", rank);
    for(int y = 0 ; y < rows; ++y) {
        for(int x = 0; x < cols; ++x) {
            fprintf(stdout, "%c", field[y*cols + x]?'#':'.');
        }
        fprintf(stdout, "\n");
    }
    usleep((size-rank-1) * 1000);
}

void goi_init(bool *field, int rows, int cols, field_seed_t seed) {
    if(!field) {
        fprintf(stderr, "Must allocate field first\n");
        return;
    }

    switch (seed) {
    case BLOCK:
        field[0*cols + 0] = true;
        field[(cols-1)*cols + rows-1] = true;
        field[0*cols + rows-1] = true;
        field[(cols-1)*cols + 0] = true;
        break;
    case GLIDER:
        field[0*cols + 1] = true;
        field[1*cols + 2] = true;
        field[2*cols + 0] = true;
        field[2*cols + 1] = true;
        field[2*cols + 2] = true;
        break;
    default:
        break;
    }
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

void goi_start(bool *field, int rows, int cols, field_seed_t seed) {
    if (rank == 0) {
        fprintf(stdout, "field ptr: %p, rows: %i, cols: %i\n", field, rows, cols);
        
        goi_init(field, rows, cols, seed);
    }

    int neighbor_top, neighbor_bot;
    
    int shifts[size];
    int counts[size];

    goi_parallel_configure(shifts, counts, &neighbor_bot, &neighbor_top, rows, cols);

    int my_rows = counts[rank] / cols;

    bool *matrix = (bool*)malloc(sizeof(bool) * (my_rows + 2) * cols);
    
    MPI_Scatterv(field, counts, shifts, MPI_C_BOOL, 
        matrix + cols, counts[rank], MPI_C_BOOL, 0, MPI_COMM_WORLD);
    field = matrix + cols;
    memcpy(matrix,                 field + (my_rows-1)*cols, sizeof(bool) * cols);
    memcpy(matrix + (my_rows+1)*cols, field,                 sizeof(bool) * cols);

    bool *next_matrix = (bool*)malloc(sizeof(bool) * (my_rows + 2) * cols);
    memset(next_matrix, 0, sizeof(bool) * (my_rows + 2) * cols);
    bool *next = next_matrix + cols;

    size_t states[ITERATIONS_COUNT];
    memset(states, 0, sizeof(size_t)*ITERATIONS_COUNT);

    int iteration = 0;
    
    bool stop_vector[size];
    memset(stop_vector, 0, sizeof(bool)*size);

    bool stop_me = false;

    bool stop = false;
    while(!stop) {
        states[iteration] = goi_hash(field, cols, my_rows, SEED);
        // goi_show(field, my_rows, cols);

        MPI_Request top_send_req, bot_send_req;
        MPI_Isend(field, cols, MPI_C_BOOL, 
            neighbor_top, 1, MPI_COMM_WORLD, &top_send_req);
        MPI_Isend(field + counts[rank] - cols, cols, MPI_C_BOOL, 
            neighbor_bot, 0, MPI_COMM_WORLD, &bot_send_req);
        
        MPI_Request top_rec_req, bot_rec_req;
        MPI_Irecv(matrix, cols, MPI_C_BOOL, 
            neighbor_top, 0, MPI_COMM_WORLD, &top_rec_req);
        MPI_Irecv(field + counts[rank], cols, MPI_C_BOOL, 
            neighbor_bot, 1, MPI_COMM_WORLD, &bot_rec_req);

        if(iteration > 0) {
            stop_me = states[iteration] == states[iteration-1];
        }

        // printf("rank: %d; %d\n", rank, stop_me);

        MPI_Request stop_recv_req;
        MPI_Iallgather(&stop_me, 1, MPI_C_BOOL, 
            stop_vector, 1, MPI_C_BOOL, MPI_COMM_WORLD, &stop_recv_req);

        goi_std_step(field, my_rows, cols, next);

        MPI_Wait(&top_send_req, MPI_STATUSES_IGNORE);
        MPI_Wait(&top_rec_req, MPI_STATUSES_IGNORE);
        goi_std_step(matrix, 3, cols, next_matrix);
        
        MPI_Wait(&bot_send_req, MPI_STATUSES_IGNORE);
        MPI_Wait(&bot_rec_req, MPI_STATUSES_IGNORE);
        goi_std_step(matrix + (my_rows - 1) * cols, 3, cols, next_matrix + (my_rows - 1) * cols);

        MPI_Wait(&stop_recv_req, MPI_STATUSES_IGNORE);

        // printf("rank: %d; stop: %d%d\n", rank, stop_vector[0], stop_vector[1]);

        stop = true;
        for(int i = 0; i < size; ++i) {
            stop &= stop_vector[i];
        }

        if (stop) break;

        SWAP(field, next, bool*);
        SWAP(matrix, next_matrix, bool*);

        ++iteration;

        if (iteration >= ITERATIONS_COUNT) stop |= true;
    }

    if (rank == 0) {
        fprintf(stdout, "Ended after %i interations\n", iteration);
    }

    free(matrix);
    free(next_matrix);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Must specify exactly 3 arguments: width, height, seed\n");
        return 0;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = strtol(argv[1], NULL, 10);
    int cols = strtol(argv[2], NULL, 10);
    size_t seed = strtoull(argv[3], NULL, 10);

    rows = max(3, rows);
    cols = max(3, cols);

    bool *field = NULL;

    if (rank == 0) {
        field = (bool*)malloc(sizeof(bool) * rows * cols);
        memset(field, 0, rows*cols*sizeof(bool));
    }

    goi_start(field, rows, cols, (field_seed_t)seed);

    if (rank == 0) free(field);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
