#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include <mpi.h>
#include <pthread.h>

enum COMMUNICATION_TAG {
    REQUEST_TAG,
    ANSWER_TAG
};

enum WORK_STATUS {
    TASK_SENT,
    ASK_FOR_TASK,
    NO_TASKS,
    WORK_DONE,
};

#define MAIN_PROC_RANK 0
#define ITERATIONS_COUNT 5
#define TASKS_PER_PROC 250
#define L_CONST 1000
#define MUTEX_TOTAL 3

pthread_mutex_t tasks_list_mutex;
pthread_mutex_t my_task_id_mutex;
pthread_mutex_t tasks_given_mutex;

int rank, size;

int* tasks_list;
int tasks_given, my_task_id;

int count_tasks_to_give() {
    return size / (rank + 1);
}

void do_work(double *localRes, int taskIdx) {
    for (int i = 0; i < tasks_list[taskIdx]; ++i) {
        *localRes += exp(sin(i));
    }
}

void redistribute_load(int iteraion) {
    pthread_mutex_lock(&tasks_list_mutex);
    for (int i = rank * TASKS_PER_PROC; i < (rank + 1) * TASKS_PER_PROC; ++i) {
        tasks_list[i] = abs(50 - i % TASKS_PER_PROC) * abs(rank - (iteraion % size)) * L_CONST;
    }
    pthread_mutex_unlock(&tasks_list_mutex);
}

bool recieve_new_tasks(int sponsor_rank, int *received_tasks, int *other_task_id) {
    int send_msg = ASK_FOR_TASK, recv_msg;

    MPI_Send(&send_msg, 1, MPI_INT, sponsor_rank, REQUEST_TAG, MPI_COMM_WORLD);
    MPI_Recv(&recv_msg, 1, MPI_INT, sponsor_rank, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (recv_msg == TASK_SENT) {
        MPI_Recv(received_tasks, 1, MPI_INT,
            sponsor_rank, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        pthread_mutex_lock(&tasks_list_mutex);
        MPI_Recv(&tasks_list[sponsor_rank * TASKS_PER_PROC], *received_tasks, MPI_INT,
            sponsor_rank, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pthread_mutex_unlock(&tasks_list_mutex);

        *other_task_id = sponsor_rank * TASKS_PER_PROC;
    
        return true;
    }

    return false;
}

void* worker_routine(void *args) {
    int local_task_count, total_task_count;
    
    // init worker stats
    double start_time, local_time, max_time, min_time;
    double average_disbalance = 0.0;
    double local_work_res = 0.0, global_work_res;

    for (int iter = 0; iter < ITERATIONS_COUNT; ++iter) {
        if (rank == MAIN_PROC_RANK) {
            printf("===========ITER %i===========\n", iter);
        }

        redistribute_load(iter);
        local_task_count = 0;

        pthread_mutex_lock(&my_task_id_mutex);
        my_task_id = TASKS_PER_PROC * rank;
        pthread_mutex_unlock(&my_task_id_mutex);

        pthread_mutex_lock(&tasks_given_mutex);
        tasks_given = 0;
        pthread_mutex_unlock(&tasks_given_mutex);

        start_time = MPI_Wtime();

        while (my_task_id < TASKS_PER_PROC * (rank + 1) - tasks_given) {
            do_work(&local_work_res, my_task_id);

            pthread_mutex_lock(&my_task_id_mutex);
            my_task_id++;
            pthread_mutex_unlock(&my_task_id_mutex);

            local_task_count++;
        }

        bool has_new_tasks;
        do {
            has_new_tasks = false;
            
            for (int cur_sponsor = 0; cur_sponsor < size; ++cur_sponsor) {
                int received_tasks, other_task_id;
                
                bool has_recieved = recieve_new_tasks(cur_sponsor, &received_tasks, &other_task_id);

                if (cur_sponsor != rank && has_recieved) {
                    for (int i = 0; i < received_tasks; ++i) {
                        do_work(&local_work_res, other_task_id);
                        
                        ++other_task_id;
                        ++local_task_count;
                    }

                    has_new_tasks = true;

                    break;
                }
            }
        } while (has_new_tasks);

        local_time = MPI_Wtime() - start_time;
        
        printf("Elapsed time on proc %d: %.2f\n", rank, local_time);
        MPI_Barrier(MPI_COMM_WORLD);
        printf("Tasks done on proc %d: %d\n", rank, local_task_count);
        
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MAIN_PROC_RANK, MPI_COMM_WORLD);
        MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, MAIN_PROC_RANK, MPI_COMM_WORLD);
        MPI_Reduce(&local_task_count, &total_task_count, 1, MPI_INT, MPI_SUM, MAIN_PROC_RANK, MPI_COMM_WORLD);

        if (rank == MAIN_PROC_RANK) {
            printf("Tasks total: %d\n", total_task_count);
            printf("Disbalance time: %.2f\n", max_time - min_time);
            printf("Disbalance proportion: %.2f%%\n", (max_time - min_time) / max_time * 100);
            
            average_disbalance += (max_time - min_time) / max_time * 100;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    int done_msg = WORK_DONE;
    MPI_Send(&done_msg, 1, MPI_INT, rank, REQUEST_TAG, MPI_COMM_WORLD);

    MPI_Reduce(&local_work_res, &global_work_res, 1, MPI_DOUBLE, MPI_SUM, MAIN_PROC_RANK, MPI_COMM_WORLD);

    if (rank == MAIN_PROC_RANK) {
        printf("==========RESULTS===========\n");
        printf("Global result: %.2f\n", global_work_res);
        printf("Average Disbalance: %.2f%%\n", average_disbalance / ITERATIONS_COUNT);
    }

    return NULL;
}

void* sender_routine(void *args) {

    MPI_Status recv_status;
    int send_msg, recv_msg;

    while (true) {
        MPI_Recv(&recv_msg, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &recv_status);
        
        int sender = recv_status.MPI_SOURCE;
        
        bool other_proc_need_tasks = recv_msg == ASK_FOR_TASK && sender != rank;

        pthread_mutex_lock(&my_task_id_mutex);
        bool dont_have_tasks = my_task_id > (rank + 1) * TASKS_PER_PROC - tasks_given;
        pthread_mutex_unlock(&my_task_id_mutex);

        if (other_proc_need_tasks && dont_have_tasks) {
            send_msg = NO_TASKS;
            MPI_Send(&send_msg, 1, MPI_INT, sender, ANSWER_TAG, MPI_COMM_WORLD);
        } else if (other_proc_need_tasks) {
            int sending_tasks_count = count_tasks_to_give();

            pthread_mutex_lock(&tasks_given_mutex);
            tasks_given += sending_tasks_count;
            pthread_mutex_unlock(&tasks_given_mutex);

            pthread_mutex_lock(&tasks_list_mutex);
            int *sendingTasks = &tasks_list[(rank + 1) * TASKS_PER_PROC - tasks_given];
            pthread_mutex_unlock(&tasks_list_mutex);

            send_msg = TASK_SENT;
            MPI_Send(&send_msg, 1, MPI_INT, 
                sender, ANSWER_TAG, MPI_COMM_WORLD);
            MPI_Send(&sending_tasks_count, 1, MPI_INT, 
                sender, ANSWER_TAG, MPI_COMM_WORLD); 
            MPI_Send(sendingTasks, sending_tasks_count, MPI_INT, 
                sender, ANSWER_TAG, MPI_COMM_WORLD); 
            
        } else if (recv_msg == WORK_DONE) {
            break;
        }
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    int providedLevel;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &providedLevel);

    if (providedLevel != MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Error on pthread init\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    pthread_mutex_t all_mutexes[] = {tasks_list_mutex, my_task_id_mutex, tasks_given_mutex};

    for (int i = 0; i < MUTEX_TOTAL; ++i) {
        if (pthread_mutex_init(&all_mutexes[i], NULL)) {
            fprintf(stderr, "Error on initializing mutex\n");
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    pthread_t sender;
    pthread_t worker;

    tasks_list = (int*)malloc(TASKS_PER_PROC * size * sizeof(int));

    pthread_create(&sender, NULL, sender_routine, NULL);
    pthread_create(&worker, NULL, worker_routine, NULL);

    pthread_join(sender, NULL);
    pthread_join(worker, NULL);

    for (int i = 0; i < MUTEX_TOTAL; ++i) {
        pthread_mutex_destroy(&all_mutexes[i]);
    }
    free(tasks_list);

    MPI_Finalize();
    return EXIT_SUCCESS;
}