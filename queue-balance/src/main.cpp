#include <mpi.h>

#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <cmath>
#include <cassert>

int rank;
int size;

std::mutex list_mutex;
std::mutex no_tasks_left;
std::mutex end_iteration;

std::vector<int> task_list;

bool all_done = false;

constexpr int ITERATION_COUNT = 10;
constexpr int TASKS_PER_ITERATION = 144;
constexpr int ROOT = 0;
constexpr int MESSAGE_END_ITERATION = -1;
constexpr int MESSAGE_WHAT_TO_DO = -2;

int getRandomTask() {
    return 1000000;
}

void distributeTasks() {
    for (int iteration = 0; iteration < ITERATION_COUNT; iteration++) {
        for (int i = 0; i < TASKS_PER_ITERATION; i++) {
            int message;
            MPI_Status status;
            MPI_Recv(&message,
                1,
                MPI_INT,
                MPI_ANY_SOURCE,
                0,
                MPI_COMM_WORLD,
                &status);

            if (message == MESSAGE_WHAT_TO_DO) {
                int send_message = getRandomTask() * (status.MPI_SOURCE + 1);
                MPI_Send(&send_message,
                    1,
                    MPI_INT,
                    status.MPI_SOURCE,
                    1,
                    MPI_COMM_WORLD);
            } else {
                assert(false);
            }
        }

        int workers_ended_iteration = 0;
        while (workers_ended_iteration < size) {
            int message;
            MPI_Status status;
            MPI_Recv(&message,
                1,
                MPI_INT,
                MPI_ANY_SOURCE,
                0,
                MPI_COMM_WORLD,
                &status);

            if (message == MESSAGE_WHAT_TO_DO) {
                int send_message = MESSAGE_END_ITERATION;
                MPI_Send(&send_message,
                    1,
                    MPI_INT,
                    status.MPI_SOURCE,
                    1,
                    MPI_COMM_WORLD);
            } else  {
                assert(false);
            }
            workers_ended_iteration++;
        }
    }
}

void receiverTask() {
    for (int iteration = 0; iteration < ITERATION_COUNT; iteration++) {
        while (true) {
            no_tasks_left.lock();
            if (!task_list.empty()) {
                no_tasks_left.unlock();
                continue;
            }
            list_mutex.lock();

            int send_message = MESSAGE_WHAT_TO_DO;
            MPI_Send(&send_message, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD);

            int message;
            MPI_Recv(&message, 1, MPI_INT, ROOT, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (message > 0) {
                task_list.push_back(message);
            } else {
	            if (message == MESSAGE_END_ITERATION) {
	            	list_mutex.unlock();
	            	no_tasks_left.unlock();
	            	break;
	            }
            }

            list_mutex.unlock();
            no_tasks_left.unlock();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    all_done = true;
}


void completeTask(const int task) {
    double ans = 0.0;
    for (int i = 0; i < task; i++) {
        ans += sin(i);
    }
}

void workerTask() {
    long long tasks_done = 0;
    while (!all_done) {
        std::lock_guard<std::mutex> lock(no_tasks_left);
        while (!task_list.empty()) {
            list_mutex.lock();

            int task = task_list.back();
            task_list.pop_back();
            
            list_mutex.unlock();
            
            tasks_done += task;
            
            completeTask(task);
        }
    }
    std::cout << rank << " has done " << tasks_done << std::endl;
}

int main(int argc, char** argv) {
    double start;
    int provided = 0;

    MPI_Init_thread(&argc,
        &argv,
        MPI_THREAD_MULTIPLE,
        &provided);
	assert(provided == MPI_THREAD_MULTIPLE && "This mpi version not support multithreading");

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::thread task_distributor_thread;
    if (rank == ROOT) {
        start = MPI_Wtime();
        task_distributor_thread = std::thread(distributeTasks);
    }

    std::thread receiver_thread(receiverTask);
    std::thread worker_thread(workerTask);

    if (rank == ROOT) {
	    task_distributor_thread.join();
    }
    receiver_thread.join();
    worker_thread.join();


    if (rank == ROOT) {
	    double end = MPI_Wtime();
        std::cout << "Time: " << end - start << std::endl;
    }
    MPI_Finalize();

    return 0;
}