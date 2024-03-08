#include <stdio.h>
#include <mpi.h>

#define c1 1.23456
#define c2 6.54321

#define n 4
double f[n];
double x[n];

int pid;
int np;

double sgn(double x) {
    if (x < 0.0) {
        return -1.0;
    } else {
        return 1.0;
    }
}

void calcForce(int i) {
    int j;
    double diff, tmp;

    // Calculate Force Particle value from 0 to i
    for (j = 0; j < i; j++) {
        diff = x[i] - x[j];
        tmp = c1 / (diff * diff * diff) - c2 * sgn(x[j]) / (diff * diff);

        // Accumulate the value in the local array f
        f[i] += tmp;
        f[j] -= tmp;
    }
}

int main(int argc, char **argv) {
    x[0] = 10;
    x[1] = 20;
    x[2] = 30;
    x[3] = 40;
    f[0] = 0;
    f[1] = 0;
    f[2] = 0;
    f[3] = 0;

    // Initialize MPI init to get the number of processors
    MPI_Init(&argc, &argv);
    // Get rank for each processor for MPI_COMM_WORLD
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get the number of processors available in MPI_COMM_WORLD
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // Assign F array to 0.0 if PID is 0
    if (pid == 0) {
        for (int i = 0; i < n; i++) {
            f[i] = 0.0;
        }
    }

    // Allocate each element of input particle to each processor and calculate value
    for (int i = pid; i < n; i = i + np) {
        calcForce(i);
    }

    // Gather the results from all processes to process 0
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, f, n, MPI_DOUBLE, MPI_COMM_WORLD);

    // Print F value
    if (pid == 0) {
        for (int i = 0; i < n; i++) {
            printf("%f \n", f[i]);
        }
    }

    // Finish MPI Execution
    MPI_Finalize();

    return 0;
}