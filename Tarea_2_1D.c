#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N_GLOBAL  400
#define STEPS     1000

#define IDX(i,j) ((i) * (local_ny + 2) + (j))

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N_GLOBAL % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: N_GLOBAL debe ser divisible por size.\n");
            fprintf(stderr, "N_GLOBAL = %d, size = %d\n", N_GLOBAL, size);
        }
        MPI_Finalize();
        return 1;
    }

    int local_nx = N_GLOBAL / size;
    int local_ny = N_GLOBAL;

    int up   = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int down = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    int local_size = (local_nx + 2) * (local_ny + 2);

    double* data     = calloc(local_size, sizeof(double));
    double* data_new = calloc(local_size, sizeof(double));
    double* rho      = calloc(local_size, sizeof(double));

    if (data == NULL || data_new == NULL || rho == NULL) {
        fprintf(stderr, "Proceso %d: error al reservar memoria.\n", rank);
        MPI_Finalize();
        return 1;
    }

    double x0 = 0.5 * (double)N_GLOBAL;
    double y0 = 0.5 * (double)N_GLOBAL;
    double sigma = 0.1 * (double)N_GLOBAL;

    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            int gi = rank * local_nx + (i - 1);
            int gj = j - 1;

            double dx = (double)gi - x0;
            double dy = (double)gj - y0;

            rho[IDX(i,j)] = exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
        }
    }

    MPI_Request req[4];

    MPI_Barrier(MPI_COMM_WORLD);
    double t_inicio = MPI_Wtime();

    for (int step = 0; step < STEPS; step++) {

        MPI_Isend(&data[IDX(1, 1)], local_ny, MPI_DOUBLE,
                  up, 0, MPI_COMM_WORLD, &req[0]);

        MPI_Isend(&data[IDX(local_nx, 1)], local_ny, MPI_DOUBLE,
                  down, 1, MPI_COMM_WORLD, &req[1]);

        MPI_Irecv(&data[IDX(0, 1)], local_ny, MPI_DOUBLE,
                  up, 1, MPI_COMM_WORLD, &req[2]);

        MPI_Irecv(&data[IDX(local_nx + 1, 1)], local_ny, MPI_DOUBLE,
                  down, 0, MPI_COMM_WORLD, &req[3]);

        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

        for (int i = 1; i <= local_nx; i++) {
            for (int j = 1; j <= local_ny; j++) {
                data_new[IDX(i,j)] = 0.25 * (
                    data[IDX(i+1,j)] +
                    data[IDX(i-1,j)] +
                    data[IDX(i,j+1)] +
                    data[IDX(i,j-1)] -
                    rho[IDX(i,j)]
                );
            }
        }

        if (rank == 0) {
            for (int j = 1; j <= local_ny; j++)
                data_new[IDX(1,j)] = 0.0;
        }

        if (rank == size - 1) {
            for (int j = 1; j <= local_ny; j++)
                data_new[IDX(local_nx,j)] = 0.0;
        }

        for (int i = 1; i <= local_nx; i++) {
            data_new[IDX(i,1)] = 0.0;
            data_new[IDX(i,local_ny)] = 0.0;
        }

        double* tmp = data;
        data = data_new;
        data_new = tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_fin = MPI_Wtime();

    double suma_local = 0.0;
    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            suma_local += fabs(data[IDX(i,j)]);
        }
    }

    double suma_global = 0.0;
    MPI_Reduce(&suma_local, &suma_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Poisson 2D con Jacobi y MPI\n");
        printf("N_GLOBAL:       %d x %d\n", N_GLOBAL, N_GLOBAL);
        printf("STEPS:          %d\n", STEPS);
        printf("Procesos:       %d\n", size);
        printf("Particion MPI:  1D por filas\n");
        printf("Bloque local:   %d x %d\n", local_nx, local_ny);
        printf("Tiempo total:   %.6f s\n", t_fin - t_inicio);
        printf("Suma |data|:    %.6e\n", suma_global);
    }

    free(data);
    free(data_new);
    free(rho);

    MPI_Finalize();
    return 0;
}
