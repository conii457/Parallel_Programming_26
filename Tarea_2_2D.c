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

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0}; 
    int reorder = 0;

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int up, down, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

    if (N_GLOBAL % dims[0] != 0 || N_GLOBAL % dims[1] != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: N_GLOBAL debe ser divisible por dims[0] y dims[1].\n");
            fprintf(stderr, "N_GLOBAL = %d, dims = %d x %d\n", N_GLOBAL, dims[0], dims[1]);
        }
        MPI_Finalize();
        return 1;
    }

    int local_nx = N_GLOBAL / dims[0];
    int local_ny = N_GLOBAL / dims[1];

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
            int gi = coords[0] * local_nx + (i - 1);
            int gj = coords[1] * local_ny + (j - 1);

            double dx = (double)gi - x0;
            double dy = (double)gj - y0;

            rho[IDX(i,j)] = exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
        }
    }

    MPI_Datatype col_type;
    MPI_Type_vector(local_nx, 1, local_ny + 2, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    MPI_Request req[8];

    MPI_Barrier(cart_comm);
    double t_inicio = MPI_Wtime();

    for (int step = 0; step < STEPS; step++) {

        MPI_Isend(&data[IDX(1, 1)], local_ny, MPI_DOUBLE,
                  up, 0, cart_comm, &req[0]);

        MPI_Isend(&data[IDX(local_nx, 1)], local_ny, MPI_DOUBLE,
                  down, 1, cart_comm, &req[1]);

        MPI_Irecv(&data[IDX(0, 1)], local_ny, MPI_DOUBLE,
                  up, 1, cart_comm, &req[2]);

        MPI_Irecv(&data[IDX(local_nx + 1, 1)], local_ny, MPI_DOUBLE,
                  down, 0, cart_comm, &req[3]);

        MPI_Isend(&data[IDX(1, 1)], 1, col_type,
                  left, 2, cart_comm, &req[4]);

        MPI_Isend(&data[IDX(1, local_ny)], 1, col_type,
                  right, 3, cart_comm, &req[5]);

        MPI_Irecv(&data[IDX(1, 0)], 1, col_type,
                  left, 3, cart_comm, &req[6]);

        MPI_Irecv(&data[IDX(1, local_ny + 1)], 1, col_type,
                  right, 2, cart_comm, &req[7]);

        MPI_Waitall(8, req, MPI_STATUSES_IGNORE);

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

        if (coords[0] == 0) {
            for (int j = 1; j <= local_ny; j++)
                data_new[IDX(1,j)] = 0.0;
        }

        if (coords[0] == dims[0] - 1) {
            for (int j = 1; j <= local_ny; j++)
                data_new[IDX(local_nx,j)] = 0.0;
        }

        if (coords[1] == 0) {
            for (int i = 1; i <= local_nx; i++)
                data_new[IDX(i,1)] = 0.0;
        }

        if (coords[1] == dims[1] - 1) {
            for (int i = 1; i <= local_nx; i++)
                data_new[IDX(i,local_ny)] = 0.0;
        }

        double* tmp = data;
        data = data_new;
        data_new = tmp;
    }

    MPI_Barrier(cart_comm);
    double t_fin = MPI_Wtime();

    double suma_local = 0.0;
    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            suma_local += fabs(data[IDX(i,j)]);
        }
    }

    double suma_global = 0.0;
    MPI_Reduce(&suma_local, &suma_global, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        printf("Poisson 2D con Jacobi y MPI\n");
        printf("N_GLOBAL:       %d x %d\n", N_GLOBAL, N_GLOBAL);
        printf("STEPS:          %d\n", STEPS);
        printf("Procesos:       %d\n", size);
        printf("Grilla MPI:     %d x %d\n", dims[0], dims[1]);
        printf("Bloque local:   %d x %d\n", local_nx, local_ny);
        printf("Tiempo total:   %.6f s\n", t_fin - t_inicio);
        printf("Suma |data|:    %.6e\n", suma_global);
    }

    MPI_Type_free(&col_type);
    free(data);
    free(data_new);
    free(rho);

    MPI_Finalize();
    return 0;
}
