#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// parámetros del sistema
#define L 32        // tamaño de la red 
#define N (L * L)   // n° total de spins
#define J 1.0
#define B 0.0

// parámetros de simulación
#define N_EQ      2000      // sweeps de equilibración
#define N_MEAS    5000      // n° de mediciones
#define SKIP      10        // sweeps entre mediciones

int S[L][L];    // matriz de spins 32x32

// inicialización aleatoria
void inicializar_aleatorio() {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            S[i][j] = (rand() % 2 == 0) ? 1 : -1;
        }
    }
}

// MAGNETIZACIÓN MEDIA POR SITIO
double magnetizacion_media() {
    double m = 0.0;

    #pragma omp parallel for reduction(+:m)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            m += S[i][j];
        }
    }

    return m / (double)N;
}

// ΔE al invertir el spin --> cuánto cambia la energía si el spin (i,j) cambia de signo
int delta_energia(int i, int j) {
    int arriba    = S[(i - 1 + L) % L][j];
    int abajo     = S[(i + 1) % L][j];
    int izquierda = S[i][(j - 1 + L) % L];
    int derecha   = S[i][(j + 1) % L];

    int suma_vecinos = arriba + abajo + izquierda + derecha;

    return (int)(2.0 * J * S[i][j] * suma_vecinos + 2.0 * B * S[i][j]);
}

// METRÓPOLIS --> selecciona un spin aleatorio e intenta invertirlo
void metropolis_intento(double beta) {
    int i = rand() % L;
    int j = rand() % L;
    int dE = delta_energia(i, j);

    if (dE <= 0) {      // si mejora la energía --> aceptar
        S[i][j] = -S[i][j];
    } else {            // si empeora la energía --> aceptar por probabilidad
        double p = exp(-beta * dE);
        double r = (double)rand() / RAND_MAX;
        if (r < p) {
            S[i][j] = -S[i][j];
        }
    }
}

void sweep(double beta) {       // N = L^2 intentos de metrópolis
    for (int n = 0; n < N; n++) {
        metropolis_intento(beta);
    }
}

void equilibrar(double beta) {
    for (int k = 0; k < N_EQ; k++) {
        sweep(beta);
    }
}

// CORRELACIÓN C(r)
void calcular_correlacion(double beta, double C[]) {

    for (int r = 0; r < L / 2; r++) {   // inicializar acumulador
        C[r] = 0.0;
    }

    for (int muestra = 0; muestra < N_MEAS; muestra++) {    // mediciones

        for (int k = 0; k < SKIP; k++) {    // evolucionar el sistema
            sweep(beta);
        }

        double m = magnetizacion_media();   // magnetización instantánea
        double m2 = m * m;

        double C_temp[L / 2];  // correlación instantánea

        #pragma omp parallel for
        for (int r = 0; r < L / 2; r++) {
            double suma = 0.0;

            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    suma += S[i][j] * S[i][(j + r) % L];    // horizontal
                    suma += S[i][j] * S[(i + r) % L][j];    // vertical
                }
            }

            C_temp[r] = suma / (2.0 * N);   // promedio espacial
            C_temp[r] -= m2;                // correlación conectada
        }

        C_temp[0] = 1.0 - m2;   // valor en r = 0

        for (int r = 0; r < L / 2; r++) {   // acumular
            C[r] += C_temp[r];
        }
    }

    for (int r = 0; r < L / 2; r++) {   // promedio fin
        C[r] /= N_MEAS;

        if (C[r] < 0.0) {
            C[r] = 0.0;
        }
    }
}

// ESTIMAR XI
double estimar_xi(double C[]) {

    double suma = 0.0;
    int contador = 0;

    for (int r = 1; r < L / 2 - 1; r++) {

        if (C[r] > 0.0 && C[r + 1] > 0.0) {

            double razon = C[r + 1] / C[r];

            if (razon > 0.0 && razon < 1.0) {

                double xi_local = -1.0 / log(razon);

                suma += xi_local;
                contador++;
            }
        }
    }

    if (contador == 0) {
        return 0.0;
    }

    return suma / contador;
}

int main() {
    srand(time(NULL));

    FILE *fp_xi = fopen("ising_2_xi.dat", "w");
    if (fp_xi == NULL) {
        printf("Error al crear xi_vs_T.dat\n");
        return 1;
    }

    FILE *fp_corr = fopen("ising_2_corr.dat", "w");
    if (fp_corr == NULL) {
        printf("Error al crear correlaciones.dat\n");
        fclose(fp_xi);
        return 1;
    }

    fprintf(fp_xi, "# T xi\n");
    fprintf(fp_corr, "# T r C(r)\n");

    double C[L / 2];

    for (double T = 1.5; T <= 4.0; T += 0.1) {  // temperaturas

        double beta = 1.0 / T;

        printf("Calculando T = %.3f\n", T);

        inicializar_aleatorio();

        equilibrar(beta);

        calcular_correlacion(beta, C);

        for (int r = 0; r < L / 2; r++) {   // guardar C(r)
            fprintf(fp_corr, "%.6f %d %.8f\n", T, r, C[r]);
        }
        fprintf(fp_corr, "\n");

        double xi = estimar_xi(C);

        fprintf(fp_xi, "%.6f %.8f\n", T, xi);   // guardar xi(T)

        fflush(fp_xi);
        fflush(fp_corr);
    }

    fclose(fp_xi);
    fclose(fp_corr);
    return 0;
}