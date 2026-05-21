#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// parámetros del sistema
#define L 16        // tamaño de la red 
#define N (L * L)   // n° total de spins 
#define J 1.0
#define B 0.0       // campo magnetico externo 
#define T 2.0       // temperatura 
#define BETA (1.0 / T)

// parámetros de simulación
#define TOTAL_ITER 200000   // n° total de intentos
#define SAVE_EVERY 100      // cada 100 iteraciones guardar un dato 

int S[L][L];    // matriz de spins 16x16

// MAGNETIZACIÓN TOTAL --> suma de todos los spins
int magnetizacion_total() {
    int M = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            M += S[i][j];
        }
    }
    return M;
}

// ENERGÍA TOTAL
double energia_total() {
    double E = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            E += -J * S[i][j] *
                 (S[(i + 1) % L][j] +
                  S[i][(j + 1) % L]);   //evita contar cada interacción dos veces
        }
    }
    return E;
}

// ΔE al invertir el spin --> cuánto cambia la energía si el spin (i,j) cambia de signo
int delta_energia(int i, int j) {
    // vecinos 
    int arriba    = S[(i - 1 + L) % L][j];
    int abajo     = S[(i + 1) % L][j];
    int izquierda = S[i][(j - 1 + L) % L];
    int derecha   = S[i][(j + 1) % L];
    int suma_vecinos = arriba + abajo + izquierda + derecha;
    return (int)(2.0 * J * S[i][j] * suma_vecinos + 2.0 * B * S[i][j]);
}

// configuración aleatoria
void inicializar_aleatorio() {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            S[i][j] = (rand() % 2 == 0) ? -1 : 1;   // cada spin tiene 50% de probabilidad de -1 o +1
        }
    }
}

// METRÓPOLIS --> selecciona un spin aleatorio e intenta invertirlo
void metropolis() {
    int i = rand() % L;
    int j = rand() % L;
    int dE = delta_energia(i, j);   // ΔE

    if (dE <= 0) {      // si mejora la energía --> aceptar
        S[i][j] = -S[i][j];
    } else {            // si empeora la energía --> aceptar por probabilidad 
        double p = exp(-BETA * dE);
        double r = (double)rand() / RAND_MAX;
        if (r < p) {
            S[i][j] = -S[i][j];
        }
    }
}

int main() {
    srand(time(NULL));      // inicializar semilla aleatoria

    FILE *fp = fopen("ising_1.dat", "w"); 
    if (fp == NULL) {
        printf("Error al crear el archivo.\n");
        return 1;
    }
    fprintf(fp, "# iteracion magnetizacion energia\n"); 

    inicializar_aleatorio();

    fprintf(fp, "%d %d %.6f\n",     // guardar estado inicial
            0,
            magnetizacion_total(),
            energia_total());

    for (int iter = 1; iter <= TOTAL_ITER; iter++) {    // simulación principal

        metropolis();       // un solo intento de metrópolis

        // guardar datos cada SAVE_EVERY iteraciones
        if (iter % SAVE_EVERY == 0) {
            fprintf(fp, "%d %d %.6f\n",
                    iter,
                    magnetizacion_total(),
                    energia_total());
        }
    }

    fclose(fp);
    return 0;
}

// No es necesario paralelizar, ya que el cálculo es relativamente rápido incluso sin paralelizar. 