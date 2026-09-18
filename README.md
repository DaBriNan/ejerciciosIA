# \# Problema de Juguete

# 

#! \[Fotografía](images/problemasJuguete.jpg)







\# Problema A\*



!\[Problema A\*](images/problemaAasterisco.jpg)







\# Evaluación



!\[Evaluación](images/EvalAasterisco.jpg)


## Dino EDA1

![Dino EDA 1](images/dinoEda1.jpg)



## Dino EDA2

![Dino EDA 1](images/dinoEda1.jpg)



## Red Neuronal en C (Problema XOR)

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EPOCAS 50000
#define LR 0.1f // Tasa de aprendizaje (Learning Rate)

// Arquitectura de la red: 2 entradas -> 2 ocultas -> 1 salida
float w_oculta[2][2]; // [neurona_oculta][entrada]
float b_oculta[2];    // Bias de cada neurona oculta

float w_salida[2];    // Pesos desde capa oculta a neurona de salida
float b_salida;       // Bias de neurona de salida

// Función de activación sigmoide real
float sigmoide(float s) {
    return 1.0f / (1.0f + expf(-s));
}

// Derivada de la sigmoide: f'(x) = f(x) * (1 - f(x))
float derivada_sigmoide(float out) {
    return out * (1.0f - out);
}

// Inicialización aleatoria uniforme entre -1.0 y 1.0
void inicializar_pesos(void) {
    for (int i = 0; i < 2; i++) {
        b_oculta[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        for (int j = 0; j < 2; j++) {
            w_oculta[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
    b_salida = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < 2; i++) {
        w_salida[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

// Paso hacia adelante (Feedforward) y ajuste (Backpropagation)
float entrenar_patron(float x0, float x1, float target) {
    float entradas[2] = {x0, x1};
    float h_act[2]; // Activación de la capa oculta
    float out_act;  // Activación de salida

    // 1. FORWARD: Capa de Entrada -> Capa Oculta
    for (int i = 0; i < 2; i++) {
        float net = w_oculta[i][0] * entradas[0] + w_oculta[i][1] * entradas[1] + b_oculta[i];
        h_act[i] = sigmoide(net);
    }

    // 2. FORWARD: Capa Oculta -> Capa de Salida
    float net_out = w_salida[0] * h_act[0] + w_salida[1] * h_act[1] + b_salida;
    out_act = sigmoide(net_out);

    // 3. BACKPROPAGATION: Gradiente del error en Salida
    float error = target - out_act;
    float delta_salida = error * derivada_sigmoide(out_act);

    // 4. BACKPROPAGATION: Retropropagación a la Capa Oculta
    float delta_oculta[2];
    for (int i = 0; i < 2; i++) {
        delta_oculta[i] = derivada_sigmoide(h_act[i]) * (delta_salida * w_salida[i]);
    }

    // 5. ACTUALIZACIÓN DE PESOS: Capa de Salida
    for (int i = 0; i < 2; i++) {
        w_salida[i] += LR * delta_salida * h_act[i];
    }
    b_salida += LR * delta_salida;

    // 6. ACTUALIZACIÓN DE PESOS: Capa Oculta
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            w_oculta[i][j] += LR * delta_oculta[i] * entradas[j];
        }
        b_oculta[i] += LR * delta_oculta[i];
    }

    return out_act;
}

// Inferencia pura (sin entrenamiento)
float inferencia(float x0, float x1) {
    float h0 = sigmoide(w_oculta[0][0] * x0 + w_oculta[0][1] * x1 + b_oculta[0]);
    float h1 = sigmoide(w_oculta[1][0] * x0 + w_oculta[1][1] * x1 + b_oculta[1]);
    return sigmoide(w_salida[0] * h0 + w_salida[1] * h1 + b_salida);
}

int main(void) {
    srand((unsigned int)time(NULL));
    inicializar_pesos();

    // Tabla de verdad XOR: {x0, x1, target}
    float dataset[4][3] = {
        {1.0f, 1.0f, 0.0f},
        {1.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 1.0f},
        {0.0f, 0.0f, 0.0f}
    };

    printf("Entrenando red neuronal...\n");
    for (int ep = 0; ep < EPOCAS; ep++) {
        for (int p = 0; p < 4; p++) {
            entrenar_patron(dataset[p][0], dataset[p][1], dataset[p][2]);
        }
    }

    printf("\nResultados finales despues del entrenamiento:\n");
    for (int p = 0; p < 4; p++) {
        float x0 = dataset[p][0];
        float x1 = dataset[p][1];
        float real = inferencia(x0, x1);
        printf("Entrada: (%.0f, %.0f) -> Esperado: %.0f | Salida Red: %f\n", 
               x0, x1, dataset[p][2], real);
    }

    return 0;
}
```
