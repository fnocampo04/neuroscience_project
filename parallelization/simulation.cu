#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES // for C
#include <math.h>
#include <time.h>

// Error checking macro
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                             \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,       \
               cudaGetErrorString(e));                               \
        exit(0);                                                     \
    }                                                                \
}

// Custom atomic add for double
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// Function to generate Lorentzian distributed random number
__host__ double random_lorentzian(double x0, double gamma) {
    double u = (double)rand() / RAND_MAX;
    return x0 + gamma * tan(M_PI * (u - 0.5));
}

// Derivative computation
__device__ void derivs(
    double* y, double* dydx, 
    double eta, double I_B, double I_S, 
    double tau_e, double tau_d, double tau_f, double U_0
) {
    dydx[1] = (y[1]*y[1] + eta + I_B + I_S)/tau_e;
    dydx[2] = (1-y[2])/ tau_d;
    dydx[3] = (U_0-y[3])/ tau_f;
}

// RK4 integration method
__device__ void rk4(
    double* y, double* dydx, int n, double x, double h, double* yout,
    double eta, double I_B, double I_S, 
    double tau_e, double tau_d, double tau_f, double U_0
) {
    double dym[4], dyt[4], yt[4];
    double hh = h * 0.5;
    double h6 = h / 6.0;

    // First step
    for (int i = 1; i <= n; i++) {
        yt[i] = y[i] + hh * dydx[i];
    }
    
    // Compute derivatives at midpoint
    derivs(yt, dyt, eta, I_B, I_S, tau_e, tau_d, tau_f, U_0);
    
    // Second midpoint
    for (int i = 1; i <= n; i++) {
        yt[i] = y[i] + hh * dyt[i];
    }
    derivs(yt, dym, eta, I_B, I_S, tau_e, tau_d, tau_f, U_0);
    
    // Third point
    for (int i = 1; i <= n; i++) {
        yt[i] = y[i] + h * dym[i];
        dym[i] += dyt[i];
    }
    derivs(yt, dyt, eta, I_B, I_S, tau_e, tau_d, tau_f, U_0);
    
    // Final update
    for (int i = 1; i <= n; i++) {
        yout[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i]);
    }
}

// Main simulation kernel
__global__ void neuralNetworkKernel(
    double* V, double* etas, double* X, double* U, 
    int N, double t, double h, double Vp, double Vr, 
    double I_B, double J, double tau_e, double U_0,
    double tau_d, double tau_f, double I_S,
    double* spikes, int* spikeCount, double* collective_S
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (i <= N) {
        double f[4], df[4];
        f[1] = V[i];
        f[2] = X[i];
        f[3] = U[i];
        double eta = etas[i];
        
        // Compute derivatives
        derivs(f, df, eta, I_B, I_S, tau_e, tau_d, tau_f, U_0);
        
        // RK4 integration
        rk4(f, df, 3, t, h, f, eta, I_B, I_S, 
            tau_e, tau_d, tau_f, U_0);
        
        // Update neuron state
        V[i] = f[1];
        X[i] = f[2];
        U[i] = f[3];
        
        // Spike condition
        if (V[i] >= Vp) {
            int spikeIndex = atomicAdd(spikeCount, 1);
            spikes[spikeIndex * 2] = t;
            spikes[spikeIndex * 2 + 1] = i;
            
            // Compute collective spike contribution
            atomicAddDouble(collective_S, J * X[i] * U[i] / N);
            
            V[i] = Vr;
            X[i] = X[i] - X[i] * U[i];
            U[i] = U[i] + U_0 * (1 - U[i]);
        }
    }
}

// Kernel to update neurons with collective spike contribution
__global__ void updateNeuronsKernel(
    double* V, int N, double collective_S
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (i <= N) {
        V[i] += collective_S;
    }
}

int main() {
    // Simulation parameters
    const double maxt = 1000.0;
    const double h = 0.0015;
    const int N = 2000;
    const double I_B = -1.0;
    const double J = 15.0;
    const double tau_e = 15.0;
    const double Vp = 100.0;
    const double Vr = -100.0;
    const double U_0 = 0.2;
    const double tau_d = 200.0;
    const double tau_f = 1500.0;
    
    // Memory allocation
    double *h_V, *h_etas, *h_X, *h_U;
    h_V = (double*)malloc((N+1) * sizeof(double));
    h_etas = (double*)malloc((N+1) * sizeof(double));
    h_X = (double*)malloc((N+1) * sizeof(double));
    h_U = (double*)malloc((N+1) * sizeof(double));
    
    // Device memory
    double *d_V, *d_etas, *d_X, *d_U, *d_spikes, *d_collective_S;
    int *d_spikeCount;
    
    cudaMalloc(&d_V, (N+1) * sizeof(double));
    cudaMalloc(&d_etas, (N+1) * sizeof(double));
    cudaMalloc(&d_X, (N+1) * sizeof(double));
    cudaMalloc(&d_U, (N+1) * sizeof(double));
    cudaMalloc(&d_spikes, (N * maxt/h * 2) * sizeof(double));
    cudaMalloc(&d_spikeCount, sizeof(int));
    cudaMalloc(&d_collective_S, sizeof(double));
    
    // Initialize random seed
    srand(time(NULL));
    
    // Initialize conditions
    for (int i = 1; i <= N; i++) {
        h_etas[i] = random_lorentzian(0, 0.25);
        h_V[i] = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
        h_X[i] = 1.0;
        h_U[i] = U_0;
    }
    
    // Copy initial conditions to device
    cudaMemcpy(d_V, h_V, (N+1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_etas, h_etas, (N+1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, (N+1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U, (N+1) * sizeof(double), cudaMemcpyHostToDevice);
    
    // Open output files
    FILE *Spikes = fopen("Spikes_cuda.dat", "w");
    
    // Simulation loop
    double t = 0.0;
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    while (t <= maxt) {
        // Determine input current
        double I_S = ((t >= 150 && t <= 300) || (t >= 450 && t <= 600)) ? 2.0 : 0.0;
        
        // Reset collective spike contribution and spike count
        double h_collective_S = 0.0;
        int h_spikeCount = 0;
        cudaMemcpy(d_spikeCount, &h_spikeCount, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_collective_S, &h_collective_S, sizeof(double), cudaMemcpyHostToDevice);
        
        // Launch neural network kernel
        neuralNetworkKernel<<<gridSize, blockSize>>>(
            d_V, d_etas, d_X, d_U, 
            N, t, h, Vp, Vr, 
            I_B, J, tau_e, U_0,
            tau_d, tau_f, I_S,
            d_spikes, d_spikeCount, d_collective_S
        );
        cudaCheckError();
        
        // Copy collective spike contribution back
        cudaMemcpy(&h_collective_S, d_collective_S, sizeof(double), cudaMemcpyDeviceToHost);
        
        // Update neurons with collective spike contribution
        updateNeuronsKernel<<<gridSize, blockSize>>>(d_V, N, h_collective_S);
        cudaCheckError();
        
        // Copy spike count back
        cudaMemcpy(&h_spikeCount, d_spikeCount, sizeof(int), cudaMemcpyDeviceToHost);
        
        // If spikes occurred, write to file
        if (h_spikeCount > 0) {
            double* h_spikes = (double*)malloc(h_spikeCount * 2 * sizeof(double));
            cudaMemcpy(h_spikes, d_spikes, h_spikeCount * 2 * sizeof(double), cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < h_spikeCount; i++) {
                fprintf(Spikes, "%lf %d\n", h_spikes[i*2], (int)h_spikes[i*2 + 1]);
            }
            
            free(h_spikes);
        }
        
        // Update time
        t += h;
    }
    
    // Clean up
    fclose(Spikes);
    
    // Free host memory
    free(h_V);
    free(h_etas);
    free(h_X);
    free(h_U);
    
    // Free device memory
    cudaFree(d_V);
    cudaFree(d_etas);
    cudaFree(d_X);
    cudaFree(d_U);
    cudaFree(d_spikes);
    cudaFree(d_spikeCount);
    cudaFree(d_collective_S);
    
    return 0;
}