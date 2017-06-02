
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#define THREADS 1024

const double t1 = 500000; // 50000
const int N = 500; // 10000			//	колличество элементов на которые разделен стержень 
const double t_final = 600;	//	время наблюдаемого процесса
const double L = 0.1;		//	длина стержня
const double lambda = 46;	//	параметра лямбда в уравнении теплопроводности
const double ro = 7800;		//	плотность в уранении теплопроводности
const double c = 460;		//	теплоемкость материала в уравнении теплопроводности
const double T_0 = 20;		//	Начальная температура
const double T_l = 300;		//	Температура на грацие х=0
const double T_r = 100;		//	Температура на границе х=L

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void yavn(double time, double h, double* T_arr, double a, double tau, double T_l, double T_r, double t_final)
{
	int i = threadIdx.x;
	double* T_prev = T_arr;
	T_arr[0] = T_l;
	T_arr[N - 1] = T_r;
	for (double j = time; j < t_final; j += tau)
	{
		T_prev = T_arr;
		//for (int i = 1; i < N - 1; i++)
		if (i < N -1) 
			T_arr[i] = T_prev[i] + a*tau / pow(h, 2)*(T_prev[i + 1] - 2 * T_prev[i] + T_prev[i - 1]);
		
	}
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{

	auto h_step = [](double L, int N) { return L / (N - 1); };
	double h = h_step(L, N);
	double *x = (double *)malloc(40000 * sizeof(double));
	double time = 0;
	double *T_arr;
	double *res;
	double a = lambda / (ro*c);
	double tau = 0.25 * pow(h, 2) / a; /// из условий устойчивости

	cudaMalloc((void **)&T_arr, sizeof(double));
	cudaMemcpy(T_arr, x, sizeof(double), cudaMemcpyHostToDevice);
	yavn << <1, THREADS >> > (time, h, T_arr, a, tau, T_l, T_r, t_final);
	cudaMemcpy(&res, T_arr, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(T_arr);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
