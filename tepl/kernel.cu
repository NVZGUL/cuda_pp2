
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <thrust/device_vector.h> 
#include <thrust/transform.h> 
#include <thrust/sequence.h> 
#include <thrust/copy.h> 
#include <thrust/fill.h> 
#include <thrust/replace.h> 
#include <thrust/functional.h>
#include <cuda.h>
#include <random>
#include <cmath>


#define THREADS 1024

#define PI      3.141592653589793f
#define PI_M2   6.283185307179586f
//NBODY
const int Ni1 = 1000; //	number of points
__device__ __constant__ const int Ni = 1000; //	number of points
__device__ __constant__ const float G = 6.673e-11;	// gravity force
__device__ __constant__ const float TIMESTAMP = 1e11;

//TEPL
const double t1 = 5000; // 50000
const int N = 50000;
const double L = 0.1;		//	длина стержня
const double t_final1 = 600;
__device__ __constant__ int N1 = 50000; // 10000			//	колличество элементов на которые разделен стержень 
__device__ __constant__ double t_final = 600;	//	время наблюдаемого процесса
__device__ __constant__ double lambda = 46;	//	параметра лямбда в уравнении теплопроводности
__device__ __constant__ double ro = 7800;		//	плотность в уранении теплопроводности
__device__ __constant__ double c = 460;		//	теплоемкость материала в уравнении теплопроводности
__device__ __constant__ double T_0 = 20;		//	Начальная температура
__device__ __constant__ double T_l = 300;		//	Температура на грацие х=0
__device__ __constant__ double T_r = 100;		//	Температура на границе х=L

__device__
struct Body
{
	float3 position;
	float3 velocity;
	float3 force;

	float m;
};

__device__ Body update(Body p, float timestamp)
{
	p.velocity.x += timestamp*p.force.x / p.m;
	p.velocity.y += timestamp*p.force.y / p.m;
	p.velocity.z += timestamp*p.force.z / p.m;
	p.position.x += timestamp*p.velocity.x;
	p.position.y += timestamp*p.velocity.y;
	p.position.z += timestamp*p.velocity.z;
	return p;
}

__device__ Body resetForce(Body p)
{
	p.force.x = 0.0;
	p.force.y = 0.0;
	p.force.z = 0.0;
	return p;
}
//Add force to particle a by particle b
__device__ Body addForce(Body a, Body b)
{
	float EPS = 3E4;      // softening parameter (just to avoid infinities)
	float dx = b.position.x - a.position.x;
	float dy = b.position.y - a.position.y;
	float dz = b.position.z - a.position.z;
	float dist = sqrt(dx*dx + dy*dy + dz*dz);
	float F = (G * a.m * b.m) / (dist*dist + EPS*EPS);
	a.force.x += F * dx / dist;
	a.force.y += F * dy / dist;
	a.force.z += F * dz / dist;
	return a;

}

std::random_device rd;
std::mt19937 gen(rd());

float rand(float r)
{
	std::uniform_real_distribution<float> dis(0, r);
	return dis(gen);
}
float rand(float l, float r)
{
	std::uniform_real_distribution<float> dis(l, r);
	return dis(gen);
}

float3 getPoint(float r)
{
	const float
		phi = rand(PI_M2),
		sintheta = rand(-1.f, 1.f),
		costheta = std::sqrt(1.f - sintheta*sintheta);

	float3 point{
		r * std::cos(phi) * sintheta,
		r * std::sin(phi) * sintheta,
		r * costheta
	};

	return point;
}


__global__ void Nbody(Body* particles, int numberofiterations)
{
	int k = threadIdx.x;
	if (k < numberofiterations) {
		for (int i = 0; i < Ni; i++)
		{
			particles[i] = resetForce(particles[i]);
			for (int j = 0; j < Ni; j++)
			{
				if (i != j)
				{
					particles[i] = addForce(particles[i], particles[j]);
				}

			}
		}
		//loop again to update the time stamp here
		for (int i = 0; i < Ni; i++)
		{
			particles[i] = update(particles[i], TIMESTAMP);
		}
	}
}

__global__ void yavn(double time, double h, double* T_arr)
{
	int i = threadIdx.x + time;
	double* T_prev = T_arr;
	double a = lambda / (ro*c);
	double tau = 0.25 * pow(h, 2) / a;
	T_arr[0] = T_l;
	T_arr[N1 - 1] = T_r;
	if (i < t_final)
	{
		T_prev = T_arr;
		for (int i = 1; i < N1 - 1; i++)
		{
			T_arr[i] = T_prev[i] + a*tau / pow(h, 2)*(T_prev[i + 1] - 2 * T_prev[i] + T_prev[i - 1]);
		}
	}
}

__global__ void neyavn(double time, double tau, double h, double* alfa, double*  beta, double* T_arr)
{
	double a_i, b_i, c_i, f_i;
	int i = threadIdx.x + time;
	if (i < t_final)
	{
		time += tau;
		alfa[0] = 0;
		beta[0] = T_l;
		for (int i = 1; i < N1 - 1; i++)
		{
			a_i = lambda / pow(h, 2);
			b_i = 2 * lambda / pow(h, 2) + ro * c / tau;
			c_i = lambda / pow(h, 2);
			f_i = -ro*c*T_arr[i] / tau;
			alfa[i] = a_i / (b_i - c_i*alfa[i - 1]);
			beta[i] = (c_i*beta[i - 1] - f_i) / (b_i - c_i*alfa[i - 1]);
		}
		T_arr[N - 1] = T_r;
		for (int i = N1 - 2; i >= 0; i--)
		{
			T_arr[i] = alfa[i] * T_arr[i + 1] + beta[i];
		}
	}
}


int main()
{
	auto tau_step = [](double t) { return t / t1; };
	auto h_step = [](double L, int N) { return L / (N - 1); };
	double h = h_step(L, N);
	double time = 0;
	double tau = tau_step(t_final1);
	float elapsedTime;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	thrust::device_vector<double> T_arr(N);
	thrust::device_vector<double> alfa(N);
	thrust::device_vector<double> beta(N);
	// fill vector with 0 
	thrust::fill(T_arr.begin(), T_arr.end(), 0);
	thrust::fill(alfa.begin(), alfa.end(), 0);
	thrust::fill(beta.begin(), beta.end(), 0);
	// print vector 
	//thrust::copy(T_arr.begin(), T_arr.end(), std::ostream_iterator<double>(std::cout, "\n"));
	double * T = thrust::raw_pointer_cast(T_arr.data());
	double * a = thrust::raw_pointer_cast(alfa.data());
	double * b = thrust::raw_pointer_cast(beta.data());
	neyavn << <1, THREADS >> > (time, tau, h, a, b, T);
	//yavn << <1, THREADS >> > (time, h, T);
	thrust::copy(T_arr.begin(), T_arr.end(), std::ostream_iterator<double>(std::cout, "\n"));
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms\n", elapsedTime);
	/*
	thrust::device_vector<Body> particles(Ni1);
	//randomly generating N Particles

	for (int i = 0; i < Ni1; i++) {
		float rx = float(1e18*exp(-1.8)*(.5 - rand()));
		float3 point = getPoint(rx);
		float vx = float(1e18*exp(-1.8)*(.5 - rand()));
		float vy = float(1e18*exp(-1.8)*(.5 - rand()));
		float vz = float(1e18*exp(-1.8)*(.5 - rand()));
		float mass = float(1.98892e30*rand() * 10 + 1e20);
		particles[i] = { point, { vx,vy,vz }, {0,0,0}, mass};
	}
	Body * p = thrust::raw_pointer_cast(particles.data());
	int numberofiterations = 800;
	//Nbody_simple(particles, numberofiterations);
	//Nbody_thread(particles, numberofiterations);
	Nbody<<<1,THREADS>>>(p, numberofiterations);
	*/
    return 0;
}

