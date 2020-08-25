#include "stdio.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>
//Defining number of elements in Array
#define N 10000
//Definin number of iterations of the entire program
#define M 10	

#ifndef EXPORT_API
#define EXPORT_API __attribute__ ((visibility("default")))
#endif 



//Defining Kernel function for vector addition
__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
	//Getting block index of current kernel
	int tid = blockIdx.x;	// handle the data at this index
	if (tid < N)
		d_c[tid] = d_a[tid] + d_b[tid];
}

int main(void) {
	//Defining host arrays
	//int h_a[N], h_b[N], h_c[N];
	//Defining device pointers
	int *d_a, *d_b, *d_c;
	// allocate the memory
	cudaMallocManaged((void**)&d_a, N * sizeof(int));
	cudaMallocManaged((void**)&d_b, N * sizeof(int));
	cudaMallocManaged((void**)&d_c, N * sizeof(int));
	//Initializing Arrays
	for (int i = 0; i < M; i++){
	for (int y = 0; y < N; y++) {
		d_a[y] = 2*y*y;
		d_b[y] = y ;
		
	}
	
	//Calling kernels with N blocks and one thread per block, passing device pointers as parameters
	gpuAdd << <N, 1 >> >(d_a, d_b, d_c);
	//Copy result back to host memory from device memory
	cudaDeviceSynchronize();

	printf("Vector addition on GPU \n");
	//Printing result on console
	for (int y = 0; y < N; y++) {
		printf("The sum of %d element is %d + %d = %d\n", y, d_a[y], d_b[y], d_c[y]);
	}
	}
	
	//Free up memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
