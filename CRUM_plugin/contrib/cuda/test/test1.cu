#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void add(int a, int b, int *c)
{
	*c = a+b;
}

int main(int argc, char **argv)
{
	// test
	int a = 2, b = 3, c;
	int *cuda_c = NULL;

	cudaMalloc(&cuda_c, sizeof(int));
	add<<<1,1>>>(a, b, cuda_c);
        sleep(10);
	cudaMemcpy(&c, cuda_c, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(cuda_c);

	printf("%d + %d = %d\n", a, b, c);
        sleep(10);
	exit(EXIT_SUCCESS);
}
