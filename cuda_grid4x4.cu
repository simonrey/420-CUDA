
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define N 4

#define TAG 0
#define RHO 0.5 // related to pitch
#define ETA 2e-4 // related to duration of sound
#define BOUNDARY_GAIN 0.75 // clamped edge vs free edge


__global__ void process(float * u, float * u1, float * u2, int T){

  //center
  float sum_of_neighbors, previous_value, previous_previous_value;

  int i = threadIdx.x/4;
  int j = threadIdx.x%4;

  int tid = (i * 4) + j; // Indexes

  //[((i-1)*4)+j]
  //[(i*4)+(j-1)]
  int t;
  for (t = 0; t < T; t++) {

    if(tid == 5 || tid == 6 || tid == 9 || tid == 10){
        sum_of_neighbors = u1[((i-1)*4)+j] + u1[((i+1)*4)+j] + u1[(i*4)+(j-1)] + u1[(i*4)+(j+1)];
        previous_value = u1[tid];
        previous_previous_value = u2[tid];
        u[tid] = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
    }
    __syncthreads();
    // update side points
    // 0 * 4) + j
    if(tid == 1 || tid == 2 || tid == 4 || tid == 8 || tid == 7 || tid == 11 || tid == 13 || tid == 14){
      u[i] = BOUNDARY_GAIN * u[4+i]; // top
      u[((N-1) * 4) + i] = BOUNDARY_GAIN * u[(N-2)*4 + i]; // bottom
      u[i*4] = BOUNDARY_GAIN * u[i*4 + 1]; // left
      u[i*4 + N-1] = BOUNDARY_GAIN * u[i*4 + N-2]; // right
    }

    __syncthreads();
    // update corners
    if(tid == 0 || tid == 3 || tid == 12 || tid == 15){
      u[0] = BOUNDARY_GAIN * u[4];
      u[(N-1)*4] = BOUNDARY_GAIN * u[(N-2)*4];
      u[N-1] = BOUNDARY_GAIN * u[N-2];
      u[(N-1)*4 + N-1] = BOUNDARY_GAIN * u[4*(N-1)+N-2];
    }
    if(tid == 10)
      printf("%f,\n", u[10]);

    float * temp;
    temp = u2;
    u2 = u1;
    u1 = u;
    u = temp;
  }

}


int main(int argc, char *argv[]){

    //Get number of iterations
  int T = atoi(argv[1]);

  const int ARRAY_SIZE = 16;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float u[ARRAY_SIZE];
  float u1[ARRAY_SIZE];
  float u2[ARRAY_SIZE];

	for (int i = 0; i < ARRAY_SIZE; i++) {
		u[i] = 0.0;
    u1[i] = 0.0;
    u2[i] = 0.0;

	}
  u1[10] = 1.0;
	float h_out[ARRAY_SIZE];

	// declare GPU memory pointers
	float * d_u;
	float * d_u1;
  float * d_u2;

	// allocate GPU memory
	cudaMalloc(&d_u, ARRAY_BYTES);
  cudaMalloc(&d_u1, ARRAY_BYTES);
	cudaMalloc(&d_u2, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_u, u, ARRAY_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u1, u1, ARRAY_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u2, u2, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
  int threadsPerBlock = 16;
  int numBlocks = 1;
  clock_t tic = clock();
  process<<<numBlocks,threadsPerBlock>>>(d_u, d_u1, d_u2, T);
  cudaDeviceSynchronize();

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_u, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  cudaFree(d_u);
	cudaFree(d_u1);
	cudaFree(d_u2);
  clock_t toc = clock();
  printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

    return 0;
}
