#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include <time.h>


__global__ void process(unsigned char * d_in, unsigned char * d_out, int d_width, int d_height){

    //Get the thread ID accross all the blocks
    //int tid = blockIdx.x*blockDim.x+threadIdx.x;

    int i = (threadIdx.x);
    int j = (blockIdx.x);
    float value;
    float w[3][3] =
    {
      1,2,-1,
      2,0.25,-2,
      1,-2,-1
    };

    if(i >= 0 && i < d_height-1 && j >= 0 && j < d_width-1){
      int k, ii, jj;
      for (k = 0; k < 3; k++) {
        value = 0;
        for (ii = 0; ii < 3; ii++) {
          for (jj = 0; jj < 3; jj++) {
            value += d_in[4*d_width*(i+ii-1) + 4*(j+jj-1) + k] * w[ii][jj];
          }
        }
        value = value > 255 ? 255 : value;
        value = value < 0 ? 0 : value;
        d_out[4*(d_width-2)*(i-1) + 4*(j-1) + k] = value;
      }
      d_out[4*(d_width-2)*(i-1) + 4*(j-1) + 3] = d_in[4*d_width*i + 4*j + 3]; // A
    }
}


int main(int argc, char *argv[]){

    //Gets input and output filenames
    char * input_filename = argv[1];
    char * output_filename = argv[2];

    printf("%i\n%s\n%s\n",argc,input_filename,output_filename);

    //Data init.
    unsigned error;
    unsigned char *in, *out;
    unsigned char *d_in, *d_out;
    unsigned width, height;
    int size;

    //Load the input file and turn it into an unsigned char array called 'in' with 'width' and 'height'
    error = lodepng_decode32_file(&in, &width, &height, input_filename);
  	if(error){
  		printf("error %u: %s\n", error, lodepng_error_text(error));
  		return 0;
  	}

    //The size in numbers of values in the 'in' array
    size = width*height*4*sizeof(char);


    //'out' is the new unsigned char array for the editted values from 'in'
    out = (unsigned char*)malloc(size);

    //Malloc within the GPU
    //Give name of the data items we use and their sizes
    cudaMalloc(&d_in, size);
	  cudaMalloc(&d_out, size);


    //Copy over 'in' to 'd_in' in the GPU
    //Can't just move 'in' to GPU evidently
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);



    int threadsPerBlock = height;

    int numBlocks = width;

    printf("Threads per block: %i\tNumber of blocks: %i\n",threadsPerBlock,numBlocks);
    //We start the kernel with the number of blocks we want with the number of threads per block we want
    //Pass the values we allocated in the GPU before so the GPU knows which data items it can use in this kernel
    clock_t tic = clock();
    process<<<numBlocks,threadsPerBlock>>>(d_in, d_out, width, height);

    //Kernels are blocking; use as point of synchronization

    //Copy the values from 'd_out' in GPU memory into 'out' in the CPU memory
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    //Free all that memory in the GPU
    cudaFree(d_out);
    cudaFree(d_in);

    clock_t toc = clock();
    printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

    //send 'out' to be turned into a picture with output name
    lodepng_encode32_file(output_filename, out, width-2, height-2);

    //free all the things
    free(out);
    free(in);
    return 0;
}
