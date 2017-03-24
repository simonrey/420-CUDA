/*
Author: Simon Reynders
Class: ECSE 420
Lab 3 - CUDA Rectify

Rectify with one thread per pixel
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lodepng.h"
#include <time.h>


__global__ void process(unsigned char * d_in, unsigned char * d_out){
    
    //Get the thread ID accross all the blocks 
    int tid = blockIdx.x*blockDim.x+threadIdx.x;

    //A thread will go through with its ID representing a pixel and go through each
    //RGBA value and edit as necessary
  	d_out[4*tid] = d_in[4*tid] < 127 ? 127 : d_in[4*tid]; // R
  	d_out[4*tid+1] = d_in[4*tid+ 1] < 127 ? 127 : d_in[4*tid+ 1]; // G
  	d_out[4*tid+2] = d_in[4*tid+ 2] < 127 ? 127 : d_in[4*tid+ 2]; // B
  	d_out[4*tid+3] = d_in[4*tid+ 3]; // A

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

    //Define max number of threads possible per block
    int threadsPerBlock = 1024;

    //We make as many blocks with 1024 threads as necessary to cover all pixels
    int numBlocks = ((width*height)/threadsPerBlock);

    //We start the kernel with the number of blocks we want with the number of threads per block we want
    //Pass the values we allocated in the GPU before so the GPU knows which data items it can use in this kernel


    clock_t tic = clock();
    process<<<numBlocks,threadsPerBlock>>>(d_in, d_out);

	


    //Kernels are blocking; use as point of synchronization

    //Copy the values from 'd_out' in GPU memory into 'out' in the CPU memory
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    //Free all that memory in the GPU
    cudaFree(d_out);
    cudaFree(d_in);

    //find the number of pixels that have not yet been editted by the GPU
    int val=((width*height)%threadsPerBlock);
    printf("Leftover amount : %i\nnumBlocks : %i\n",val,numBlocks);

    //for each uneditted pixel, rectify locally on CPU
    int i;
    for(i = size-val;i<=size;i++){
        out[i] = in[i] < 127 ? 127 : in[i];
    }
    clock_t toc = clock();
    printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
    //send 'out' to be turned into a picture with output name
    lodepng_encode32_file(output_filename, out, width, height);

    //free all the things
    free(out);
    free(in);
    return 0;
}

