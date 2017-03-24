#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include <time.h>

__global__ void process(unsigned char * d_in, unsigned char * d_out, int d_width, int d_height){

    //Get the thread ID accross all the blocks
    int tid = blockIdx.x*blockDim.x+threadIdx.x;

    //A thread will go through with its ID representing a pixel and go through each
    //RGBA value and edit as necessary
    unsigned char max_R = 0, max_G = 0, max_B = 0, max_A = 0;

    int i = 2*(tid) % d_height;
    int j = 2*(blockIdx.x) % d_width;

    max_R = d_in[4*d_width*i + 4*j] > max_R ? d_in[4*d_width*i + 4*j]:max_R;
    max_R = d_in[4*d_width*i + 4*(j+1)] > max_R ? d_in[4*d_width*i + 4*(j+1)]:max_R;
    max_R = d_in[4*d_width*(i+1) + 4*j] > max_R ? d_in[4*d_width*(i+1) + 4*j]:max_R;
    max_R = d_in[4*d_width*(i+1) + 4*(j+1)] > max_R ? d_in[4*d_width*(i+1) + 4*(j+1)]:max_R;
    d_out[d_width*i + 2*j] = max_R;

    max_G = d_in[4*d_width*i + 4*j + 1] > max_G ? d_in[4*d_width*i + 4*j + 1]:max_G;
    max_G = d_in[4*d_width*i + 4*(j+1) + 1] > max_G ? d_in[4*d_width*i + 4*(j+1) + 1]:max_G;
    max_G = d_in[4*d_width*(i+1) + 4*j + 1] > max_G ? d_in[4*d_width*(i+1) + 4*j + 1]:max_G;
    max_G = d_in[4*d_width*(i+1) + 4*(j+1) + 1] > max_G ? d_in[4*d_width*(i+1) + 4*(j+1) + 1]:max_G;
    d_out[d_width*i + 2*j + 1] = max_G;

    max_B = d_in[4*d_width*i + 4*j + 2] > max_B ? d_in[4*d_width*i + 4*j + 2]:max_B;
    max_B = d_in[4*d_width*i + 4*(j+1) + 2] > max_B ? d_in[4*d_width*i + 4*(j+1) + 2]:max_B;
    max_B = d_in[4*d_width*(i+1) + 4*j + 2] > max_B ? d_in[4*d_width*(i+1) + 4*j + 2]:max_B;
    max_B = d_in[4*d_width*(i+1) + 4*(j+1) + 2] > max_B ? d_in[4*d_width*(i+1) + 4*(j+1) + 2]:max_B;
    d_out[d_width*i + 2*j + 2] = max_B;

    max_A = d_in[4*d_width*i + 4*j + 3] > max_A ? d_in[4*d_width*i + 4*j + 3]:max_A;
    max_A = d_in[4*d_width*i + 4*(j+1) + 3] > max_A ? d_in[4*d_width*i + 4*(j+1) + 3]:max_A;
    max_A = d_in[4*d_width*(i+1) + 4*j + 3] > max_A ? d_in[4*d_width*(i+1) + 4*j + 3]:max_A;
    max_A = d_in[4*d_width*(i+1) + 4*(j+1) + 3] > max_A ? d_in[4*d_width*(i+1) + 4*(j+1) + 3]:max_A;

    d_out[d_width*i + 2*j + 3] = max_A;
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
    out = (unsigned char*)malloc(size/4);

    //Malloc within the GPU
    //Give name of the data items we use and their sizes
    cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, size/4);


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
    cudaMemcpy(out, d_out, size/4, cudaMemcpyDeviceToHost);

    //Free all that memory in the GPU
    cudaFree(d_out);
    cudaFree(d_in);

    clock_t toc = clock();
    printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

    //send 'out' to be turned into a picture with output name
    lodepng_encode32_file(output_filename, out, width/2, height/2);

    //free all the things
    free(out);
    free(in);
    return 0;
}
