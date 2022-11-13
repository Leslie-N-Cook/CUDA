#include<stdio.h>
#include<unistd.h>
#include<sys/types.h>

# define SIZE 10240         //total number of threads
const int N = 1024;         //threads per block
const int blocksize = 2;    //solution a and b requires 2 blocks

 /*******************************************************************/
 __global__ 
 void nonCyclic(unsigned long int *a_d, unsigned long int *b_d, unsigned long int *c_d)
{
    //tid : index of the thread
    int tid = blockIdx.x * blockDim.x * (SIZE / blockDim.x / gridDim.x) + threadIdx.x;
    
    for (int j = 0; j< (SIZE / blockDim.x / gridDim.x); ++j) // (10240 / 1024 / 2) = 5
    {
        c_d[tid + (blockDim.x * j)] = a_d[tid + (blockDim.x * j)] * b_d[tid + (blockDim.x * j)];
    }
}

int main()
{

    unsigned long int *a_d, *b_d, *c_d; // device copies of a, b, c
    unsigned long int *a_h, *b_h, *c_h; // host copies of a, b, c

    //NOTE:This gets the amount of bytes needed for the array
    //sizeof(unsigned long int) in bytes times the size of the array
    const unsigned long int iSize = SIZE * sizeof(unsigned long int);

    /****************** solution #1 - Two blocks: NONcyclic ******************/
    //allocates the memory on the CPU side with the size 
    //computed above needs unsigned long int* to make sure that it is an array of unsigned long ints
    a_h = (unsigned long int*)malloc(iSize);
    b_h = (unsigned long int*)malloc(iSize);
    c_h = (unsigned long int*)malloc(iSize);

    //NOTE:This must be done BEFORE copping memory
    //loading the arrays
    for (int n = 0; n < SIZE; ++n)
    {
        a_h[n] = (2 * n);         //even numbers in array a
        b_h[n] = ((2 * n) + 1);   //odd numbers in array b
        c_h[n] = 0;             //array c initialized to 0
    }

    //allocates the memory on the GPU size void** and & is just needed
    cudaMalloc((void**) &a_d, iSize);
    cudaMalloc((void**) &b_d, iSize);
    cudaMalloc((void**) &c_d, iSize);

    //copies the memory on the cpu side to the GPU 
    cudaMemcpy(a_d, a_h, iSize, cudaMemcpyHostToDevice);
 	cudaMemcpy(b_d, b_h, iSize, cudaMemcpyHostToDevice);

    //1-D grid with 2 blocks
	dim3 gridDim_non(blocksize ,1); 	

    //1-D block with 1024 threads per block 
	dim3 blockDim_non(N, 1);

    //calls the GPU functions with the perameters a_d, b_d, c_d
	nonCyclic<<<gridDim_non, blockDim_non>>>(a_d, b_d, c_d);

    //copies the GPU memory to the CPU 
    cudaMemcpy(c_h, c_d, iSize, cudaMemcpyDeviceToHost);

    //deallocate GPU memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    //display the solution
    printf("\nSolution 1:\nTwo blocks and Non-cyclic (c[0], c[10239]) = (");
    printf("%d", c_h[0]);
    printf(", ");
    printf("%d", c_h[10239]);
    printf(")\n");

    //deallocate CPU memory
    free(a_h);
    free(b_h);
    free(c_h);
    
     /*******************************************************************/
    return 0;
}