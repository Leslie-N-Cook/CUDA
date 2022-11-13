#include<stdio.h>
#include<unistd.h>
#include<sys/types.h>

# define SIZE 10240         //total number of threads

const int N = 1024;         //threads per block
const int BlockSize = 10; 

 /*******************************************************************/

 __global__
void tenBlocks(unsigned long int *a_d, unsigned long int *b_d, unsigned long int *c_d)
{
     //tid : index of the thread
   int tid = blockIdx.x * blockDim.x + threadIdx.x;

    c_d[tid] = a_d[tid] * b_d[tid];
}
int main()
{

    unsigned long int *a_d, *b_d, *c_d; // device copies of a, b, c
    unsigned long int *a_h, *b_h, *c_h; // host copies of a, b, c

    //NOTE:This gets the amount of bytes needed for the array
    //sizeof(unsigned long int) in bytes times the size of the array
    const unsigned long int iSize = SIZE * sizeof(unsigned long int);

     //allocates the memory on the CPU side with the size 
    //computed above needs unsigned long int* to make sure that it is an array of unsigned long ints
    a_h = (unsigned long int*)malloc(iSize);
    b_h = (unsigned long int*)malloc(iSize);
    c_h = (unsigned long int*)malloc(iSize);

    //NOTE:This must be done BEFORE copping memory
    //loadunsigned long int the arrays
    for (unsigned long int n = 0; n < SIZE; ++n)
    {
        a_h[n] = 2 * n;         //even numbers in array a
        b_h[n] = (2 * n) + 1;   //odd numbers in array b
        c_h[n] = 0;             //array c initialized to 0
    }
    cudaMalloc((void**) &a_d, iSize);
    cudaMalloc((void**) &b_d, iSize);
    cudaMalloc((void**) &c_d, iSize);

    //copies the memory on the cpu side to the GPU 
    cudaMemcpy(a_d, a_h, iSize, cudaMemcpyHostToDevice);
 	cudaMemcpy(b_d, b_h, iSize, cudaMemcpyHostToDevice);
 
    //1-D grid with 10 blocks
	dim3 dimGridten(BlockSize ,1); 	

    //1-D block with 1024 threads per block 
	dim3 dimBlockten(N, 1);

    //calls the GPU functions with the perameters a_d, b_d, c_d
	tenBlocks <<<dimGridten, dimBlockten>>>(a_d, b_d, c_d);

    //copies the GPU memory to the CPU c_d->c_h
    cudaMemcpy(c_h, c_d, iSize, cudaMemcpyDeviceToHost );


    //deallocates the GPU memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
   

    //display the solution
     printf("\nSolution 3:\nTen blocks (c[0], c[10239]) = (");
     printf("%d", c_h[0]);
     printf(", ");
     printf("%d", c_h[10239]);
     printf(")\n");
  
    //deallocates the CPU memory
    free(a_h);
    free(b_h);
    free(c_h);

    return 0;

}