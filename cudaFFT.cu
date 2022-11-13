/**********************************************************************
ASSIGNMENT 4: Fast Fourier Transform
Names: Leslie Cook and Parker Hagmaier
Due Date: 10/04/2022
Cooley-Tukey FFT algorithm: radix-2 decimation in time 
this algorithm divides the Discrete Fourier Transform (DFT) over 2 parts 
and sums the even-index (2*n) and odd-index (2*n+1) inputs
then combines them together to form the whole product of complex numbers
this is a general technique of divide and conquer algorithms 
by dividing a large problem into smaller ones and
exploiting special symmetry properties to speed up computation time

The FFT makes use of the following Euler's Identity: 
 e^(iø) = cos(ø)+i*sin(ø)
 cos(-ø) = cos(ø)
 sin(-ø) = -sin(ø)
 i^2 = -1

Our program computes the FFT for a total of 8192 samples
and utilizes a 1 dimension grid of 4 blocks
The max number of threads per block is 1024
so each block will have to be computed twice to reach 8192 total threads
The threads are split into even and odd indexes that computes 4096 elements each
and returns a total of 8192 complex numbers

For verification of the correct output, the 8192 real and 
8192 imaginary values were input into an online FFT calculator 
https://scistatcalc.blogspot.com/2013/12/fft-calculator.html
that produced 8192 real and imaginary ouput values 
that match the values calculated in this program. 
Attched are the text files of the real and imaginary inputs and 
the output from the FFT calculator.
***********************************************************************/

#include <math.h>
#include <stdio.h>
#define size (8192)
const int blocksize = 4;
const int N = 1024;
//struct: complex 
//real: identifies the real values of the complex numbers
//imag: identifies the imaginary values of the complex numbers 
struct Complex{
    double real;
    double imag;
};
//__device__ : tells the program to run on the CPU
//funtction: struct Complex: complexAdd
//params: Complex x, Complex y 
//adds two complex numbers
__device__
struct Complex complexAdd(struct Complex x, struct Complex y){
    struct Complex answer;
    answer.real = (x.real + y.real); 
    answer.imag = (x.imag - y.imag) ;
    return answer;
}

//__device__ : tells the program to run on the CPU
//funtction: struct Complex: complexMul
//params: Complex x, Complex y 
//foil multiplication of the real and imaginary parts of each complex number
__device__
struct Complex complexMul(struct Complex x, struct Complex y){
    struct Complex answer;
    answer.real = (x.real * y.real) + (x.imag * y.imag);
    answer.imag = (x.real* y.imag) - (y.real* x.imag);
    return answer;
}
//__global__ : tells the program to run on the GPU
//funtction: FFT for Fast Fourier Transform
//params: struct Complex *input, struct Complex *answer 
//*input points to our given input values from the time-domain table
//*answer returns our values after FFT calculation
__global__ 
void FFT(struct Complex *input, struct Complex *answer){
    //struct: Complex evenSum
    //will sum of all even index threads
    struct Complex evenSum;
    evenSum.real = 0;
    evenSum.imag = 0;
    //struct: Complex oddSum
    //will sum of all odd index threads
    struct Complex oddSum;
    oddSum.real = 0;
    oddSum.imag = 0;

    //id = thread indexes 
    int id =(blockIdx.x * blockDim.x + threadIdx.x);

    //setting the K value into odd and even indexes with variables named evenK & oddK 
    //recall: the definition of even is 2*n & the definition of odd is 2*n+1
    //evenK will map all even threads [0-8190]
    int evenK = (id * 2); //evenK handles 4096 threads at the even indexes
    //oddK will map all odd threads [1-8191]
    int oddK = (id * 2 + 1); //oddK handles 4096 threads at the odd indexes
    
    //for loop for running the calucations from 0 to (size/2) = 8192/2 = 4096
    //since the complex numbers are divided into odd and even indexes,
    //we only need to loop half the size of the max number of elements
    for (int n=0; n<(size/2); n++)
    {
        //evenFraction  = (-2*PI*n*(2*k)/size) is computed in the even indexes of sine and cosine below
        //recall: evenK = ((blockIdx.x * blockDim.x + threadIdx.x)*2) to account for all 4096 even threads
        double evenFraction = 2 * M_PI * n * evenK /size;
        //oddFraction  = (-PI*n*(2*k+1)/size) is computed in the odd indexes of sine and cosine below
        //recall: oddK = ((blockIdx.x * blockDim.x + threadIdx.x)*2+1) to account for all 4096 odd threads
        double oddFraction = M_PI * n * oddK /size;
        //NOTE: evenK + oddK = 4096 + 4096 = 8192 threads total that will be returned for the answer

        //struct: Complex eulerEven
        //define the euler identies for calculations
        struct Complex eulerEven;
        eulerEven.real = 0;
        eulerEven.imag = 0;
        //struct: Complex eulerOdd
        //define the euler identies for calculations
        struct Complex eulerOdd;
        eulerOdd.real = 0;
        eulerOdd.imag = 0;
        //using the euler identies for complex numbers to calculate the cos() and sin()
        //and sum over all n-even indexes
        eulerEven.real = cos(evenFraction);
        eulerEven.imag = sin(evenFraction);
         //using the euler identies for complex numbers to calculate the cos() and sin()
        //and sum over n-odd indexes + the twiddle factor for oddK 
        eulerOdd.real = cos(oddFraction+oddFraction);
        eulerOdd.imag = sin(oddFraction+oddFraction);
        
        //sums even components using the comnplexAdd function defined above
        //calls complexMul struct defined above to perform foil multiplication
        //using the defined input[n] and eulerEven identies
        evenSum = complexAdd(evenSum, complexMul(input[n], eulerEven));
        //sums even components using the comnplexAdd function defined above
        //calls complexMul struct defined above to perform foil multiplication
        //using the defined input[n] and eulerOdd identies
        oddSum = complexAdd(oddSum, complexMul(input[n], eulerOdd));
}
//returns the answer mapped to the thread id's of the even index threads
answer[evenK] = evenSum;
//returns the answer mapped to the thread id's of the odd index threads
answer[oddK] = oddSum;
}

int main() {
  //fsize variable gets the amount of total bytes needed in memory
  //sizeof(struct Complex) in bytes * the size of the array (size=8192)
  const int fsize = size*sizeof(struct Complex);
  //initializing the 8 real values given in the time domain table
  double eightReal[8] = {3.6, 2.9, 5.6, 4.8, 3.3, 5.9, 5, 4.3};
  //initializeing the 8 imaginary values given in the time domain table
  double eightImag[8] = {2.6, 6.3, 4, 9.1, 0.4, 4.8, 2.6, 4.1};

  //input[size]: defines the input array of 8192 samples that will hold the
  //eight given time-domain table values and initalize everthing else to zero
  struct Complex input[size];
  //temp is used a as a temporary place holder that is used for setting
  //the complex numbers into the input array of real and imaginary values
  struct Complex temp;
  for (int i =0; i < size; i ++){
    //if the index (i) is 0-7 set the eightReal and eightImag values (from above) 
    //into the temp place holder, then set the the input array to temp
    if (i < 8){
      temp.real = eightReal[i];
      temp.imag = eightImag[i];
      input[i] = temp; //the given time-domain values are set in the first 8 indexes 
    }
    //else, if the i is 8-8192 set the values in the input array to 0
    else{
      temp.real = 0;
      temp.imag = 0;
      input[i] = temp; //the rest of the real and imaginary arrays are set to 0
    }
  }
  struct Complex *d_input; //device copy of the input array values
  struct Complex *d_answer; //device copy of the answer to be returned after calculations
  //array to hold our results after the calucuations 
  struct Complex results[size];
  //allocates the memory on the GPU side 
  cudaMalloc((void**)&d_input, fsize);
  cudaMalloc((void**)&d_answer, fsize);
  //copies the memory on the CPU side to the GPU 
  cudaMemcpy( d_input, input, fsize, cudaMemcpyHostToDevice );
  cudaMemcpy( d_answer, results, fsize, cudaMemcpyHostToDevice );
  //1-D grid with 4 blocks
  dim3 fourBlocks(blocksize, 1);  
  //1-D block with 1024 threads per block  
  dim3 threads(N, 1);
   //calls the GPU functions with the parameters d_input, d_answer
  FFT<<<fourBlocks, threads>>> (d_input, d_answer);
  //copies the GPU memory back to the CPU and
  //stores the answer from the GPU (d_answer) into an array named results on the CPU side
  cudaMemcpy(results, d_answer, fsize, cudaMemcpyDeviceToHost);
  //deallocate GPU memory
  cudaFree( d_input );
  //display the first eight results
  printf("==================================\n");
  printf("  Total Processed Samples: %i\n", size);
  printf("==================================\n");
  for (int i = 0; i < 8; i++) {
    printf("k = %i \t %f  +  %fi \n", i, results[i].real, results[i].imag);
  }
  printf("==================================\n");
  //display results in threads 4096-4103
  for (int i = 4096; i < 4104; i++) {
    printf("k = %i %f  +  %fi \n", i, results[i].real, results[i].imag);
  }
  printf("==================================\n");
  //deallocate GPU memory
  cudaFree(d_answer);

  return 0;
}