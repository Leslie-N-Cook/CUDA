/************************************************************
Assignment 5: Convolution
Names: Parker Hagmaier and Leslie Cook
Due Date: 11/07/2022

A convolution is a mathematical operation on two functions
F(m) and G(n), that produces in a third function H(n).
H(n) results in a size that is F(m) + G(n) - 1 combined.
F(m) has a total of 4096 (M) elements in ascending order starting at 1
G(n) has a total of 1024 (N) elements in decending order startingn at 1024
H(n) is returned with a total of 5119 elements 

Mathematical expression:
For finite sequences F(m) with M values and G(n) with N values,
(F * G)(n) = H(n) = ∑ f(m)g(n-m), where n = 0 ... N+M-2, and
F(m) = 0 when m < 0 and m ≥ M, and
G(n) = 0 when n < 0 and n ≥ N
******************************************************/
#include <stdio.h>
#include <stdlib.h>
#define M 4096
#define N 1024

int main () 
{  
  //dynamically allocate the size of the arrays F, G, and H
  //hSize gets the memory in bytes for the arrays F, G, & H, where M+N-1=5119
   int hSize = (N + M - 1) * sizeof(int);
    
    int* F = (int*)malloc(hSize); //host copy for array F
    int* G = (int*)malloc(hSize); //host cooy for array G
    int* H = (int*)malloc(hSize); //host copy for array H
   
  //initialize array F with 4096 consecutive ascending interger numbers starting at 1
    for (int i = 0; i < M; ++i)
    {
        F[i] = i+1;
    }
  //initialize array G with 1024 decending integer numbers starting at 1024
     int temp = N; // set a temporay variable to 1024
     for (int j = 0; j < N; ++j)
    { 
        G[j] = temp; //set G[0] to 1024
        temp = temp - 1; //update temp - 1 for decending integers
    }

   //initialize array H, setting M+N-1 = 4096+1024-1 = 5119 elements to zero
    for (int k = 0; k<(M+N); ++k)
    {
      H[k] = 0;
    }
      int sum = 0;
    // start convolution loop
     for (int n = 0; n < (M+N); ++n)
     {
      //if n is less that zero and greater than or equal to N (1024) 
      //set G[n] equal to zero
       if (n<0 && n>=N)
        G[n]=0;
       for (int m = 0; m <= n; ++m)
       {
        //if m < 0 and m is greater than or equal to M (4096)
        //set F[m] equal to zero
       if (m<0 && m>=M)
        F[m] = 0;
       //the result multiplies aligning F and G arrays 
       // and sums the elements to store in array H
        H[n] += F[m] * G[n-m];
       }
       //perform a checksum and add all elements of the array together
        sum = sum + H[n];
     }// end convolution loop
    //display deliverables
    printf("\nNumber of elements in array F: %d\n", M);
    printf("Number of elements in array G: %d\n", N);
    printf("Number of elements in array H: %d\n\n", M+N-1);
    printf("Sum of all elements in array H: %d\n\n", sum);
    //write output for array H to a .csv file
    FILE *fptr; //*fptr: file pointer
    fptr = fopen("Parker_Leslie_A5output.csv", "w+"); //open .csv file
    fprintf(fptr,"Convolution array H:\n");
     for (int x=0; x<(M+N-1); x++)
    {
      fprintf(fptr,"H[%d]: %d\n",x,H[x]); //write array H(n) to .csv file
    }
   fclose(fptr); //close .csv file
    
   return 0;
}
