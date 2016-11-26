#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define CHECK(e) { int res = (e); if (res) printf("CUDA ERROR %d\n", res); }

// 11 is 2048
#define N (1 << 6)


    __global__
void add(int *a, int *c)
{
    int tid = (blockIdx.x * gridDim.y * blockDim.x * blockDim.y) + (blockIdx.y * blockDim.x * blockDim.y) + (threadIdx.x * blockDim.y) + threadIdx.y;

    /*int row = blockDim.x * blockIdx.x + threadIdx.x;*/
    /*int col = blockDim.y * blockIdx.y + threadIdx.y;*/

    /*int tid = ((gridDim.y * blockDim.y) * col) + row;*/

    int i = tid / N;
    int j = tid % N;

    if (i < N && j < N)
    {
        c[i * N + j] = a[j * N + i];
    }
}
    /*int row = blockDim.x * blockIdx.x + x;*/
    /*int col = blockDim.y * blockIdx.y + threadIdx.y;*/

    /*int tid = (gridDim.y * blockDim.y * blockDim.y * blockIdx.y) + (gridDim.y * blockDim.y * threadIdx.y) + (blockDim.x * blockIdx.x) + threadIdx.x;*/



    /*__global__*/
/*void add(int *a, int *c)*/
/*{*/
    /*__shared__ int cache[N/2][N/2];*/

    /*int x = threadIdx.x;*/
    /*int y = threadIdx.y;*/

    /*int row = blockDim.x * blockIdx.x + x;*/
    /*int col = blockDim.y * blockIdx.y + y;*/

    /*int tid = ((gridDim.y * blockDim.y) * col) + row;*/
    /*int i = tid / N;*/
    /*int j = tid % N;*/

    /*x = threadIdx.y;*/
    /*y = threadIdx.x;*/

    /*int row2 = blockDim.x * blockIdx.x + x;*/
    /*int col2 = blockDim.y * blockIdx.y + y;*/

    /*int tid2 = ((gridDim.y * blockDim.y) * col2) + row2;*/
    /*int i2 = tid2 / N;*/
    /*int j2 = tid2 % N;*/


    /*c[i2 * N + j2] = a[i * N + j];*/

    /*c[i * N + j] = a[j * N + i];*/
    /*c[i * N + j] = a[i2 * N + j2];*/
    /*c[i * N + j] = 1;*/
/*}*/

    /*__shared__ int cache[N/2][N/2];*/

    /*int x = threadIdx.x;*/
    /*int y = threadIdx.y;*/

    /*int row = blockDim.x * blockIdx.x + x;*/
    /*int col = blockDim.y * blockIdx.y + y;*/

    /*int tid = ((gridDim.y * blockDim.y) * col) + row;*/
    /*int i = tid / N;*/
    /*int j = tid % N;*/

    /*cache[y][x] = a[i * N + j];*/

    /*__syncthreads();*/

    /*c[j * N + i] = cache[y][x];*/


void transpose(int *a)
{
    int i, j, t, x;

    for(t = 0 ; t < N*N ; t++)
    {
        i = t / N;
        j = t % N;

        if (i < j)
        {
            x = a[i*N + j];
            a[i*N + j] = a[j*N + i];
            a[j*N + i] = x;
        }
    }
}

void print(int *a)
{
    int t = 0;
    int i = 0;
    int j = 0;


    for(t = 0 ; t < N*N ; t++)
    {
        i = t / N;
        j = t % N;

        if (j == 0)
        {
            printf( "\n");
        }

        printf("%d ", a[i*N + j]);

    }
    printf( "\n");
    printf( "\n");
}

void printTemp(int *a)
{
    int i = 0;
    int j = 0;

    for(i = 0 ; i < N ; i++)
    {
        for(j = 0 ; j < N ; j++)
        {
            printf("%d ", a[i*N + j]);
        }
        printf( "\n");
    }
    printf( "\n");
    printf( "\n");
}

bool isTranspose(int *a, int *c)
{
    int i, j, t;

    for(t = 0 ; t < N*N ; t++)
    {
        i = t / N;
        j = t % N;

        if (a[i*N + j] != c[j*N + i])
        {
            return false;

        }
    }

    return true;
}



int gmain(int argc, char **argv)
{
    int i = 0;
    int j = 0;

    int *a;
    int *c;


    // Allocate Normal Memory
    a = (int *)malloc(N * N * sizeof(int));
    c = (int *)malloc(N * N * sizeof(int));

    // Initialize some values
    for(i = 0 ; i < N ; i++)
    {
        for(j = 0 ; j < N ; j++)
        {
            a[i*N + j] = (i * j + i) % 10;
        }
    }

    int *devA;
    int *devC;

    // Initialize Cuda Memory
    CHECK(cudaMalloc(&devA, N * N * sizeof(int)));
    CHECK(cudaMalloc(&devC, N * N * sizeof(int)));

    // Copy Cuda Memory
    CHECK(cudaMemcpy(devA, a, N * N * sizeof(int), cudaMemcpyHostToDevice));

    // Run the kernel
    dim3 dimBlock(2, 2);
    dim3 dimGrid(N/2, N/2);
    add<<<dimBlock,dimGrid>>>(devA, devC);

    // Return the Cuda Memory
    CHECK(cudaMemcpy(c, devC, N * N * sizeof(int), cudaMemcpyDeviceToHost));

    /*print(a);*/
    /*print(c);*/
    assert(isTranspose(a, c));

    printf("Transposition went well!");


    // Run free my darlings
    cudaFree(devA);
    cudaFree(devC);

    free(a);
    free(c);

    return 0;
}
