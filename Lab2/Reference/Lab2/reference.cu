#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 32

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, 
                               int numARows, int numAColumns, 
                               int numBRows, int numBColumns, 
                               int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here

  /* Allocated shared memory matricies for subtile A and subtile B */
  __shared__ float subTile_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTile_B[TILE_WIDTH][TILE_WIDTH];

  /* Define thread identification info */
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  /* Define global index */
  int global_col = blockIdx.x * blockDim.x + tx;
  int global_row = blockIdx.y * blockDim.y + ty;

  /* Declare and initialize an accumulator */
  float Pvalue = 0;
  
  /* Iterate thru tiles */
  for (int m = 0; m < ((TILE_WIDTH + numAColumns - 1) / TILE_WIDTH); m++) {
    /* Copy tile A from global to shared */
    if (global_row < numARows && (m * TILE_WIDTH + tx) < numAColumns) {
      subTile_A[ty][tx] = A[global_row * numAColumns + (m * TILE_WIDTH + tx)];
    } else {
      subTile_A[ty][tx] = 0.0;
    }
    /* Copy tile B from global to shared */
    if ((m * TILE_WIDTH + ty) < numBRows && global_col < numBColumns) {
      subTile_B[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + global_col];
    } else {
      subTile_B[ty][tx] = 0.0;
    }
    __syncthreads();
    
    /* Calculate Pvalue */
    for (int k = 0; k < TILE_WIDTH; k++) {
      Pvalue += subTile_A[ty][k] * subTile_B[k][tx];
    }
    __syncthreads();
  }
  /* Assert index is in bounds and write Pvalue back to global memory */
  if (global_row < numARows && global_col < numBColumns) {
    C[global_row * numCColumns + global_col] = Pvalue;
  }
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
    
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid(ceil((float)numCColumns / TILE_WIDTH), ceil((float)numCRows / TILE_WIDTH), 1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, 
                                          numARows, numAColumns, 
                                          numBRows, numBColumns,
                                          numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
