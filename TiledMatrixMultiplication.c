#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Compute C = A * B
#define TILE_WIDTH 16
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
  int Row = blockIdx.x*blockDim.x + threadIdx.x;
  int Col = blockIdx.y*blockDim.y + threadIdx.y;
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  float cValue = 0;
	
  for (int k=0;k<=(numAColumns-1)/TILE_WIDTH;k++){
	if (k*TILE_WIDTH+ty<numAColumns&&Row<numCRows)
	  s_A[tx][ty] = A[Row*numAColumns+k*TILE_WIDTH+ty];
	else
	  s_A[tx][ty] = 0;
	if ((k*TILE_WIDTH+tx)<numBRows&&Col<numCColumns)
	  s_B[tx][ty] = B[(k*TILE_WIDTH+tx)*numBColumns+Col];
	else
	  s_B[tx][ty] = 0;
	__syncthreads();
	for (int i=0;i<TILE_WIDTH;i++){
	  cValue+=s_A[tx][i]*s_B[i][ty];  
	}
	__syncthreads();
  }
	
	if (Row<numCRows&&Col<numCColumns){
	  C[Row*numCColumns+Col] = cValue;
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
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(sizeof(float)*numCRows*numCColumns);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceA,sizeof(float)*numARows*numAColumns);
  cudaMalloc((void **)&deviceB,sizeof(float)*numBRows*numBColumns);
  cudaMalloc((void **)&deviceC,sizeof(float)*numCRows*numCColumns);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA,hostA,sizeof(float)*numARows*numAColumns,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB,hostB,sizeof(float)*numBRows*numBColumns,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC,hostC,sizeof(float)*numCRows*numCColumns,cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  //int TILE_WIDTH = 16;
  dim3 dimGrid((numCRows-1)/TILE_WIDTH+1,(numCColumns-1)/TILE_WIDTH+1,1);
  dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid,dimBlock>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC,deviceC,sizeof(float)*numCRows*numCColumns,cudaMemcpyDeviceToHost);
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