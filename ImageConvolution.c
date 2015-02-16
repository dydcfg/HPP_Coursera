#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

#define BlockSize 16
#define OBlockSize BlockSize-Mask_width-1

//@@ INSERT CODE HERE

__global__ void imageConvolution(float *graphMatrix, float *oGraphMatrix, const float* __restrict__ M, int imageWidth, int imageHeight, int imageChannels, int maskRows, int maskColumns){

  __shared__ float sGraph[BlockSize][BlockSize][3];	
  int inCol = blockIdx.x*(blockDim.x - Mask_width + 1) + threadIdx.x - Mask_radius;
  int inRow = blockIdx.y*(blockDim.y - Mask_width + 1) + threadIdx.y - Mask_radius;
  int outCol = inCol+Mask_radius;
  int outRow = inRow+Mask_radius;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  float Value[3];
	
	for (int i=0;i<3;i++){
		if (inRow<imageHeight&&inRow>=0&&inCol<imageWidth&&inCol>=0)
            sGraph[ty][tx][i] = graphMatrix[(inRow*imageWidth+inCol)*imageChannels+i];
		else 
			sGraph[ty][tx][i] = 0;
		Value[i] = 0;
	}
  __syncthreads();
  
  if (ty+Mask_width-1<BlockSize&&tx+Mask_width-1<BlockSize)
  for (int i=0;i<Mask_width;i++)
	  for (int j=0;j<Mask_width;j++)
	      for (int k=0;k<imageChannels;k++)
	          Value[k]+=M[i*maskColumns+j]*sGraph[ty+i][tx+j][k];

  for (int i=0;i<3;i++){
	  if (Value[i]<0)
		  Value[i] = 0;
	  if (Value[i]>1)
		  Value[i] = 1;
  }
  
  if (outCol<imageWidth&&outRow<imageHeight&&tx<BlockSize-Mask_width+1&&ty<BlockSize-Mask_width+1)
	  for (int i=0;i<imageChannels;i++)
          oGraphMatrix[(outRow*imageWidth+outCol)*imageChannels+i] = Value[i];
	  
  __syncthreads();
}



int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	dim3 dimGrid((imageWidth-1)/(BlockSize-Mask_width+1)+1,(imageHeight-1)/(BlockSize-Mask_width+1)+1,1);
	dim3 dimBlock(BlockSize,BlockSize,1);
	imageConvolution<<<dimGrid,dimBlock>>>(deviceInputImageData,deviceOutputImageData,deviceMaskData,imageWidth,imageHeight,imageChannels,maskRows,maskColumns);
	
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}