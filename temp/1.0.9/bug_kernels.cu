/*
 * bug.h
 *
 * Bug CUDA kernels
 *
 *  Created on: Mar 3, 2012
 *      Author: Maxym Zastavny
 */

//#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <cublas.h>
#include <cufft.h>
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h>

#include "hdr/bug_struct.h"


using namespace std;

__global__ void setTo(float * a, float value)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    a[index] = value;
}

__global__ void divide(float * a, float value)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    a[index] = abs(a[index])/value;
}

__global__ void limitation(float * a, int value)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    a[index] = (int)abs(a[index]);

    if(a[index] >= value)
    {
    	a[index] = value;
    }
}

__global__ void divideMat(float * a, float * b, float value)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(b[index] != 0)
    {
    	a[index] = a[index] / sqrt(b[index]);
    }
    else
    {
    	a[index] = a[index] / sqrt(value);
    }
}

__global__ void genFx(float * a, int width, float yf, float xf)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int y = index / width;
    int x = index - y * width;

    a[index] = exp(- (abs(y) / yf + abs(x) / xf) / sqrtf(2));
}

__global__ void devGenFx(float * a, int width, float yf, float xf)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int y = index / width;
    int x = index - y * width;

    a[index] = exp(- (abs(y) / yf + abs(x) / xf) / sqrtf(2));
}


__global__ void matrixMul(cufftComplex * a, cufftComplex * b, cufftComplex * c)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    c[index].x =  a[index].x * b[index].x + a[index].y * b[index].y;
    c[index].y =  a[index].y * b[index].x - a[index].x * b[index].y;

}

__global__ void matrixSQR(float * a, float * b)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    b[index] =  a[index] * a[index];
}


__global__ void computeShift(float * frame1, float * frame2, float * tempMatrix, int xShift, int yShift, window searchWindow, window capturedWindow)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int delta = (capturedWindow.width - searchWindow.width) / 2 + capturedWindow.width * (capturedWindow.height - searchWindow.height) / 2;
    int delta2 = (index / searchWindow.width) * (capturedWindow.width - searchWindow.width);

    tempMatrix[index] = frame1[index + delta + delta2] - frame2[index + delta  + delta2 + xShift + yShift * capturedWindow.width];
}

__global__ void subtract(float * frame1, float * frame2, float * frame3, int threshold)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    frame3[index] =  abs(frame1[index] - frame2[index]);

    if(frame3[index] < threshold)
        frame3[index] = 0;

    frame3[index] = frame3[index] * 5;

    if(frame3[index] > 255)
        frame3[index] = 255;
}

__global__ void subtractWithShift(float * frame1, float * frame2, float * frame3, int threshold, shift imageShift, int stringLength)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int frame1Y = index / stringLength;
    int frame1X = index - frame1Y * stringLength;

    if(frame1X + imageShift.x >= 0 && frame1X + imageShift.x < stringLength && frame1Y + imageShift.y >= 0 && frame1Y + imageShift.y < stringLength)
    {
        frame3[index] =  abs(frame1[index] - frame2[index + imageShift.x + imageShift.y * stringLength]);
    }
    else
    {
        frame3[index] = 0;
    }

    if(frame3[index] < threshold)
        frame3[index] = 0;

    frame3[index] = frame3[index] * 5;

    if(frame3[index] > 255)
        frame3[index] = 255;
}

__global__ void compare(float * frame1, float * frame2, int threshold)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(abs(frame1[index] - frame2[index]) < threshold)
        frame1[index] = 0;
}

__global__ void exchanngeFrames(float * frame1, float * frame2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    frame1[index] = frame2[index];
}

void channgeFramesSequence(float * devImage1, float * devImage2, float * devImage3, dim3 blocks, dim3 threads)
{
    blocks.x = blocks.x * 2; /// !!!!

    exchanngeFrames <<<blocks, threads>>> (devImage1, devImage2);
    exchanngeFrames <<<blocks, threads>>> (devImage2, devImage3);
}

__global__ void getWindow(float * frame1, float * frame2, window searchWindow, window capturedWindow)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int delta = (capturedWindow.width - searchWindow.width) / 2 + capturedWindow.width * (capturedWindow.height - searchWindow.height) / 2;
    int delta2 = (index / searchWindow.width) * (capturedWindow.width - searchWindow.width);

    frame1[index] = frame2[index + delta + (int)searchWindow.centerX + delta2 + (int)searchWindow.centerY * capturedWindow.width];
}

__global__ void noralizeImage(float *image, int max, float maxValue, float minValue)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    image[index] = (max/(maxValue - minValue))*(image[index] - minValue);
}

string to_string(int val)
{
    char buff[32];
    sprintf(buff,"%d",val);
    return string(buff);
}
/*
__global__ void integralScan(float * input, float * output,int n)
{
    extern __shared__ float temp[];
    int tdx = threadIdx.x; int offset = 1;
    temp[2*tdx] = input[2*tdx];
    temp[2*tdx+1] = input[2*tdx+1];

    for(int d = n>>1; d > 0; d >>= 1)
    {
        __syncthreads();
        if(tdx < d)
        {
            int ai = offset*(2*tdx+1)-1;
            int bi = offset*(2*tdx+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if(tdx == 0) temp[n - 1] = 0;
    for(int d = 1; d < n; d *= 2)
    {
        offset >>= 1; __syncthreads();
        if(tdx < d)
        {
            int ai = offset*(2*tdx+1)-1;
            int bi = offset*(2*tdx+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();
    output[2*tdx]= temp[2*tdx];
    output[2*tdx+1] = temp[2*tdx+1];
}
*/
__global__ void integralScan(float * input, float * output,int n)
{
    extern __shared__ float temp[];
    int tdx = threadIdx.x; int offset = 1;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    temp[2*tdx] = input[2*tdx];
//    temp[2*tdx+1] = input[2*tdx+1];
    temp[2*tdx] = input[2*index];
    temp[2*tdx+1] = input[2*index+1];

    for(int d = n>>1; d > 0; d >>= 1)
    {
        __syncthreads();
        if(tdx < d)
        {
            int ai = offset*(2*tdx+1)-1;
            int bi = offset*(2*tdx+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if(tdx == 0) temp[n - 1] = 0;
    for(int d = 1; d < n; d *= 2)
    {
        offset >>= 1; __syncthreads();
        if(tdx < d)
        {
            int ai = offset*(2*tdx+1)-1;
            int bi = offset*(2*tdx+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();
//    output[2*tdx]= temp[2*tdx];
//    output[2*tdx+1] = temp[2*tdx+1];
    output[2*index]= temp[2*tdx];
    output[2*index+1] = temp[2*tdx+1];

}

__global__ void integralTranspose(float *input, float *output, int width, int height)
{
    #define BLOCK_DIM 16
    __shared__ float temp[BLOCK_DIM][BLOCK_DIM+1];
    int xIndex = blockIdx.x*BLOCK_DIM + threadIdx.x;
    int yIndex = blockIdx.y*BLOCK_DIM + threadIdx.y;

    if((xIndex < width) && (yIndex < height))
    {
        int id_in = yIndex * width + xIndex;
        temp[threadIdx.y][threadIdx.x] = input[id_in];
    }
    __syncthreads();

    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if((xIndex < height) && (yIndex < width))
    {
        int id_out = yIndex * height + xIndex;
        output[id_out] = temp[threadIdx.x][threadIdx.y];
    }
}

__global__ void thresholdImage(float *image, int threshold, int mode)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(image[index] < threshold)
    {
        image[index] = 0;
    }
    else if(mode == 1)
    {
    	image[index] = 255;
    }
}

__global__ void gaussFiltration(float *id, float *od, int w, int h, float a)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if (x >= w) return; //ограничение количества общих блоков

    id += x;    // advance pointers to correct column
    od += x;

    // forward pass
    float yp = *id;  // previous output
    for (int y = 0; y < h; y++)
    {
        float yc = *id + a*(yp - *id);   // simple lerp between current and previous value
		*od = yc;
        id += w; od += w;    // move to next row
        yp = yc;
    }

    // reset pointers to point to last element in column
    id -= w;
    od -= w;

    // reverse pass
    // ensures response is symmetrical
    yp = (*id);
    for (int y = h-1; y >= 0; y--)
    {
        float yc = *id + a*(yp - *id);
		*od = (*od + yc)*0.5f;
        id -= w; od -= w;  // move to previous row
        yp = yc;
    }
}

__global__ void sobelOperator(float *input_data, float *output_data, window procWindow)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int yidx = index / procWindow.width;
    int xidx = index - yidx * procWindow.width;

    if(xidx > 0 && yidx > 0 && xidx < procWindow.width - 1 && yidx < procWindow.height - 1)
    {
        output_data[index] = abs(
            input_data[index - procWindow.width - 1] * (-1) +
            input_data[index - procWindow.width - 0] * (-2) +
            input_data[index - procWindow.width + 1] * (-1) +
            input_data[index -      0           - 1] * (0)  +
            input_data[index -      0           - 0] * (0)  +
            input_data[index -      0           + 1] * (0)  +
            input_data[index + procWindow.width - 1] * (1)  +
            input_data[index + procWindow.width - 0] * (2)  +
            input_data[index + procWindow.width + 1] * (1));

        output_data[index] += abs(
            input_data[index - procWindow.width - 1] * (-1) +
            input_data[index - procWindow.width - 0] * (0)  +
            input_data[index - procWindow.width + 1] * (1)  +
            input_data[index -      0           - 1] * (-2) +
            input_data[index -      0           - 0] * (0)  +
            input_data[index -      0           + 1] * (2)  +
            input_data[index + procWindow.width - 1] * (-1) +
            input_data[index + procWindow.width - 0] * (0)  +
            input_data[index + procWindow.width + 1] * (1));
    }
    else
    {
        output_data[index] = 0;
    }
}

__global__ void updateEtalonWindow (float *input_data, float *output_data, window inputWindow, window outputWindow, float coef)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int yidx = index / inputWindow.width;
    int xidx = index - yidx * inputWindow.width;

    int deltax = (inputWindow.width - outputWindow.width) / 2 + outputWindow.centerX;
    int deltay = (inputWindow.height - outputWindow.height) / 2 + outputWindow.centerY;

    if(xidx > deltax && yidx > deltay && xidx < deltax + outputWindow.width && yidx < deltay + outputWindow.height)
    {
        output_data[index] = output_data[index] * (1. - coef) +
        input_data[index + (int)outputWindow.centerX + (int)outputWindow.centerY * inputWindow.width] * coef;
    }

    if(xidx <= (inputWindow.width - outputWindow.width) / 2 ||
       xidx >= (inputWindow.width - outputWindow.width) / 2 + outputWindow.width ||
       yidx <= (inputWindow.height - outputWindow.height) / 2 ||
       yidx >= (inputWindow.height - outputWindow.height) / 2 + outputWindow.height)
    {
    	output_data[index] = 0;
    }

}

__global__ void updateEtalonWindow2 (float *input_data, float *output_data, window inputWindow, window outputWindow, float coef)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int yidx = index / inputWindow.width;
    int xidx = index - yidx * inputWindow.width;

    int deltax = (inputWindow.width - outputWindow.width) / 2 + outputWindow.centerX;
    int deltay = (inputWindow.height - outputWindow.height) / 2 + outputWindow.centerY;

    if(xidx > deltax && yidx > deltay && xidx < deltax + outputWindow.width && yidx < deltay + outputWindow.height)

    {
       // output_data[index] = output_data[index] * ( 1 - coef) +
         // input_data[index + (int)outputWindow.centerX + (int)outputWindow.centerY * inputWindow.width] * coef;

        output_data[index] = input_data[index + (int)outputWindow.centerX + (int)outputWindow.centerY * inputWindow.width];
    }

    else
    {
        output_data[index] = 0;
    }

}

void normalize(float *image, window searchWindow, int max)
{
    int maxIdx = cublasIsamax(searchWindow.width * searchWindow.height, image, 1);
    int minIdx = cublasIsamin(searchWindow.width * searchWindow.height, image, 1);

    float *maxValue;
    float *minValue;

    maxValue = (float *)calloc(1, sizeof(float));
    minValue = (float *)calloc(1, sizeof(float));

    cudaMemcpy(minValue, image + minIdx - 1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(maxValue, image + maxIdx - 1, sizeof(float), cudaMemcpyDeviceToHost);

    noralizeImage <<<searchWindow.width, searchWindow.height>>> (image, max, *maxValue, *minValue);

    free(minValue);
    free(maxValue);
}

void getShifts(float * frame1, float * frame2, float * frame3, float * devTempMatrix, int T, int maxShift, dim3 blocks, dim3 threads, window searchWindow, window capturedWindow)
{
    float minDifference, sum;
    int deltaX, deltaY, deltaX2, deltaY2;


    for(int yShift = -maxShift; yShift < maxShift; yShift++)
    {
        for(int xShift = -maxShift; xShift < maxShift; xShift++)
        {
            computeShift <<<blocks, threads>>> (frame1, frame2, devTempMatrix, xShift, yShift, searchWindow, capturedWindow);
            sum = cublasSasum(searchWindow.width * searchWindow.height, devTempMatrix, 1);
            if(xShift == - maxShift && yShift == - maxShift)
            {
                minDifference = sum;
                deltaX = xShift;
                deltaY = yShift;
            }
            if(sum < minDifference)
            {
                minDifference = sum;
                deltaX = xShift;
                deltaY = yShift;
            }
        }
    }
   // cout<< "First : " << deltaY<<" "<<deltaX << " " << minDifference << endl;

    for(int yShift = -maxShift + deltaY; yShift < maxShift + deltaY; yShift++)
    {
        for(int xShift = -maxShift + deltaX; xShift < maxShift + deltaX; xShift++)
        {
            computeShift <<<blocks, threads>>> (frame1, frame3, devTempMatrix, xShift, yShift, searchWindow, capturedWindow);
            sum = cublasSasum(searchWindow.width * searchWindow.height, devTempMatrix, 1);
            if(xShift == - maxShift && yShift == - maxShift)
            {
                minDifference = sum;
                deltaX2 = xShift;
                deltaY2 = yShift;
            }
            if(sum < minDifference)
            {
                minDifference = sum;
                deltaX2 = xShift;
                deltaY2 = yShift;
            }
        }
    }
    //cout<<"Second : "<< deltaY2<<" "<<deltaX2 << " " << minDifference << endl;
}

__global__ void cuAdd(float * a, float add)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    a[index] = a[index] + add;
}

__global__ void initIntensityValues(float * intensity, float * particleX, float * particleY, float * velocityX, float * velocityY, float * inpuImage, int stringLength)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    intensity[index] = (float) inpuImage[(int) particleX[index] + (int) velocityX[index] + ((int) particleY[index] + (int) velocityY[index]) * stringLength];
}

__global__ void initIntensityDeviationMatrix(float * intensityDeviation, float sigma)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    intensityDeviation[index] = exp( - (index * index) / (2 * sigma * sigma) ) / sqrt(2 * 3.141592653 * sigma);
}

__global__ void initPositionDeviationMatrix(float * positionDeviation, int stringLength, float sigma)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int deltaY = (int)(index / stringLength) - stringLength / 2;
    int deltaX = index - (deltaY + stringLength / 2) * stringLength - stringLength / 2;

    positionDeviation[index] = exp( - (deltaX * deltaX + deltaY * deltaY) / (2 * sigma * sigma) ) / sqrt(2 * 3.141592653 * sigma * sigma);
}

__global__ void procPDFIntensity(float * PDF, float * intensity, float * deviationMatrix, float * inputImage, float * particleX,  float * particleY, int stringLength, int widthPDF, int heightPDF)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    int particleNumber = index / (widthPDF * heightPDF);
    int positionPDFY = (index - particleNumber * (widthPDF * heightPDF)) / (widthPDF) - (heightPDF) / 2;
    int positionPDFX =  (index - particleNumber * (widthPDF * heightPDF)) - positionPDFY * (widthPDF) - (heightPDF) * (widthPDF) / 2;

//    PDF[index] = deviationMatrix[(int) abs(inputImage[(int) particleX[particleNumber] + positionPDFX +
//      ((int) particleY[index / (widthPDF * heightPDF)] + positionPDFY) * stringLength] - intensity[particleNumber])];
    PDF[index] = abs(inputImage[(int) particleX[particleNumber] + positionPDFX +
      ((int) particleY[index / (widthPDF * heightPDF)] + positionPDFY) * stringLength] - intensity[particleNumber]);

}

__global__ void lengthEstimation(float * PDF, int sizePDF, int threshold, float * existence)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    for(int i = 0; i < sizePDF; i++)
    {
        if(PDF[index * sizePDF + i] == 0)
        {
            sum++;
        }
    }

    if(sum > threshold)
    {
        existence[index] = 0;
    }
}

__global__ void procPDFIntensity2(float * PDF,float * deviationMatrix)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    PDF[index] = deviationMatrix[(int)PDF[index]];
}

__global__ void procPDFPosition(float * PDF, float * positionDeviation, float * existence, int sizePDF)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int particleNumber = index / sizePDF;

    if(existence[particleNumber] == 1)
        PDF[index] = PDF[index] * positionDeviation[sizePDF/2];
    if(existence[particleNumber] > 1)
        PDF[index] = PDF[index] * positionDeviation[index - particleNumber * sizePDF];
}


__global__ void findMaxPosition(float * PDF, float * maxPosition, float * maxValue, float * existance, int sizePDF)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float max = 0;
    float notOnePeak = 0;

    if( existance[index] > 0)
    {
        for(int i = 0; i < sizePDF; i++)
        {
            if(PDF[index * sizePDF + i] > max)
            {
                max = PDF[index * sizePDF + i];
                maxPosition[index] = i;
            }
        }
        for(int i = 0; i < sizePDF; i++)
        {
            if(PDF[index * sizePDF + i] == max)
            {
                notOnePeak++;
            }
        }
    }

    if(existance[index] == 0 || notOnePeak > 3)
    {
        max = 0;
    }

    maxValue[index] = max;
}

__global__ void predict(float * PDF, float * existence,float * maxPosition, float * particleX, float * particleY, float * velocityX, float * velocityY, float * intensity, float * inputImage, int widthPDF, int heightPDF, int sizePDF, int stringLength, int height, int searchWindowWidth, int searchWindowHeight, int searchWindowCenterX, int searchWindowCenterY)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(existence[index] == 1)
    {
        velocityX[index] = 0;
        velocityY[index] = 0;
    }
    else if(existence[index] > 1)
    {
        velocityX[index] += maxPosition[index] - int(maxPosition[index] / widthPDF)*(widthPDF) - widthPDF / 2;
        velocityY[index] += int(maxPosition[index] / widthPDF) - int(heightPDF / 2);
    }

    particleX[index] = particleX[index] + velocityX[index];
    particleY[index] = particleY[index] + velocityY[index];


    if(particleX[index] > stringLength - (stringLength - searchWindowWidth) / 2 + searchWindowCenterX ||
      particleX[index] < (stringLength - searchWindowWidth) / 2 + searchWindowCenterX
      || particleY[index] > height - (height - searchWindowHeight) / 2 + searchWindowCenterY
      || particleY[index] < (height - searchWindowHeight) / 2 + searchWindowCenterY)
        existence[index] = 0.;
}

__global__ void normalize(float * a, float sum)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    a[index] = a[index] / sum;
}

__global__ void countNonZero(float * a, float * b)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(a[index] > 1)
        b[index] = 1;
}

__global__ void removeStaticObjects(float * vX, float * vY, float * existence, float * maxValue)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(existence[index] > 1 && vX == 0 && vY == 0)
    {
        existence[index] = 0;
        maxValue[index] = 0;
    }
}

__global__ void threshold(float * inputMatrix, float * stateMatrix, float threshold)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(stateMatrix[index] >= 1 && inputMatrix[index] >= threshold)
        stateMatrix[index]++;
    else
        stateMatrix[index] = 0;
}

__global__ void particlesResamplingGPU(float * positionX, float * positionY, float * velocityX, float * velocityY, float * intensity, float * existance, float * randomX, float * randomY, float * resampleParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(existance[index] > 1 && resampleParticles[index] == 0)
    {
        ;
        //existance[index]++;
    }
    else if(resampleParticles[index] > 0)
    {
        positionX[index] = positionX[(int)resampleParticles[index]];// + (int)randomX[index]/120 - 2;
        positionY[index] = positionY[(int)resampleParticles[index]]; //+ (int)randomY[index]/60 - 2;
        velocityX[index] = velocityX[(int)resampleParticles[index]];
        velocityY[index] = velocityY[(int)resampleParticles[index]];
        intensity[index] = intensity[(int)resampleParticles[index]];
        //existance[index] = existance[(int)resampleParticles[index]];
        existance[index] = 1;
    }
}

__global__ void particlesResamplingGPU2(float * positionX, float * positionY, float * velocityX, float * velocityY, float * intensity, float * existance, float * image, float * randomX, float * randomY, float * resampleParticles, int stringLength)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(existance[index] >= 3)
    {
        positionX[index] += randomX[index] / 60 - 3;
        positionY[index] += randomY[index] / 60 - 3;
        //velocityX[index] += randomX[index] / 100 - 2;
        //velocityY[index] += randomY[index] / 100 - 2;
        intensity[index] += randomX[index] / 100 - 2;
    }
    else if(existance[index] < 1)
    {
        positionX[index] = randomX[index];
        positionY[index] = randomY[index];
        velocityX[index] = 0;
        velocityY[index] = 0;
        intensity[index] = (float) image[(int) positionX[index] + (int) velocityX[index] + ((int) positionY[index] + (int) velocityY[index]) * stringLength];
        existance[index] = 1;
    }
}

__global__ void targetExistnceProbability(float *weight, float *positionX, float *positionY, float *tempMatrix, int xMin, int xMax, int yMin, int yMax)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(positionX[index] > xMin && positionX[index] < xMax && positionY[index] > yMin && positionY[index] < yMax)
        tempMatrix[index] = weight[index];
    else
        tempMatrix[index] = 0;
}

__global__ void findLiveParticles(float *existence, float *tempMatrix)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(existence[index] > 1)
        tempMatrix[index] = 1;
    else
        tempMatrix[index] = 0;
}

__global__ void findLiveParticlesPos(float *existence, float *position, float *tempMatrix)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(existence[index] > 1)
        tempMatrix[index] = position[index];
    else
        tempMatrix[index] = 0;
}

__global__ void genMatrix(float *input)
{
    int tdx = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index/128 < 64)
    	input[index] = 1.;
    else
    	input[index] = 0.;
}

__global__ void rotate_1(float *input, float *output, int width)
{
    extern __shared__ float temp[];
    int tdx = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    temp[tdx] = input[index];
    __syncthreads();

    if(tdx < width / 2)
    {
    	output[index] = temp[width / 2  - 1 - tdx];
    }
    else
    {
    	output[index] = temp[3 * width / 2 - 1 - tdx];
    }
    __syncthreads();
}

__global__ void rotate_2(float *input, float *output, int width, int heigth)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tdx = threadIdx.x;

    if(index / width < heigth / 2)
    {
    	output[index] = input[(heigth / 2 - 1 - index / width) * width + threadIdx.x];
    }
    else
    {
    	output[index] = input[(3 * heigth / 2 - 1 - index / width) * width + threadIdx.x];
    }
}

__global__ void makeSAT(float *input, float *output, window procWindow, window etalonWindow)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int ydx = index / procWindow.width;
    int xdx = index - ydx * procWindow.width;

    if(xdx < procWindow.width - etalonWindow.width && ydx < procWindow.height - etalonWindow.height)
    {
    	output[index + etalonWindow.width / 2 + etalonWindow.height * procWindow.width / 2] =
    	  input[index] +
    	  input[index + etalonWindow.width + (etalonWindow.height) * procWindow.width] -
    	  input[index + etalonWindow.width] -
    	  input[index + (etalonWindow.height) * procWindow.width];
    }
    else
    {
    	output[index + etalonWindow.width / 2 + etalonWindow.height * procWindow.width / 2] = 0;
    }
    __syncthreads();
}

__global__ void copyMat(float *input, float *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    output[index] = input[index];
}

__global__ void setImageROI(float *input, float *output, window process_window, window etalon_window, int width, int heigth)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int ydx = index / process_window.width;
    int xdx = index - ydx * process_window.width;

    if(xdx > width && xdx < width + etalon_window.width / 3 && ydx > heigth && ydx < heigth + etalon_window.height / 3)
    {
    	output[index] = input[index];
    }
    else
    {
    	output[index] = 0;
    }
}

void particlesResamplingCPU(float * existence, float * maxValue, float * resempledTo, float thresholdValue, int particleNumber)
{
    int deadCount = 0;

    for(int i = 0; i < particleNumber; i++)
        resempledTo[i] = 0;


    int count = 0;

    for(int i = 0; i < particleNumber; i++)
    {
        if(existence[i] >= 3 && resempledTo[i] == 0 && maxValue[i] - thresholdValue / particleNumber > 0)
        {
            for(int j = deadCount; j < particleNumber; j++)
            {
                if(j >= particleNumber - 1)
                {
                    i = particleNumber;
                    break;
                }
                if(existence[j] < 1 && resempledTo[j] == 0)
                {
                    resempledTo[j] = i;
                    maxValue[i] -= thresholdValue / particleNumber;
                    count ++;
                }
                if(maxValue[i] < thresholdValue / particleNumber)
                {
                    deadCount = j;
                    j = particleNumber;
                }
            }
        }
    }
}

void distributionFunction(float *inputMatrix, int size, int stringLenght)
{
//--------------------------------- noralization ----------------------------------------
    float sum = 0;

    for(int i = 0; i < size; i++)
    {
        sum+=inputMatrix[i];
    }

    for(int i = 0; i < size; i++)
    {
        inputMatrix[i] = inputMatrix[i] / sum;
    }
//--------------------------------- redistribution weights ------------------------------
    sum = 0;

    for(int i = 0; i < size; i++)
    {
        inputMatrix[i] = sum + inputMatrix[i];
        sum += inputMatrix[i];
    }
}


