/*
 * bug.cu
 *
 * Bug classes functions
 *
 *  Created on: Mar 3, 2012
 *      Author: Maxym Zastavny
 */

#include <string>
#include <curand.h>
#include <cufft.h>

#include "bug.h"
#include "bug_kernels.cu"
#include "hdr/bug_struct.h"

using namespace std;

float votef(int, int, int);
void getBMR(float *, float *, window *, window *);

PF::PF(window capWindow, window procWindow, long int n, int width, int height,
  float varI, float varV, float varThreshold, int frameNum)
{
    capturedWindow = capWindow;
    _searchWindow = procWindow;
    particlesNumber = n;
    widthPDF = width;
    heightPDF = height;
    sizePDF = widthPDF * heightPDF;
    sigmaI = varI;
    sigmaV = varV;
    framesMatch = frameNum;
    thread = dim3(512, 1);
    blocks = dim3( n / thread.x, 1);
    blocksPDF = dim3( n * sizePDF / thread.x, 1);
    thresholdValue = varThreshold;

    // Allocate n floats on host
    hostDataX = (float *)calloc(n, sizeof(float));
    hostDataY = (float *)calloc(n, sizeof(float));
    hostImage = (float *)calloc(capWindow.width *
      capWindow.height, sizeof(float));
    hostMaxValue = (float *)calloc(n, sizeof(float));
    hostAliveParticles = (float *)calloc(n, sizeof(float));
    hostDataExistence = (float *)calloc(n, sizeof(float));;
    hostPositionDeviation = (float *)calloc(sizePDF, sizeof(float));

    // Allocate n floats on device
    cudaMalloc((void **)&devDataExistence, n * sizeof(float));
    cudaMalloc((void **)&devDataX, n * sizeof(float));
    cudaMalloc((void **)&devDataY, n * sizeof(float));
    cudaMalloc((void **)&devDataVelocityX, n * sizeof(float));
    cudaMalloc((void **)&devDataVelocityY, n * sizeof(float));
    cudaMalloc((void **)&devDataIntensity, n * sizeof(float));
    cudaMalloc((void **)&devDataWeight, n * sizeof(float));
    cudaMalloc((void **)&devDataPDF, n * sizePDF * sizeof(float));
    cudaMalloc((void **)&devImage, capWindow.width *
      capWindow.height * sizeof(float));

    cudaMalloc((void **)&devIntensityDeviation, 256 * sizeof(float));
    cudaMalloc((void **)&devPositionDeviation, sizePDF * sizeof(float));
    cudaMalloc((void **)&devMaxPosition, n * sizeof(float));
    cudaMalloc((void **)&devMaxValue, n * sizeof(float));
    cudaMalloc((void **)&devResamplingX, n * sizeof(float));
    cudaMalloc((void **)&devResamplingY, n * sizeof(float));
    cudaMalloc((void **)&devlengthEstimation, n * sizeof(float));

    cudaMalloc((void **)&devTempMatrix, n * sizeof(float));
    cudaMalloc((void **)&devDontMove, n * sizeof(float));
    cudaMalloc((void**)&devParticleDistribution, sizePDF * sizeof(float));

    // Create pseudo-random number generator and  Set seed
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    hostTemp = (float *)calloc(sizePDF, sizeof(float));
}

PF::~PF()
{
    // Cleanup device
    curandDestroyGenerator(gen);
    cudaFree(devDataExistence);
    cudaFree(devDataX);
    cudaFree(devDataY);
    cudaFree(devDataVelocityX);
    cudaFree(devDataVelocityY);
    cudaFree(devDataIntensity);
    cudaFree(devDataWeight);
    cudaFree(devDataPDF);
    cudaFree(devImage);

    cudaFree(devIntensityDeviation);
    cudaFree(devPositionDeviation);
    cudaFree(devMaxPosition);
    cudaFree(devMaxValue);
    cudaFree(devResamplingX);
    cudaFree(devResamplingY);
    cudaFree(devlengthEstimation);
    cudaFree(devTempMatrix);
    cudaFree(devDontMove);
    cudaFree(devParticleDistribution);

    // Cleanup host
    free(hostDataX);
    free(hostDataY);
    free(hostImage);
    free(hostMaxValue);
    free(hostAliveParticles);
    free(hostDataExistence);
    free(hostPositionDeviation);
}

void PF::initParticles()
{

//--------------------------------- init X, Y values ------------------------------------
    // Generate n floats on device
    curandGenerateUniform(gen, devDataX, particlesNumber);
    cublasSscal (particlesNumber, _searchWindow.width, devDataX, 1);
    cuAdd<<< blocks , thread >>> (devDataX, (capturedWindow.width -
      _searchWindow.width)/2 + _searchWindow.centerX);

    curandGenerateUniform(gen, devDataY, particlesNumber);
    cublasSscal (particlesNumber, _searchWindow.height, devDataY, 1);
    cuAdd<<< blocks , thread >>> (devDataY, (capturedWindow.height -
      _searchWindow.height)/2 + _searchWindow.centerY);

//--------------------------------- init speed values -----------------------------------
    // fill the matrix with zeros
    setTo<<< blocks , thread >>> (devDataVelocityX, 0.);
    setTo<<< blocks , thread >>> (devDataVelocityY, 0.);

//--------------------------------- init Intensity values -------------------------------
    initIntensityValues<<< blocks , thread >>> (devDataIntensity,
      devDataX, devDataY, devDataVelocityX, devDataVelocityY,
      devImage, capturedWindow.width);

//--------------------------------- init PDF values -------------------------------------
    // fill the matrix with ones
    setTo<<< blocksPDF , thread >>> (devDataPDF, 1.);

//--------------------------------- init existance --------------------------------------
    // fill the matrix with ones
    setTo<<< blocks, thread >>> (devDataExistence, 1.);

//----------------------- init Intensity and Position deviation Matrix ------------------
    initIntensityDeviationMatrix<<< 1, 256 >>> (devIntensityDeviation, sigmaI);
    initPositionDeviationMatrix<<< 1, (widthPDF * heightPDF) >>>
      (devPositionDeviation, widthPDF, sigmaV);
}

void PF::getImage(float *image)
{
    cudaMemcpy(devImage, image, sizeof(float) * capturedWindow.width *
      capturedWindow.height, cudaMemcpyHostToDevice);
}

void PF::predictParticles()
{
    //setTo<<<blocks, thread>>>(devMaxPosition, 100.);
    predict<<< blocks, thread >>> (devDataPDF, devDataExistence, devMaxPosition,
      devDataX, devDataY, devDataVelocityX, devDataVelocityY, devDataIntensity,
      devImage, widthPDF, heightPDF, sizePDF, capturedWindow.width,
      capturedWindow.height, _searchWindow.width, _searchWindow.height,
      _searchWindow.centerX, _searchWindow.centerY);
}

void PF::roughingParticles()
{
    particlesResamplingGPU2<<< blocks , thread >>> (devDataX, devDataY,
      devDataVelocityX, devDataVelocityY, devDataIntensity, devDataExistence,
      devImage, devResamplingX, devResamplingY, devTempMatrix, capturedWindow.width);
}

void PF::getParticlesWeights()
{
//========================= Weight Update ==================================================
    setTo<<< blocksPDF , thread >>> (devDataPDF, 1.);
//------------------------- weight by intensity update -------------------------------------
    // calculation of the difference in the intensity of the particle
    //and the intensity of each pixel by the local search window
    procPDFIntensity<<< blocksPDF , thread >>> (devDataPDF,
      devDataIntensity,devIntensityDeviation, devImage, devDataX,
      devDataY, capturedWindow.width, widthPDF, heightPDF);

    lengthEstimation<<< blocks, thread >>> (devDataPDF, sizePDF, 5,
      devDataExistence);

    procPDFIntensity2<<< blocksPDF , thread >>> (devDataPDF,
      devIntensityDeviation);
//------------------------- weight by position update --------------------------------------
    procPDFPosition<<< blocksPDF , thread >>> (devDataPDF,
      devPositionDeviation, devDataExistence, sizePDF);
//========================= getting weights of the particles ===============================
    findMaxPosition<<< blocks, thread >>> (devDataPDF, devMaxPosition,
      devMaxValue, devDataExistence, sizePDF);
}

void PF::normalizeParticlesWeights()
{
//========================= Normalizing of weight coefficients =============================
    // finding sum of weight coefficients
    float sum = cublasSasum(particlesNumber, devMaxValue, 1);
    // normalization
    normalize<<< blocks , thread >>> (devMaxValue, sum);

}

void PF::resempleParticles()
{

//========================= Resampling =====================================================
//------------------------- choicing of alive particles by weight --------------------------
    threshold<<< blocks , thread >>> (devMaxValue, devDataExistence,
      thresholdValue / particlesNumber );

//------------------------- Generating random x and y for new particles --------------------
    curandGenerateUniform(gen, devResamplingX, particlesNumber);
    cublasSscal (particlesNumber, _searchWindow.width, devResamplingX, 1);
    cuAdd<<< blocks , thread >>> (devResamplingX, (capturedWindow.width -
      _searchWindow.width)/2 + _searchWindow.centerX);

    curandGenerateUniform(gen, devResamplingY, particlesNumber);
    cublasSscal (particlesNumber, _searchWindow.height, devResamplingY, 1);
    cuAdd<<< blocks , thread >>> (devResamplingY, (capturedWindow.height -
      _searchWindow.height)/2 + _searchWindow.centerY);
//------------------------- resampling dead particles to alive particles ---------------------

    cudaMemcpy(hostMaxValue, devMaxValue, particlesNumber *
      sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostDataExistence, devDataExistence, particlesNumber *
      sizeof(float), cudaMemcpyDeviceToHost);

    particlesResamplingCPU(hostDataExistence, hostMaxValue,
      hostAliveParticles, thresholdValue, particlesNumber);

    // copy resapled particles data to GPU
    //setTo<<< blocks, thread >>> (devTempMatrix, 0.);
    cudaMemcpy(devTempMatrix, hostAliveParticles, particlesNumber *
      sizeof(float), cudaMemcpyHostToDevice);

    //particlesResamplingGPU
    particlesResamplingGPU<<< blocks , thread >>> (devDataX, devDataY,
      devDataVelocityX, devDataVelocityY, devDataIntensity, devDataExistence,
      devResamplingX, devResamplingY, devTempMatrix);

}

void PF::getPFOutput(float *dataX, float *dataY, float *dataExistence)
{
	cudaMemcpy(dataX, devDataX, particlesNumber *
	  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dataY, devDataY, particlesNumber *
      sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dataExistence, devDataExistence, particlesNumber *
      sizeof(float), cudaMemcpyDeviceToHost);
}

CLS_filter::CLS_filter(window captWindow, window procWindow, int T)
{
    period = T;

    capturedWindow = captWindow;
    processWindow = procWindow;

    threads = dim3(512, 1);
    blocks = dim3( processWindow.width * processWindow.height / threads.x, 1);

    cufftPlan2d(&planFFT, processWindow.height, procWindow.width, CUFFT_R2C);
    cufftPlan2d(&planIFFT, procWindow.height, procWindow.width, CUFFT_C2R);

    // Allocate memory on device for direct and inverse Fast Fourier transform
    cudaMalloc((void**)&presentImageFFT, sizeof(cufftComplex)*procWindow.width
      * procWindow.height);
    cudaMalloc((void**)&lastImageFFT, sizeof(cufftComplex)*procWindow.width
      * procWindow.height);
    cudaMalloc((void**)&fxFFT, sizeof(cufftComplex)*procWindow.width
      * procWindow.height);
    cudaMalloc((void**)&outputIFFT, sizeof(cufftComplex)*procWindow.width
      * procWindow.height);

    // Allocate memory on device for images
    cudaMalloc((void **)&devImage, captWindow.width * captWindow.height
      * sizeof(float));
    cudaMalloc((void **)&devPresentImage, procWindow.width * procWindow.height
      * sizeof(float));
    cudaMalloc((void **)&devLastImage, procWindow.width * procWindow.height
      * sizeof(float));
    cudaMalloc((void **)&devBackground, procWindow.width * procWindow.height
      * sizeof(float));
    cudaMalloc((void **)&devProcessedImage, procWindow.width * procWindow.height
      * sizeof(float));
    cudaMalloc((void **)&devFX, procWindow.width * procWindow.height
      * sizeof(float));
    cudaMalloc((void **)&devCorrelation, procWindow.width * procWindow.height
      * sizeof(float));

    // Allocate memory on device for buffers
    cudaMalloc((void **)&devBackgroundBuffer, T * procWindow.width *
      procWindow.height * sizeof(float));
    hostShiftsBuffer = (float *)calloc(capturedWindow.width *
      capturedWindow.height, sizeof(float));
}

CLS_filter::~CLS_filter()
{
    cufftDestroy(planFFT);
    cufftDestroy(planIFFT);

    cudaFree(presentImageFFT);
    cudaFree(lastImageFFT);
    cudaFree(fxFFT);
    cudaFree(outputIFFT);

    cudaFree(devImage);
    cudaFree(devPresentImage);
    cudaFree(devLastImage);
    cudaFree(devBackground);
    cudaFree(devProcessedImage);
    cudaFree(devFX);
    cudaFree(devCorrelation);

    cudaFree(devBackgroundBuffer);
    free(hostShiftsBuffer);
}

void CLS_filter::genFx(float xf, float yf)
{
    devGenFx <<<blocks, threads>>> (devFX, processWindow.height, xf, yf);
    cufftExecR2C(planFFT, devFX, fxFFT);
}

void CLS_filter::getImage(float *image, int n)
{
    frameNumber = n;

    exchanngeFrames <<<blocks, threads>>> (devLastImage, devPresentImage);

    cudaMemcpy(devImage, image, sizeof(float) * capturedWindow.width *
      capturedWindow.height, cudaMemcpyHostToDevice);
    getWindow <<<blocks, threads>>> (devPresentImage, devImage,
      processWindow, capturedWindow);
}

void CLS_filter::getBackground()
{
    // Fast Foutier transform
    cufftExecR2C(planFFT, devLastImage, presentImageFFT);

    // Q and fx multiplication
    matrixMul<<<blocks, threads>>> (presentImageFFT, fxFFT, outputIFFT);

    // Invrerse Fourier transform
    cufftExecC2R(planIFFT, outputIFFT, devBackground);

    divide <<<blocks, threads>>> (devBackground,
      processWindow.width * processWindow.height);
}

void CLS_filter::getShift()
{
    cufftExecR2C(planFFT, devLastImage, lastImageFFT);

    // last  and present image multiplication
    matrixMul<<<blocks, threads>>> (presentImageFFT, lastImageFFT, outputIFFT);

    // Inverse transform the signal in place
    cufftExecC2R(planIFFT, outputIFFT, devCorrelation);
}

void CLS_filter::getProcImage()
{
    int max = cublasIsamax(processWindow.width * processWindow.height,
      devCorrelation, 1);

    shift etalonWindow;

    etalonWindow.y = (max - 1) / processWindow.width;
    etalonWindow.x = max - 1 - etalonWindow.y * processWindow.width;

    if(etalonWindow.x > processWindow.width / 2)
        etalonWindow.x = etalonWindow.x - processWindow.width;
    if(etalonWindow.y > processWindow.height / 2)
        etalonWindow.y = etalonWindow.y - processWindow.height;

    //subtract <<<blocks, threads>>> (devPresentImage, devLastImage, devProcessedImage, 1);
    subtractWithShift <<<blocks, threads>>> (devPresentImage, devBackground,
      devProcessedImage, 10, etalonWindow, processWindow.width);

    //normalize(devProcessedImage, processWindow, 255);
}

void CLS_filter::getOutput(float *image)
{
    cudaMemcpy(image, devProcessedImage, sizeof(float) * processWindow.width *
      processWindow.height, cudaMemcpyDeviceToHost);
}

Tracking::Tracking(window captWindow, window procWindow,
  window etWindow, int size, int thrsh)
{
	targetProbability = 1.;
	threshold = thrsh;
    gaussFilterSize = size;
    capturedWindow = captWindow;
    processWindow = procWindow;
    etalonWindow = etWindow;

    threads = dim3(512, 1);
    blocks = dim3( processWindow.width * processWindow.height / threads.x, 1);

    cufftPlan2d(&planFFT, processWindow.height, processWindow.width, CUFFT_R2C);
    cufftPlan2d(&planIFFT, processWindow.height, processWindow.width, CUFFT_C2R);

    // Allocate memory on device for direct and inverse Fast Fourier transform
    cudaMalloc((void**)&targetImageFFT, sizeof(cufftComplex) *
      processWindow.width * processWindow.height);
    cudaMalloc((void**)&backgroundImageFFT, sizeof(cufftComplex) *
      processWindow.width * processWindow.height);
    cudaMalloc((void**)&outputIFFT, sizeof(cufftComplex) *
      processWindow.width * processWindow.height);

    // Allocate memory on device for images
    cudaMalloc((void **)&devImage, capturedWindow.width
      * capturedWindow.height * sizeof(float));
    cudaMalloc((void **)&devProcessedImage, processWindow.width
      * processWindow.height * sizeof(float));
    cudaMalloc((void **)&devEtalonImage, processWindow.width
      * processWindow.height * sizeof(float));
    cudaMalloc((void **)&devCorrelation, processWindow.width
      * processWindow.height * sizeof(float));
    cudaMalloc((void **)&devTempImage_1, processWindow.width
      * processWindow.height * sizeof(float));
    cudaMalloc((void **)&devTempImage_2, processWindow.width
      * processWindow.height * sizeof(float));
    cudaMalloc((void **)&gaussFilterMatrix, gaussFilterSize
      * gaussFilterSize * sizeof(float));
}

void Tracking::genGaussFilterMatrix()
{
    float *matrix;
    float sigma;

    matrix = (float *)calloc(gaussFilterSize * gaussFilterSize, sizeof(float));

    sigma = 0.3 * (gaussFilterSize / 2 - 1) + 0.8;

    for(int j = 0; j < gaussFilterSize; j++)
    {
        for(int i = 0; i < gaussFilterSize; i++)
        {
            matrix[i + j * gaussFilterSize] = exp(- ((i - gaussFilterSize / 2)*
              (i - gaussFilterSize / 2) + (j - gaussFilterSize / 2) *
              (j - gaussFilterSize / 2)) / (2 * sigma * sigma)) /
              (2 * 3.14159 * sigma * sigma);
        }
    }

    cudaMemcpy(gaussFilterMatrix, matrix, sizeof(float) * gaussFilterSize *
      gaussFilterSize, cudaMemcpyHostToDevice);
}

void Tracking::getImage(float *image, window procWindow, int n)
{
    processWindow = procWindow;
    frameNumber = n;

    cudaMemcpy(devImage, image, sizeof(float) * capturedWindow.width *
      capturedWindow.height, cudaMemcpyHostToDevice);

    getWindow <<<blocks, threads>>> (devProcessedImage, devImage,
      processWindow, capturedWindow);
}

void Tracking::preprocess(int mode)
{
    float alfa = 0.75;

    if(mode == 1)
	{
		alfa = 0.7;
	}

    gaussFiltration<<< 1, threads >>>(devProcessedImage, devTempImage_1,
      processWindow.width, processWindow.height, alfa);
    sobelOperator <<< blocks, threads >>> (devTempImage_1,
      devProcessedImage, processWindow);



    if(mode == 1)
	{
    	thresholdImage <<< blocks, threads >>> (devProcessedImage, 20, 2);
    	normalize(devProcessedImage, processWindow, 255);
    	//limitation <<<blocks, threads>>> (devProcessedImage, 255);
    	thresholdImage <<< blocks, threads >>> (devProcessedImage, threshold, 1);

	}
    if(mode == 2)
	{
    	normalize(devProcessedImage, processWindow, 255);
    	thresholdImage <<< blocks, threads >>> (devProcessedImage, threshold, mode);
	}
}

window Tracking::getOutput(float *image, int mode)
{
	if(mode == 0)
	{
		divide <<<blocks, threads>>> (devProcessedImage, 256.);

		cudaMemcpy(image, devProcessedImage, sizeof(float) * processWindow.width
          * processWindow.height, cudaMemcpyDeviceToHost);

		divide <<<blocks, threads>>> (devProcessedImage, 1/256.);
	}
	if(mode == 1)
	{
		divide <<<blocks, threads>>> (devEtalonImage, 256.);

		cudaMemcpy(image, devEtalonImage, sizeof(float) * processWindow.width
          * processWindow.height, cudaMemcpyDeviceToHost);

		divide <<<blocks, threads>>> (devEtalonImage, 1/256.);
	}
	if(mode == 2)
	{
		cudaMemcpy(image, devCorrelation, sizeof(float) * processWindow.width
          * processWindow.height, cudaMemcpyDeviceToHost);
	}

    return processWindow;
}

void Tracking::track(int x, int y, bool trackingType)
{
    cufftExecR2C(planFFT, devProcessedImage, backgroundImageFFT);
    cufftExecR2C(planFFT, devEtalonImage, targetImageFFT);

    // last  and present image multiplication
    matrixMul<<<blocks, threads>>> (targetImageFFT,
      backgroundImageFFT, outputIFFT);

    // Inverse transform the signal in place
    cufftExecC2R(planIFFT, outputIFFT, devCorrelation);

    dim3 dimBlock(16, 16, 1); /// # define BLOCK_DIM 16
    dim3 dimGrid(32, 32, 1);

    // taking square of the elements of image
    matrixSQR<<< blocks, threads >>>(devProcessedImage, devTempImage_2);
    // clear temp image
    setTo <<< blocks, threads >>> (devTempImage_1, 0.);

    // Getting integral image
    // Getting sum in rows
    integralScan <<< processWindow.height, processWindow.width / 2,
      processWindow.width * sizeof(float) >>> (devTempImage_2,
      devTempImage_1, processWindow.width);
    // image transpose
	integralTranspose <<< dimGrid, dimBlock >>> (devTempImage_1,
	  devTempImage_2, processWindow.width, processWindow.height);
	// Getting sum in rows
	integralScan <<< processWindow.height, processWindow.width / 2,
	  processWindow.width * sizeof(float) >>> (devTempImage_2,
	  devTempImage_1, processWindow.width);
	// image transpose
	integralTranspose <<< dimGrid, dimBlock >>> (devTempImage_1,
	  devTempImage_2, processWindow.width, processWindow.height);

	// clear temp image
	setTo <<< blocks, threads >>> (devTempImage_1, 0.);
	// Getting normalized coefficients for normalization of correlation function

	makeSAT<<<blocks, threads>>>(devTempImage_2, devTempImage_1,
	  processWindow, etalonWindow);

	// centered correlation function by rows
    rotate_1 <<< processWindow.height, processWindow.width, processWindow.width
      * sizeof(float) >>> (devCorrelation, devTempImage_2, processWindow.width);
    // clear correlation function image
	setTo <<< blocks, threads >>> (devCorrelation, 0.);
	// centered correlation function by columns
    rotate_2 <<< processWindow.height, processWindow.width>>> (devTempImage_2,
      devCorrelation, processWindow.width, processWindow.height);

    // getting index of max element in normalized coefficients matrix
    int maxIndex = cublasIsamax(processWindow.width * processWindow.height,
      devTempImage_1, 1);

    // getting value of max element in normalized coefficients matrix
    float *maxValue;
    maxValue = (float *)calloc(1, sizeof(float));
    cudaMemcpy(maxValue, devTempImage_1 + maxIndex - 1, sizeof(float),
      cudaMemcpyDeviceToHost);

    // divide preprocessed image on normalized coefficients matrix
    divideMat<<< blocks, threads >>>(devCorrelation, devTempImage_1,
      *maxValue);

    // taking square of the elements of image
    matrixSQR<<< blocks, threads >>>(devEtalonImage, devTempImage_2);

    // getting sum of element of etalon image
    float sum = cublasSasum(processWindow.width * processWindow.height,
      devTempImage_2, 1);

    // divide preprocessed image on sum of etalon image elements
    divide<<< blocks, threads >>>(devCorrelation, sqrt(sum));
    divide<<< blocks, threads >>>(devCorrelation, processWindow.width *
      processWindow.height);

    // getting max value of correlation function
    maxIndex = cublasIsamax(processWindow.width * processWindow.height,
      devCorrelation, 1);
    cudaMemcpy(maxValue, devCorrelation + maxIndex - 1, sizeof(float),
      cudaMemcpyDeviceToHost);

    // getting max correlation function value and showing it
    if(*maxValue > 0. && *maxValue < 1.)
    {
			targetProbability = *maxValue;

		// getting etalon position in preprocessed image
		etalonWindow.centerY = (maxIndex - 1) / processWindow.width -
		  processWindow.width/ 2 + 1;
		etalonWindow.centerX = maxIndex - 1 - (etalonWindow.centerY +
		  processWindow.width/ 2) * processWindow.width + processWindow.width / 2 + 1;

		// strobe moving limitation
		if(abs(processWindow.centerX + etalonWindow.centerX) <
		   (capturedWindow.width - processWindow.width) / 2
		   && abs(processWindow.centerY + etalonWindow.centerY) <
		   (capturedWindow.height - processWindow.height) / 2
		   && *maxValue > 0.5)
		 {
			 processWindow.centerX += etalonWindow.centerX;
			 processWindow.centerY += etalonWindow.centerY;
		 }

	getBMR(devEtalonImage, devTempImage_1, &processWindow, &etalonWindow);

	//printf("%d \t %d \n", etalonWindow.width, etalonWindow.height);
    }
    else
    {
    	setTo <<< blocks, threads >>> (devTempImage_1, 0.);
    	setTo <<< blocks, threads >>> (devTempImage_2, 0.);
    	setTo <<< blocks, threads >>> (devCorrelation, 0.);
    }
}

void Tracking::getEtalonImage()
{
    etalonWindow.centerX = 0;
    etalonWindow.centerY = 0;
    updateEtalonWindow2 <<<blocks, threads>>> (devProcessedImage,
      devEtalonImage, processWindow, etalonWindow, 1.);
}

void Tracking::updateEtalonImage()
{
    //float alfa = 0.03;
	if(abs(processWindow.centerX + etalonWindow.centerX) <
	  (capturedWindow.width - processWindow.width) / 2
	  && abs(processWindow.centerY + etalonWindow.centerY) <
	  (capturedWindow.height - processWindow.height) / 2
	  && targetProbability > 0.5 && targetProbability < 1.
	  && abs(etalonWindow.centerX) < etalonWindow.width / 2
	  && abs(etalonWindow.centerY) < etalonWindow.height / 2)
	{
		// 0.05
		float alfa = 0.05 * targetProbability;
		updateEtalonWindow <<<blocks, threads>>> (devProcessedImage,
		  devEtalonImage, processWindow, etalonWindow, alfa);
	}
}

int Tracking::changeTrackingMode()
{
    int nonZeroNumber = (int)cublasSasum(processWindow.width * processWindow.height,
      devEtalonImage, 1) / 128;

    return nonZeroNumber;
}

void getBMR(float *image, float *temp_image, window *process_window, window *etalon_window)
{
    // масив средних значений областей
    int mean[9];
    int window_size = etalon_window->width * etalon_window->height;

    // коэфициенты учитывающие разные растояния между центрами диагональных и соседних областей
    float ws = 0.375;
    float wd = 0.250;

    // изменение позиции верхней левой и нижней правой точек эталона по х и у
    int deltaxtl, deltaytl, deltaxbr, deltaybr;

    // установка среждних значений областей
    for(int j = 0; j < 3; j++)
    {
        for(int i = 0; i < 3; i++)
        {
            setImageROI<<< process_window->width, process_window->height >>>
              (image, temp_image, *process_window, *etalon_window,
              (process_window->width - etalon_window->width) / 2 +
              i * etalon_window->width / 3,
              (process_window->height - etalon_window->height) / 2 +
              j * etalon_window->height / 3);

            mean[i+j*3] = cublasSasum(process_window->width *
              process_window->height, temp_image, 1) / (window_size / 9);
        }
    }
    /*
    printf("%d \t %d \t %d \n", mean[0], mean[1], mean[2]);
    printf("%d \t %d \t %d \n", mean[3], mean[4], mean[5]);
    printf("%d \t %d \t %d \n", mean[6], mean[7], mean[8]);
*/
    // определение оценок от каждой области
    float v1v = votef(mean[0], mean[1], mean[2]);
    float v1h = votef(mean[0], mean[3], mean[6]);
    float v1d = votef(mean[0], mean[4], mean[8]);
    float v2 = votef(mean[1], mean[4], mean[7]);
    float v3v = votef(mean[2], mean[1], mean[0]);
    float v3h = votef(mean[2], mean[5], mean[8]);
    float v3d = votef(mean[2], mean[4], mean[6]);
    float v4 = votef(mean[3], mean[4], mean[5]);
    float v6 = votef(mean[5], mean[4], mean[3]);
    float v7v = votef(mean[6], mean[7], mean[8]);
    float v7h = votef(mean[6], mean[3], mean[0]);
    float v7d = votef(mean[6], mean[4], mean[2]);
    float v8 = votef(mean[7], mean[4], mean[1]);
    float v9v = votef(mean[8], mean[7], mean[6]);
    float v9h = votef(mean[8], mean[5], mean[2]);
    float v9d = votef(mean[8], mean[4], mean[0]);

    // нахождение изменения позиции верхней левой и нижней правой точек эталона по х и у
    deltaxtl = round(wd*v1d + v4 + wd*v7d + ws*v1h + ws*v7h);
    deltaytl = round(wd*v1d + v2 + wd*v3d + ws*v1v + ws*v3v);
    deltaxbr = round(wd*v3d + v6 + wd*v9d + ws*v3h + ws*v9h);
    deltaybr = round(wd*v7d + v8 + wd*v9d + ws*v7v + ws*v9v);

    // установка максимально и минамально допустимого размера окна эталона по х
    if(etalon_window->width + deltaxbr + deltaxtl > 16 &&
       etalon_window->width + deltaxbr + deltaxtl <= process_window->width / 2)
    {
        // смещение центра окна эталона по х
        etalon_window->centerX += -(deltaxtl - deltaxbr);

        // изменение ширины окна эталона по х
        //etalon_window->width += (deltaxbr + deltaxtl)/2;
    }

    // установка максимально и минамально допустимого размера окна эталона по у
    if(etalon_window->height + deltaybr + deltaytl > 16 &&
       etalon_window->height + deltaybr + deltaytl <= process_window->height / 2)
    {
        // смещение центра окна эталона по у
        etalon_window->centerY += -(deltaytl - deltaybr);

        // изменение ширины окна эталона по у
        //etalon_window->height += (deltaybr + deltaytl)/2;
    }


   // printf("%d \t %d \t %d \t %d \n ", deltaxtl, deltaxbr, deltaytl, deltaytl);
 }

float votef(int element, int central, int opposite)
{
    float vote;

    // оценка для увеличения размеров окна эталона
    //if((element >= 0.85*central || element >=1.40*opposite) && element != 0)
    if((element >= 0.6*central || element >=1.40*opposite) && element != 0)
        vote = 0.5;
    // оценка для уменьшения размеров окна эталона
    else if(1.4*element < central || 0.9*element < opposite)
    //else if(1.3*element < central || 0.4*element < opposite)
    	vote = -0.5;
    else
        vote = 0;

    return vote;
}
