/*
 * bug.h
 *
 * Bug classes
 *
 *  Created on: Mar 3, 2012
 *      Author: Maxym Zastavny
 */

#include "hdr/bug_struct.h"

#ifndef TRACK_H_
#define TRACK_H_

class PF
{
	public:
        PF (window, window, long int, int, int, float, float, float, int);
        ~PF ();
        void initParticles();
        void getImage(float *);
        void predictParticles();
        void roughingParticles();
        void getParticlesWeights();
        void normalizeParticlesWeights();
        void resempleParticles();
        void makedecision();
        void getPFOutput(float *, float *, float *);
	private:
        window _searchWindow, capturedWindow;
        long int particlesNumber;
        size_t sizePDF, widthPDF, heightPDF;
        float sigmaI, sigmaV, thresholdValue;
        int framesMatch;
        dim3 thread, blocks, blocksPDF;
        curandGenerator_t gen;

        float *hostDataX, *hostDataY, *hostImage, *hostMaxValue,
          *hostAliveParticles, *hostDataExistence, *hostPositionDeviation;
        float *devDataExistence, *devDataX, *devDataY, *devDataVelocityX,
          *devDataVelocityY, *devDataIntensity, *devDataWeight,
          *devDataPDF, *devImage;
        float *devIntensityDeviation, *devPositionDeviation, *devMaxPosition,
          *devMaxValue, *devResamplingX, *devResamplingY, *devlengthEstimation;
        float *devTempMatrix, *devDontMove, *devParticleDistribution;
        float *hostTemp;
};

class CLS_filter
{
    public:
        CLS_filter(window, window, int);
        ~CLS_filter();
        void genFx(float, float);
        void getImage(float *, int);
        void getBackground();
        void getShift();
        void getProcImage();
        void getOutput(float *);
    private:
        int frameNumber, period;
        window capturedWindow, processWindow;
        dim3 threads, blocks;
        cufftHandle planFFT,planIFFT;
        cufftComplex *presentImageFFT, *lastImageFFT, *fxFFT;
        cufftComplex *outputIFFT;

        float *devImage, *devPresentImage, *devLastImage, *devBackground,
          *devProcessedImage, *devFX, *devCorrelation;
        float *devBackgroundBuffer, *hostShiftsBuffer;
};

class Tracking
{
	public:
    	float targetProbability;
    	int target_size;

		Tracking(window, window, window, int, int);
        //~Tracking();
        void genGaussFilterMatrix();
        void getImage(float *, window, int);
        void getEtalonImage();
        void preprocess(int);
        void track(int, int, bool);
        void updateEtalonImage();
        int changeTrackingMode();
        //void strobeUpdate();
        window getOutput(float *, int);
	private:
        int frameNumber;
        int gaussFilterSize, threshold;
        float *gaussFilterMatrix;
        window capturedWindow, processWindow, etalonWindow;
        dim3 threads, blocks;
        cufftHandle planFFT,planIFFT;
        cufftComplex *targetImageFFT, *backgroundImageFFT;
        cufftComplex *outputIFFT;

        float *devImage, *devProcessedImage, *devTempImage_1,
          *devTempImage_2, *devEtalonImage, *devCorrelation;
};


#endif /* TRACK_H_ */
