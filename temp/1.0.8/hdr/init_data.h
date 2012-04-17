/*
 * init_data.h
 *
 * Initialization data file
 *
 *  Created on: Mar 3, 2012
 *      Author: Maxym Zastavny
 */

#ifndef FILTER_CONSTANTS_H_INCLUDED
#define FILTER_CONSTANTS_H_INCLUDED

//captured window size
#define SCREEN_WIDTH 720
#define SCREEN_HEIGHT 576

// search window parameters
#define SEARCH_WINDOW_WIDTH 256
#define SEARCH_WINDOW_HEIGHT 256
#define SEARCH_WINDOW_CENTER_X 0
#define SEARCH_WINDOW_CENTER_Y 0

// track window parameters
#define TRACK_WINDOW_WIDTH 128
#define TRACK_WINDOW_HEIGHT 128
#define TRACK_WINDOW_CENTER_X 0
#define TRACK_WINDOW_CENTER_Y 0

// etalon window parameters
#define ETALON_WINDOW_WIDTH 64
#define ETALON_WINDOW_HEIGHT 64
#define ETALON_WINDOW_CENTER_X 0
#define ETALON_WINDOW_CENTER_Y 0

//---------------- PF constants ------------------------

// particles nuber and size of PDF matrix
#define PARTICLE_NUMBER 128 * 128
#define PDF_HEIGHT 15
#define PDF_WIDTH PDF_HEIGHT
#define PDF_SIZE PDF_WIDTH * PDF_HEIGHT

// target and detector model parameters
#define SIGMA_I 10
#define SIGMA_V 0.5
#define THRESHOLD_VALUE 0.5
#define FRAME_MATCH 0

//---------------- CLS constants ------------------------

#define PERIOD 3

//---------------- Tracking constants ------------------------

#define GAUSSS_FILTER_SIZE 7.0
#define FILTRATION_THRESHOLD 50
#define CORRELATION_THRESHOLD 0.9

#endif
