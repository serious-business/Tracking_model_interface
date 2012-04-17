/*
 * bug.cpp
 *
 * main project file
 *
 *  Created on: Mar 3, 2012
 *      Author: Maxym Zastavny
 */

#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand.h>
#include <cublas.h>
#include <cufft.h>

// *****************************

#include <time.h>
#include <errno.h>
#include <math.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/ipc.h>
#include <unistd.h>

#define __KEY_VALUE 0xffff1111
#define __WINDOW_WIDTH 720.0
#define __WINDOW_HEIGHT 576.0
#define __SHARED_SEGMENT_SIZE (__WINDOW_WIDTH*__WINDOW_HEIGHT*3)

// *****************************

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <sys/time.h>

#include <bug.h>
#include "hdr/init_data.h"
#include "hdr/bug_struct.h"

using namespace std;

// window in which we are tracking target
window track_window;

// frame number variable
bool first_frame;

void mouse_callback( int, int, int, int, void *);
void init_windows_params(window *, window *, window *, window *);

void correlation_tracking(Tracking, float*, bool*, bool*, bool);
void contrast_tracking(Tracking, float*, bool*, bool*, bool);

int main(int argc, char *argv[])
{
    timeval tim;

    double start_t, get_im_t, preproc_t, track_t, etalon_update_t, output_t;

    int output_mode = 0;
    int tracking_type = 0;

    enum tracking_mode { AUTO_TRACKING, CORRELATION_TRACKING, CONTRAST_TRACKING };
    bool debugg_mode = false;
    bool auto_tracking = false;
    first_frame = false;

    int target_size = 0;
	long int frame_count = 0;

    const char* filename = argc == 2 ? argv[1] :
      "/home/Max/prj/videos/helicopter.mpg";

    // window captured from the camera(video file)
    window captured_window;

    // window in which we are searching target
    window search_window;

    // etalon target image
    window etalon_window;

    // Initialize connection with GUI interface *********************************
/*
	sem_t *data_ready, *data_accepted, *process_termination;
	char *shared_mem, *frame_buffer;
	int shared_mem_id, sval;
	#ifdef WRITE_VIDEO_FRAMES
	int frame_counter = 0;
	#endif
	key_t key_value = __KEY_VALUE;

	sem_unlink("data_ready_imit_to_interface");
	sem_unlink("data_accepted_imit_to_interface");
	sem_unlink("prc_term_imit_to_interface");

	data_ready = sem_open("data_ready_imit_to_interface", O_CREAT | O_EXCL,
						  S_IRWXU | S_IRWXO, 0);
	if (data_ready == SEM_FAILED)
	{
		cout<<"Unable to open data_ready_imit_to_interface semaphore."<<errno<<endl;
		exit(EXIT_FAILURE);
	}

	data_accepted = sem_open("data_accepted_imit_to_interface", O_CREAT | O_EXCL,
			S_IRWXU | S_IRWXO, 0);
	if (data_ready == SEM_FAILED)
	{
		cout<<"Unable to open data_accepted_imit_to_interface semaphore."<<endl;
		exit(EXIT_FAILURE);
	}

	process_termination = sem_open("prc_term_imit_to_interface", O_CREAT | O_EXCL,
			S_IRWXU | S_IRWXO, 0);
	if (data_ready == SEM_FAILED)
	{
		cout<<"Unable to open prc_term_imit_to_interface."<<endl;
		exit(EXIT_FAILURE);
	}

	shared_mem_id = shmget(key_value, __SHARED_SEGMENT_SIZE,
			IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH);
	if (shared_mem_id == -1)
	{
		if (errno == EEXIST)
		{
			shared_mem_id = shmget(key_value, __SHARED_SEGMENT_SIZE,
								   S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH);
			if (shared_mem_id == -1)
			{
				cout<<"Unable to use existing shared memory segment."<<endl;
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			cout<<"Unable to allocate shared memory segment."<<endl;
			exit(EXIT_FAILURE);
		}
	}

	shared_mem = (char*)shmat(shared_mem_id, 0, 0);
	if (shared_mem == (void *)-1)
	{
		cout<<"Unable to attach shared memory segment."<<endl;
		exit(EXIT_FAILURE);
	}

	*((key_t *)shared_mem) = key_value;

	sem_post(data_ready);
*/
	// ***********************************************************************

    //init captured, search, track and etalon windows parameters
    init_windows_params(&captured_window, &search_window, &track_window,
      &etalon_window);

    // Matrixes for original and grayscaled images
    cv::Mat src_host;
    cv::Mat src_gray;

    // init pointer on image
    float *image;

    // Allocate n floats on host
    image = (float *)calloc(captured_window.width * captured_window.height,
      sizeof(float));

    // getting frame information
    cv::VideoCapture capt(filename);

    // declare Particle Filter class
    PF filterPF (captured_window, search_window, PARTICLE_NUMBER, PDF_WIDTH,
      PDF_HEIGHT, SIGMA_I, SIGMA_V, THRESHOLD_VALUE, FRAME_MATCH);

    // declare CLS filter class
    CLS_filter filterCLS (captured_window, search_window, PERIOD);

    // declare tracking class
    Tracking track (captured_window, track_window, etalon_window,
      GAUSSS_FILTER_SIZE, FILTRATION_THRESHOLD);

    // generating function for background approximation
    filterCLS.genFx(1/32., 1/32.);

    // generating Gauss filter matrix
    track.genGaussFilterMatrix();

    for (;;)
    {
		// break if frame isn't readable
        if (capt.read(src_host) == false)
            break;

        // convert to grayscale image
        cv::cvtColor(src_host, src_gray, CV_RGB2GRAY);

        // create float matrix of greyscale image
        for(int i = 0; i < SCREEN_WIDTH * SCREEN_HEIGHT; i++)
            image[i]= (float)*src_gray.ptr(0,i);

        if(first_frame == true)
        	tracking_type = 1;

        if(first_frame == true && tracking_type == 1)
        {
        	target_size = 0;
        	int num=0;

        	// get image from frame
        	track.getImage(image, track_window, frame_count);

        	// image preprocess (Gauss filter + Sobel operator + normalizing)
        	track.preprocess(tracking_type);

            // get output (tracking window size and position)
            track_window = track.getOutput(image, output_mode);

            // output image
            cv::Mat output_image(TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH,
              cv::DataType<float>::type, image);


            gettimeofday(&tim, NULL);
            output_t =tim.tv_sec+(tim.tv_usec/1000000.0) - track_t;

            cv::Mat preprocessed_u8;
            cv::convertScaleAbs(output_image, preprocessed_u8,1.,0.);

            cv::vector<cv::vector<cv::Point> > contours;
            cv::findContours( preprocessed_u8, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
            cv::Mat draw = output_image.clone();
            draw.setTo(0);

            for(int i = 0; i < contours.size(); i++)
            {
                if(contours[i].size() > target_size )
                {
                    target_size = contours[i].size();
                    num = i;
                }
            }
            cv::drawContours( draw, contours, num, cv::Scalar( 255, 0, 0 ) );
            cv::imshow("find cont", draw);
            cvMoveWindow("find cont", captured_window.width + 8, 0);
            cv::imshow("Preprocess output", output_image);
            cvMoveWindow("Preprocess output", captured_window.width + 8, track_window.height + 50);
        	first_frame = false;
        }
        else if(first_frame == false && tracking_type == 1)
        {
        	gettimeofday(&tim, NULL);
        	start_t = tim.tv_sec+(tim.tv_usec/1000000.0);

        	// get image from frame
        	track.getImage(image, track_window, frame_count);

            gettimeofday(&tim, NULL);
            get_im_t = tim.tv_sec+(tim.tv_usec/1000000.0);

        	// image preprocess (Gauss filter + Sobel operator + normalizing)
        	track.preprocess(tracking_type);

            gettimeofday(&tim, NULL);
            preproc_t = tim.tv_sec+(tim.tv_usec/1000000.0);

            // get output (tracking window size and position)
            track_window = track.getOutput(image, output_mode);

            // showing output image
            cv::Mat output_image(TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH,
              cv::DataType<float>::type, image);
            cv::imshow("Preprocess output", output_image);
            cvMoveWindow("Preprocess output", captured_window.width + 8, track_window.height + 50);

            gettimeofday(&tim, NULL);
            output_t =tim.tv_sec+(tim.tv_usec/1000000.0);

//--------------------------------------------------------------------------------------------

            cv::Mat preprocessed_u8;
            cv::convertScaleAbs(output_image, preprocessed_u8,1.,0.);

            cv::vector<cv::vector<cv::Point> > contours;
            cv::findContours( preprocessed_u8, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
            cv::Mat draw = output_image.clone();
            //cv::Mat draw = cvCreateMat(128, 128, CV_8UC4);
            draw.setTo(0);

            //printf("%d \n", contours.size());

            float cont_num = 0;
            float cont_size = 0;

            for(int i = 0; i < contours.size(); i++)
            {
                cv::drawContours( draw, contours, i, cv::Scalar( 255, 0, 0 ) );
                cv::Moments moments = cv::moments(draw, false);

                int delta_x = (int)(moments.m10 / moments.m00) -
                  track_window.width / 2;
                int delta_y = (int)(moments.m01 / moments.m00) -
                  track_window.height / 2;
/*
                if( abs(delta_x) + abs(delta_y) + k * abs(target_size - (int)contours[i].size()) < cont_size &&
                	abs(delta_x) + abs(delta_y) + target_size - contours[i].size() < 64	)
                {
               	cont_size = abs(delta_x) + abs(delta_y) + abs(target_size - (int)contours[i].size());
                	cont_num = i;
                }
*/
                float disp_xy = exp(-(abs(delta_x) + abs(delta_y)) / 10 ) ;
                float disp_size = exp(- abs((float)contours[i].size() - target_size)/ 10);

                if(disp_xy * disp_size > cont_size)
                {
                	cont_size = disp_xy * disp_size;
                	cont_num = i;
                }

                draw.setTo(0);

            }


            cv::drawContours( draw, contours, cont_num, cv::Scalar( 255, 0, 0 ) );
            cv::imshow("find cont", draw);
            cvMoveWindow("find cont", captured_window.width + 8, 0);

            cv::Moments moments = cv::moments(draw, false);

            track_window.centerX += (int)(moments.m10 / moments.m00) -
              track_window.width / 2 + 1;
            track_window.centerY += (int)(moments.m01 / moments.m00) -
              track_window.height / 2 + 1;

			if (contours.size() > 0)
				target_size = contours[cont_num].size();

            gettimeofday(&tim, NULL);
            track_t =tim.tv_sec+(tim.tv_usec/1000000.0);

//---------------------------------------------------------------------------

            system("clear");
            printf("%s \n \n", "Tracking type : CONTRAST");

            printf("%s \t %2.3f \t %s \n","Image getting time :",
              (get_im_t - start_t) * 1000, "ms");

            printf("%s \t %2.3f \t %s \n","Preprocessing time :",
              (preproc_t - get_im_t) * 1000, "ms");

            printf("%s \t %2.3f \t %s \n","Contrast track time :",
              (track_t - output_t) * 1000, "ms");

            printf("%s \t %2.3f \t %s \n","Output getting time :",
              (output_t - preproc_t) * 1000, "ms");

            printf("%s \t \t %2.3f \t %s \n \n","Total time :",
              (tim.tv_sec+(tim.tv_usec/1000000.0) - start_t) * 1000, "ms");

            printf("%s \t %d \n", "Main contour size :",
              target_size);

            printf("%s \t %1.0f \n", "Target position by x :",
              captured_window.width / 2 + track_window.centerX - 1);

            printf("%s \t %1.0f \n", "Target position by y :",
              captured_window.height / 2 + track_window.centerY - 1);


            if(target_size > 150)
            {
            	first_frame = true;
            	tracking_type = 2;

            	draw.setTo(0);
                cv::imshow("find cont", draw);
                cvMoveWindow("find cont", captured_window.width + 8, 0);
            }
        }

        if(first_frame == true && tracking_type == 2)
        {
        	// get image from frame
            track.getImage(image, track_window, frame_count);

            // image preprocess (Fauss filter + Sobel operator + normalizing)
            track.preprocess(tracking_type);

            // get etalon image from preprocessed image
            track.getEtalonImage();

            first_frame = false;
        }
        else if(first_frame == false && tracking_type == 2)
        {
        	gettimeofday(&tim, NULL);
        	start_t = tim.tv_sec+(tim.tv_usec/1000000.0);

        	// get image from frame
        	track.getImage(image, track_window, frame_count);

            gettimeofday(&tim, NULL);
            get_im_t = tim.tv_sec+(tim.tv_usec/1000000.0);

        	// image preprocess (Gauss filter + Sobel operator + normalizing)
        	track.preprocess(tracking_type);

            gettimeofday(&tim, NULL);
            preproc_t = tim.tv_sec+(tim.tv_usec/1000000.0);

        	// target tracking
        	track.track(0, 0, true);

            gettimeofday(&tim, NULL);
            track_t = tim.tv_sec+(tim.tv_usec/1000000.0);

        	// etalon image updating
        	track.updateEtalonImage(); /// processWindow

            gettimeofday(&tim, NULL);
            etalon_update_t = tim.tv_sec+(tim.tv_usec/1000000.0);

            //if(output_mode == 0)
            {
            	// get output (tracking window size and position)
            	track_window = track.getOutput(image, output_mode);

            	// showing output image
            	cv::Mat output_image(TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH,
            	  cv::DataType<float>::type, image);
            	cv::imshow("Preprocess output", output_image);
                cvMoveWindow("Preprocess output", captured_window.width + 8, track_window.height + 50);
            }
            //if(output_mode == 1)
             {
             	// get output (tracking window size and position)
            	 track_window = track.getOutput(image, output_mode+1);

            	 // showing output image
            	 cv::Mat output_image(TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH,
             	   cv::DataType<float>::type, image);
            	 cv::imshow("Etalon", output_image);
            	 cvMoveWindow("Etalon", captured_window.width + 8, track_window.height * 2 + 75);
             }
            //if(output_mode == 2)
             {
             	// get output (tracking window size and position)
             	track_window = track.getOutput(image, output_mode+2);

             	// showing output image
             	cv::Mat output_image(TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH,
             	  cv::DataType<float>::type, image);
             	cv::imshow("Correlation function", output_image);
             	cvMoveWindow("Correlation function", captured_window.width + 8, track_window.height * 3 + 100);
             }

             gettimeofday(&tim, NULL);
             output_t =tim.tv_sec+(tim.tv_usec/1000000.0);

             system("clear");
             printf("%s \n \n", "Tracking type : CORRELATION");

             printf("%s \t \t %2.3f \t %s \n","Image getting time :",
               (get_im_t - start_t) * 1000, "ms");

             printf("%s \t \t %2.3f \t %s \n","Preprocessing time :",
               (preproc_t - get_im_t) * 1000, "ms");

             printf("%s \t %2.3f \t %s \n","Norm correlation time :",
               (track_t - preproc_t) * 1000, "ms");

             printf("%s \t \t %2.3f \t %s \n","Etalon updating time :",
               (etalon_update_t - track_t) * 1000, "ms");

             printf("%s \t \t %2.3f \t %s \n","Output getting time :",
               (output_t - etalon_update_t) * 1000, "ms");

             printf("%s \t \t \t %2.3f \t %s \n \n","Total time :",
               (tim.tv_sec+(tim.tv_usec/1000000.0) - start_t) * 1000, "ms");

             printf("%s \t %2.2f \n", "Max of correlation function :",
               track.targetProbability);

             printf("%s \t %1.0f \n", "Target position by x :",
               captured_window.width / 2 + track_window.centerX - 1);

             printf("%s \t %1.0f \n", "Target position by y :",
               captured_window.height / 2 + track_window.centerY - 1);
        }


        // drawing target marker
        cv::circle(src_host, cvPoint(captured_window.width / 2 +
          track_window.centerX - 1, captured_window.height / 2 +
          track_window.centerY  - 1), 5, CV_RGB(255, 0, 0), 1, 8);

        // showing result image
        cv::imshow("Video", src_host);
        cvMoveWindow("Video", 0, 0);

        cv::Mat output(SCREEN_HEIGHT, SCREEN_WIDTH, CV_8UC3, src_host.data);

        // Frame transfer *************************************************
/*
		//check process termination semaphore
		sem_getvalue(process_termination, &sval);
		if (sval == 1)
		{
			break;
		}

		sem_wait(data_accepted);
		memcpy(shared_mem, output.data, __SHARED_SEGMENT_SIZE);
		sem_post(data_ready);
*/
        // ****************************************************************

        // getting mouse pointer position when right button is clicked
        cv::setMouseCallback("Video", mouse_callback);

        // getting input key and/or waiting 30 ms
        char c = cvWaitKey(25);

        // break if Esc is pressed
        if (c == 27)
            break;


        // update frame number
        frame_count++;
    }

    // Cleanup host
    free(image);

    // Release transfer resources *****************************************
/*
    sem_unlink("data_ready_imit_to_interface");
    sem_unlink("data_accepted_imit_to_interface");
    sem_unlink("prc_term_imit_to_interface");
    sem_close(process_termination);
    sem_close(data_ready);
    sem_close(data_accepted);
    shmdt(shared_mem);
    shmctl(shared_mem_id, IPC_RMID, 0);
*/
    // ********************************************************************


    return EXIT_SUCCESS;
}

void mouse_callback( int event, int x, int y, int flags, void* param )
{
    switch (event)
    {
    	case CV_EVENT_MOUSEMOVE:
    		break;

        case CV_EVENT_LBUTTONDOWN:
            track_window.centerX = x - 720 / 2 ;
            track_window.centerY = y - 568 / 2;
            break;

        case CV_EVENT_RBUTTONDOWN:
        	first_frame = true;
            track_window.centerX = x - 720 / 2 ;
            track_window.centerY = y - 568 / 2;
            break;

        case CV_EVENT_LBUTTONUP:
            break;
    }
}

void init_windows_params(window *capt_window, window *srch_window, window
  *tr_window , window *et_window)
{
    capt_window->width = SCREEN_WIDTH;
    capt_window->height = SCREEN_HEIGHT;
    capt_window->centerX = 0;
    capt_window->centerY = 0;

	srch_window->width = SEARCH_WINDOW_WIDTH;
    srch_window->height = SEARCH_WINDOW_HEIGHT;
    srch_window->centerX = SEARCH_WINDOW_CENTER_X;
    srch_window->centerY = SEARCH_WINDOW_CENTER_Y;

    tr_window->width = TRACK_WINDOW_WIDTH;
    tr_window->height = TRACK_WINDOW_HEIGHT;
    tr_window->centerX = TRACK_WINDOW_CENTER_X;
    tr_window->centerY = TRACK_WINDOW_CENTER_Y;

    et_window->width = ETALON_WINDOW_WIDTH;
    et_window->height = ETALON_WINDOW_HEIGHT;
    et_window->centerX = ETALON_WINDOW_CENTER_X;
    et_window->centerY = ETALON_WINDOW_CENTER_Y;
}

void correlation_tracking(Tracking track, float *image, bool *first_frame, bool *auto_tracking, bool debug_mode)
{
    timeval tim;
    double start_t, get_im_t, preproc_t, track_t, etalon_update_t, output_t;
    enum output_mode { PREPROCESS_OUTPUT, ETALON_OUTPUT, CORRELATION_OUTPUT };

	if(*first_frame == true)
	{
		// image preprocess (Fauss filter + Sobel operator + normalizing)
		track.preprocess(2);
		// get etalon image from preprocessed image
		track.getEtalonImage();

		first_frame = false;
	}
	else if(first_frame == false)
	{
        gettimeofday(&tim, NULL);
        get_im_t = tim.tv_sec+(tim.tv_usec/1000000.0);

    	// image preprocess (Gauss filter + Sobel operator + normalizing)
    	track.preprocess(2);

        gettimeofday(&tim, NULL);
        preproc_t = tim.tv_sec+(tim.tv_usec/1000000.0);

    	// target tracking
    	track.track(0, 0, true);

        gettimeofday(&tim, NULL);
        track_t = tim.tv_sec+(tim.tv_usec/1000000.0);

    	// etalon image updating
    	track.updateEtalonImage(); /// processWindow

        gettimeofday(&tim, NULL);
        etalon_update_t = tim.tv_sec+(tim.tv_usec/1000000.0);
	}

	if(debug_mode == true)
	{
		{
        	// get Preprocess output
        	track.getOutput(image, PREPROCESS_OUTPUT);

        	// showing output image
        	cv::Mat output_image(TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH,
        	  cv::DataType<float>::type, image);
        	cv::imshow("Preprocess output", output_image);
        }
        {
        	// get etalon output
        	 track_window = track.getOutput(image, ETALON_OUTPUT);

        	 // showing output image
        	 cv::Mat output_image(TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH,
         	   cv::DataType<float>::type, image);
        	 cv::imshow("Etalon", output_image);
         }
         {
        	// get correlation function output
         	track_window = track.getOutput(image, CORRELATION_OUTPUT);

         	// showing output image
         	cv::Mat output_image(TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH,
         	  cv::DataType<float>::type, image);
         	cv::imshow("Correlation function", output_image);
         }

         gettimeofday(&tim, NULL);
         output_t =tim.tv_sec+(tim.tv_usec/1000000.0);

         system("clear");
         printf("%s \n \n", "Tracking type : CORRELATION");

         printf("%s \t \t %2.3f \t %s \n","Preprocessing time :",
           (preproc_t - get_im_t) * 1000, "ms");

         printf("%s \t %2.3f \t %s \n","Norm correlation time :",
           (track_t - preproc_t) * 1000, "ms");

         printf("%s \t \t %2.3f \t %s \n","Etalon updating time :",
           (etalon_update_t - track_t) * 1000, "ms");

         printf("%s \t \t %2.3f \t %s \n","Output getting time :",
           (output_t - etalon_update_t) * 1000, "ms");

         printf("%s \t \t \t %2.3f \t %s \n \n","Total time :",
           (tim.tv_sec+(tim.tv_usec/1000000.0) - get_im_t) * 1000, "ms");

         printf("%s \t %2.2f \n", "Max of correlation function :",
           track.targetProbability);

         printf("%s \t %1.0f \n", "Target position by x :",
           SCREEN_WIDTH / 2 + track_window.centerX - 1);

         printf("%s \t %1.0f \n", "Target position by y :",
           SCREEN_HEIGHT / 2 + track_window.centerY - 1);
	}

}

void contrast_tracking(Tracking track, float *image, bool *first_frame, bool *auto_tracking, bool debug_mode)
{
    timeval tim;
    double start_t, get_im_t, preproc_t, track_t, etalon_update_t, output_t;
    int target_size;

	if(*first_frame == true)
	{
    	target_size = 0; /////add to class !!!!!!!!!!!
    	int num=0;

    	// image preprocess (Gauss filter + Sobel operator + normalizing)
    	track.preprocess(0);

        // get output (tracking window size and position)
        track_window = track.getOutput(image, 0);

        // output image
        cv::Mat output_image(TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH,
          cv::DataType<float>::type, image);

        cv::Mat preprocessed_u8;
        cv::convertScaleAbs(output_image, preprocessed_u8,1.,0.);

        cv::vector<cv::vector<cv::Point> > contours;
        cv::findContours( preprocessed_u8, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
        cv::Mat draw = output_image.clone();
        draw.setTo(0);

        for(int i = 0; i < contours.size(); i++)
        {
            if(contours[i].size() > target_size )
            {
                target_size = contours[i].size();
                num = i;
            }
        }
        cv::drawContours( draw, contours, num, cv::Scalar( 255, 0, 0 ) );
    	first_frame = false;
    }
    else if(first_frame == false)
    {
    	gettimeofday(&tim, NULL);
    	start_t = tim.tv_sec+(tim.tv_usec/1000000.0);

        gettimeofday(&tim, NULL);
        get_im_t = tim.tv_sec+(tim.tv_usec/1000000.0);

    	// image preprocess (Gauss filter + Sobel operator + normalizing)
    	track.preprocess(0);

        gettimeofday(&tim, NULL);
        preproc_t = tim.tv_sec+(tim.tv_usec/1000000.0);

        // get output (tracking window size and position)
        track_window = track.getOutput(image, 0);

        // showing output image
        cv::Mat output_image(TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH,
          cv::DataType<float>::type, image);

        gettimeofday(&tim, NULL);
        output_t =tim.tv_sec+(tim.tv_usec/1000000.0);

        cv::Mat preprocessed_u8;
        cv::convertScaleAbs(output_image, preprocessed_u8,1.,0.);

        cv::vector<cv::vector<cv::Point> > contours;
        cv::findContours( preprocessed_u8, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
        cv::Mat draw = output_image.clone();
        //cv::Mat draw = cvCreateMat(128, 128, CV_8UC4);
        draw.setTo(0);

        float cont_num = 0;
        float cont_size = 0;

        for(int i = 0; i < contours.size(); i++)
        {
            cv::drawContours( draw, contours, i, cv::Scalar( 255, 0, 0 ) );
            cv::Moments moments = cv::moments(draw, false);

            int delta_x = (int)(moments.m10 / moments.m00) -
              track_window.width / 2;
            int delta_y = (int)(moments.m01 / moments.m00) -
              track_window.height / 2;

            float disp_xy = exp(-(abs(delta_x) + abs(delta_y)) / 10 ) ;
            float disp_size = exp(- abs((float)contours[i].size() - target_size)/ 10);

            if(disp_xy * disp_size > cont_size)
            {
            	cont_size = disp_xy * disp_size;
            	cont_num = i;
            }

            draw.setTo(0);

        }

        cv::drawContours( draw, contours, cont_num, cv::Scalar( 255, 0, 0 ) );

        cv::Moments moments = cv::moments(draw, false);

        track_window.centerX += (int)(moments.m10 / moments.m00) -
          track_window.width / 2 + 1;
        track_window.centerY += (int)(moments.m01 / moments.m00) -
          track_window.height / 2 + 1;

		if (contours.size() > 0)
			target_size = contours[cont_num].size();

        gettimeofday(&tim, NULL);
        track_t =tim.tv_sec+(tim.tv_usec/1000000.0);

    	if (debug_mode == true)
    	{
            cv::imshow("Preprocess output", output_image);

            cv::imshow("find cont", draw);

            system("clear");
            printf("%s \n \n", "Tracking type : CONTRAST");

            printf("%s \t %2.3f \t %s \n","Image getting time :",
              (get_im_t - start_t) * 1000, "ms");

            printf("%s \t %2.3f \t %s \n","Preprocessing time :",
              (preproc_t - get_im_t) * 1000, "ms");

            printf("%s \t %2.3f \t %s \n","Contrast track time :",
              (track_t - output_t) * 1000, "ms");

            printf("%s \t %2.3f \t %s \n","Output getting time :",
              (output_t - preproc_t) * 1000, "ms");

            printf("%s \t \t %2.3f \t %s \n \n","Total time :",
              (tim.tv_sec+(tim.tv_usec/1000000.0) - start_t) * 1000, "ms");

            printf("%s \t %d \n", "Main contour size :",
              target_size);

            printf("%s \t %1.0f \n", "Target position by x :",
              SCREEN_WIDTH / 2 + track_window.centerX - 1);

            printf("%s \t %1.0f \n", "Target position by y :",
              SCREEN_HEIGHT / 2 + track_window.centerY - 1);
    	}

    }
}
