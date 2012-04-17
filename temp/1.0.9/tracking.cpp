/*
 * tracking.cpp
 *
 * tracking file
 *
 *  Created on: Mar 3, 2012
 *      Author: Maxym Zastavny
 */


#include "hdr/init_data.h"
#include <bug.h>
#include "hdr/bug_struct.h"
#include "video_transfer.h"

using namespace std;

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

bool correlation_tracking(Tracking track, float *image, window *track_window, bool *first_frame, int tracking_mode, bool debug_mode)
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

		*first_frame = false;
	}
	else if(*first_frame == false)
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

        *track_window = track.getOutput(image, PREPROCESS_OUTPUT);
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
        	 *track_window = track.getOutput(image, ETALON_OUTPUT);

        	 // showing output image
        	 cv::Mat output_image(TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH,
         	   cv::DataType<float>::type, image);
        	 cv::imshow("Etalon", output_image);
         }
         {
        	// get correlation function output
         	*track_window = track.getOutput(image, CORRELATION_OUTPUT);

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
           SCREEN_WIDTH / 2 + track_window->centerX - 1);

         printf("%s \t %1.0f \n", "Target position by y :",
           SCREEN_HEIGHT / 2 + track_window->centerY - 1);
	}
	int target_size = track.changeTrackingMode() / 6;

	if(target_size < 30)
	{
		return true;
	}
	else
	{
		return false;
	}

}

bool contrast_tracking(Tracking *track, float *image, window *track_window, bool *first_frame, int tracking_mode, bool debug_mode)
{
    timeval tim;
    double start_t, get_im_t, preproc_t, track_t, etalon_update_t, output_t;

	if(*first_frame == true)
	{
    	track->target_size = 0;
    	int num=0;

    	// image preprocess (Gauss filter + Sobel operator + normalizing)
    	track->preprocess(1);

        // get output (tracking window size and position)
        *track_window = track->getOutput(image, 0);

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
            if(contours[i].size() > track->target_size )
            {
                track->target_size = contours[i].size();
                num = i;
            }
        }
        cv::drawContours( draw, contours, num, cv::Scalar( 255, 0, 0 ) );
    	*first_frame = false;
    }
    else if(*first_frame == false)
    {

    	gettimeofday(&tim, NULL);
    	start_t = tim.tv_sec+(tim.tv_usec/1000000.0);

        gettimeofday(&tim, NULL);
        get_im_t = tim.tv_sec+(tim.tv_usec/1000000.0);

    	// image preprocess (Gauss filter + Sobel operator + normalizing)
    	track->preprocess(1);

        gettimeofday(&tim, NULL);
        preproc_t = tim.tv_sec+(tim.tv_usec/1000000.0);

        // get output (tracking window size and position)
        *track_window = track->getOutput(image, 0);

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
              track_window->width / 2;
            int delta_y = (int)(moments.m01 / moments.m00) -
              track_window->height / 2;

            float disp_xy = exp(-(abs(delta_x) + abs(delta_y)) / 10 ) ;
            float disp_size = exp(- abs((float)contours[i].size() - track->target_size)/ 10);

            if(disp_xy * disp_size > cont_size)
            {
            	cont_size = disp_xy * disp_size;
            	cont_num = i;
            }

            draw.setTo(0);

        }

        cv::drawContours( draw, contours, cont_num, cv::Scalar( 255, 0, 0 ) );

        cv::Moments moments = cv::moments(draw, false);

        track_window->centerX += (int)(moments.m10 / moments.m00) -
          track_window->width / 2 + 1;
        track_window->centerY += (int)(moments.m01 / moments.m00) -
          track_window->height / 2 + 1;

		if (contours.size() > 0)
			track->target_size = contours[cont_num].size();

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
              track->target_size);

            printf("%s \t %1.0f \n", "Target position by x :",
              SCREEN_WIDTH / 2 + track_window->centerX - 1);

            printf("%s \t %1.0f \n", "Target position by y :",
              SCREEN_HEIGHT / 2 + track_window->centerY - 1);
    	}
    }
    if(track->target_size > 30)
    {
    	*first_frame = true;
    	return false;
    }
    else
    {
    	return true;
    }
}
