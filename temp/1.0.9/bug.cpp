/*
 * bug.cpp
 *
 * main project file
 *
 *  Created on: Mar 3, 2012
 *      Author: Maxym Zastavny
 */


#include "hdr/init_data.h"
#include <bug.h>
#include "hdr/bug_struct.h"
#include "video_transfer.h"
#include "tracking.cpp"

using namespace std;



// window in which we are tracking target
window track_window;

// frame number variable
bool first_frame;

void mouse_callback( int, int, int, int, void *);
/*
void init_windows_params(window *, window *, window *, window *);

bool correlation_tracking(Tracking, float*, bool*, int, bool);
bool contrast_tracking(Tracking*, float*, bool*, int, bool);
*/
int main(int argc, char *argv[])
{
	long int frame_count = 0;
	VideoTransfer *video_transfer;

    const char* filename = argc == 2 ? argv[1] :
      "/home/Max/prj/videos/helicopter.mpg";

    int mode = AUTO_TRACKING;
    bool debug_mode = false;
    bool small_target = true;
    first_frame = true;


    // window captured from the camera(video file), window in which we are
    // searching target, target etalon image
    window captured_window, search_window, etalon_window;

    //init captured, search, track and etalon windows parameters
    init_windows_params(&captured_window, &search_window, &track_window,
      &etalon_window);

    // Matrixes for original and grayscaled images
    cv::Mat src_host, src_gray;

    // init pointer on image
    float *image;

    // Allocate n floats on host
    image = (float *)calloc(captured_window.width * captured_window.height,
      sizeof(float));

    // getting frame information
    cv::VideoCapture capt(filename);

    // declare tracking class
    Tracking track (captured_window, track_window, etalon_window,
      GAUSSS_FILTER_SIZE, FILTRATION_THRESHOLD);

    video_transfer = new VideoTransfer("data_ready_imit_to_interface",
    								   "data_accepted_imit_to_interface",
    								   "prc_term_imit_to_interface", __KEY_VALUE);

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

        // getting video frame to GPU
        track.getImage(image, track_window, frame_count);

        switch(mode)
        {
        	case AUTO_TRACKING:
        		if(small_target == false)
        		{
        			small_target = correlation_tracking(track, image, &track_window, &first_frame, AUTO_TRACKING, debug_mode);
        		}
        		if(small_target == true)
        		{
        		    small_target = contrast_tracking(&track, image, &track_window, &first_frame, AUTO_TRACKING, debug_mode);
        		}
        		break;
            case CORRELATION_TRACKING:
            	correlation_tracking(track, image, &track_window, &first_frame, CORRELATION_TRACKING, debug_mode);
            	break;
            case CONTRAST_TRACKING:
            	contrast_tracking(&track, image, &track_window, &first_frame, CONTRAST_TRACKING, debug_mode);
                break;
            case NONE_TRACKING:
                break;
        }

        // drawing target marker
        cv::circle(src_host, cvPoint(captured_window.width / 2 +
          track_window.centerX - 1, captured_window.height / 2 +
          track_window.centerY  - 1), 5, CV_RGB(255, 0, 0), 1, 8);

        // showing result image
        cv::imshow("Video", src_host);
        //cvMoveWindow("Video", 0, 0);

        cv::Mat output(SCREEN_HEIGHT, SCREEN_WIDTH, CV_8UC3, src_host.data);

        /*
        if (video_transfer->pass_frame(output.data) == false)
        {
        	break;
        }
        */

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
    delete video_transfer;

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
/*
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

bool correlation_tracking(Tracking track, float *image, bool *first_frame, int tracking_mode, bool debug_mode)
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

        track_window = track.getOutput(image, PREPROCESS_OUTPUT);
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

bool contrast_tracking(Tracking *track, float *image, bool *first_frame, int tracking_mode, bool debug_mode)
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
        track_window = track->getOutput(image, 0);

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
        track_window = track->getOutput(image, 0);

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

        track_window.centerX += (int)(moments.m10 / moments.m00) -
          track_window.width / 2 + 1;
        track_window.centerY += (int)(moments.m01 / moments.m00) -
          track_window.height / 2 + 1;

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
              SCREEN_WIDTH / 2 + track_window.centerX - 1);

            printf("%s \t %1.0f \n", "Target position by y :",
              SCREEN_HEIGHT / 2 + track_window.centerY - 1);
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
*/
