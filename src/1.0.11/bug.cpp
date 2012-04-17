/*
 * bug.cpp
 *
 * main project file
 *
 *  Created on: Mar 3, 2012
 *      Author: Maxym Zastavny
 */

#include "hdr/init_data.h"
#include "hdr/processing.h"
#include "hdr/processing_struct.h"
#include "hdr/video_transfer.h"
#include "hdr/imitation_utils.h"
#include "tracking.cpp"

// window in which we are tracking target
window track_window;

// frame number variable
bool first_frame;

void mouse_callback( int, int, int, int, void *);

int main(int argc, char *argv[])
{
	long int frame_count = 0;

	VideoTransfer *video_transfer;
	ImitationEngine *imit_object;

	// video file name address
	const char* filename = argc == 2 ? argv[1] :
	  "../../../../videos/helicopter.mpg";

    int tracking_mode = AUTO_TRACKING, imitation_x, imitation_y, imitation_size;
    bool debug_mode = true;
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
    image = (float *)calloc(captured_window.width * captured_window.height,
      sizeof(float));

    // getting frame information
    cv::VideoCapture capt(filename);

    // declare tracking class
    Tracking track (captured_window, track_window, etalon_window,
      GAUSSS_FILTER_SIZE, FILTRATION_THRESHOLD);

    // initialize object for video transfer purposes
    video_transfer = new VideoTransfer("data_ready_imit_to_interface",
    								   "data_accepted_imit_to_interface",
    								   "prc_term_imit_to_interface", __KEY_VALUE);

    // initialize data for the target imitation
    imit_object = new ImitationEngine();

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

        if(first_frame == true)
        	small_target = true;

        switch(tracking_mode)
        {
        	case AUTO_TRACKING:
        		if(small_target == true)
        		{
        		    small_target = contrast_tracking(&track, image,
        		      &track_window, &first_frame, AUTO_TRACKING, debug_mode);
        		}
        		if(small_target == false)
        		{
        			if(first_frame == true)
        			{
        				track.getImage(image, track_window, frame_count);
        			}
        			small_target = correlation_tracking(track, image,
        			  &track_window, &first_frame, AUTO_TRACKING, debug_mode);
        		}
        		break;
            case CORRELATION_TRACKING:
            	correlation_tracking(track, image, &track_window,
            	  &first_frame, CORRELATION_TRACKING, debug_mode);
            	break;
            case CONTRAST_TRACKING:
            	contrast_tracking(&track, image, &track_window,
            	  &first_frame, CONTRAST_TRACKING, debug_mode);
                break;
            case NONE_TRACKING:
                break;
        }

        //#####################################################################

        if (imit_object->
        	imit_process(&imitation_x, &imitation_y, &imitation_size) == true)
        {
        	cv::circle(src_host, cvPoint(imitation_x, imitation_y), 2,
        			   CV_RGB(255, 0, 0), 1, 8);
        }
        else
        {
        	imit_object->reinit_imit();
        }

        //#####################################################################

        // drawing target marker
        cv::rectangle(src_host,
          cvPoint(captured_window.width / 2 + track_window.centerX - 10,
                  captured_window.height / 2 + track_window.centerY  - 8),
          cvPoint(captured_window.width / 2 + track_window.centerX + 8,
                  captured_window.height / 2 + track_window.centerY  + 6),
          CV_RGB(255, 0, 0), 1, 8);

        // showing result image
        cv::imshow("Video", src_host);

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
        char c = cvWaitKey(20);

         // break if Esc is pressed
        if (c == 27)
            break;

        // update frame number
        frame_count++;
    }

    // Cleanup host
    free(image);
    delete video_transfer;
    delete imit_object;

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
