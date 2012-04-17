/*
 * imtation_utils.h
 *
 *  Created on: Apr 11, 2012
 *      Author: lucer
 */

#ifndef IMTATION_UTILS_H_
#define IMTATION_UTILS_H_

#include <time.h>
#include <errno.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/ipc.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

#define __WINDOW_WIDTH 720.0
#define __WINDOW_HEIGHT 576.0
#define __MAX_DISTANCE 12000.0
#define __MAX_RELATIVE_DIAMETER 2.0
#define __REAL_HARPOON_DIAMETER 0.76
#define __REAL_MOSQUIT_DIAMETER 0.343
#define __CAMERA_ANGLE_RAD_X 0.02530727
#define __CAMERA_ANGLE_RAD_Y 0.01890773
#define __OBSERVATION_POINT_HEIGHT 16.0
#define __MODELING_DATA_FILE_NAME "modeling.dat"
#define __PI 3.1415926535897
#define __MAX_IMIT_STRING_LENGTH 50

//redefined from linux/time.h
#ifndef CLOCK_REALTIME
	#define CLOCK_REALTIME 0
#endif
#ifndef TIMER_ABSTIME
	#define TIMER_ABSTIME 0x01
#endif
//###########################

using namespace std;
//structure for target coordinates (orthogonal system)
typedef struct orto_coords
{
	float x;
	float y;
	float h;
} orto_coords;

//enumeration that represents king of the angle value
enum angle_t
{
	AT_RELATIVE_BEARING,
	AT_ELEVATION_BEARING
};

//standard math function prototype redefinition
float atan2f(float y, float x);

//imitation class
class ImitationEngine
{
public:
	ImitationEngine();
	~ImitationEngine();
	bool imit_process(int *x_res, int *y_res, int *size_res);
	void reinit_imit(void);

private:
	int _right_rounding(double val);
	void _get_modeling_data(orto_coords *coords_array, uint points_count);
	unsigned int _get_strings_count(void);
	char* _get_missile_type(void);
	uint _get_time_interval(void);
	double _get_azimuth(void);
	float _get_screen_size(double distance, uint dimension, float real_size, float camera_angle_dim);
	float _get_target_coords(uint dimension, double camera_angle_dim, double target_angle,
						      double optic_angle, angle_t angle_type);
	double _get_deck_distance(orto_coords *trajectory_points, long tick_count);
	double _get_relative_bearing(orto_coords *trajectory_points, long tick_count);
	double _get_elevation_bearing(orto_coords *trajectory_points, long tick_count, double* distance);

	uint _points_count, _time_interval_ms;
	double _distance_left, _time_interval_ns, _frames_per_second,
		   _optical_elevation_bearing, _optical_relative_bearing,
		   _real_diameter;
	long _tick_count;
	orto_coords* _trajectory_points;
	fstream _fs;
	FILE *_hFile;
	char _missile_type[20], _buf[__MAX_IMIT_STRING_LENGTH];
	int _azimuth_angle_deg;
};

#endif /* IMTATION_UTILS_H_ */
