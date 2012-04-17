/*
 * imitation_utils.cpp
 *
 *  Created on: Apr 11, 2012
 *      Author: lucer
 */

#include "hdr/imitation_utils.h"

//function for frame processing
bool ImitationEngine::imit_process(int *x_res, int *y_res, int *size_res)
{
    double x_coord, y_coord;

	_distance_left = _get_deck_distance(_trajectory_points, _tick_count);

	//missile has reached its target
	if (_distance_left <= 0)
	{
		return false;
	}

	//get screen size of target in pixels
	float scr_size = _get_screen_size(_distance_left, __WINDOW_WIDTH, _real_diameter,
									   __CAMERA_ANGLE_RAD_X);

	//screen coordinates calculation
	double rel_bearing = _get_relative_bearing(_trajectory_points, _tick_count),
			 el_bearing = _get_elevation_bearing(_trajectory_points, _tick_count, &_distance_left);
	x_coord = _get_target_coords(__WINDOW_WIDTH, __CAMERA_ANGLE_RAD_X, rel_bearing,
								_optical_relative_bearing, AT_RELATIVE_BEARING);

	y_coord = _get_target_coords(__WINDOW_HEIGHT, __CAMERA_ANGLE_RAD_Y, el_bearing,
								_optical_elevation_bearing, AT_ELEVATION_BEARING);

	y_coord = __WINDOW_HEIGHT - y_coord;

	//TODO: PASS target model
	*x_res = _right_rounding(x_coord),
	*y_res = _right_rounding(y_coord),
	*size_res = _right_rounding((double)scr_size);

	//frame counter increment
    _tick_count++;
    return true;
}

//initialization routine
ImitationEngine::ImitationEngine()
{
    _distance_left = __MAX_DISTANCE;
    _points_count = _get_strings_count() - 3;
    _trajectory_points = new orto_coords[_points_count];
    _get_modeling_data(_trajectory_points, _points_count);
    _optical_elevation_bearing = 0.0;
    _optical_relative_bearing = _get_azimuth();

	//missile type detection, should be at the end
	if (strcmp(_get_missile_type(), "Harpoon") == 0)
	{
		_real_diameter = __REAL_HARPOON_DIAMETER;
		return;
	}

	if (strcmp(_get_missile_type(), "Mosquit") == 0)
	{
		_real_diameter = __REAL_MOSQUIT_DIAMETER;
		return;
	}

	_real_diameter = __REAL_HARPOON_DIAMETER;
	_tick_count = 0;
}

ImitationEngine::~ImitationEngine()
{

}

//for imitation process restart
void ImitationEngine::reinit_imit(void)
{
	_tick_count = 0;
}

//right rounding function
int ImitationEngine::_right_rounding(double val)
{
	return (int)(val + 0.5);
}

void ImitationEngine::_get_modeling_data(orto_coords *coords_array, uint points_count)
{
	_hFile = fopen(__MODELING_DATA_FILE_NAME, "r");

	if (_hFile == NULL)
	{
		cout<<"Unable to open file with model data."<<endl;
		exit(1);
	}

	fgets(_buf, __MAX_IMIT_STRING_LENGTH, _hFile);

	if (sscanf(_buf, "Time interval: %u", &_time_interval_ms) == 0)
	{
		cout<<"Unable to get value of time interval."<<endl;
		exit(1);
	}

	fgets(_buf, __MAX_IMIT_STRING_LENGTH, _hFile);

	if (sscanf(_buf, "Azimuth angle: %d", &_azimuth_angle_deg) == 0)
	{
		cout<<"Unable to get value of azimuth angle."<<endl;
		exit(1);
	}

	fgets(_buf, __MAX_IMIT_STRING_LENGTH, _hFile);

	if (sscanf(_buf, "Type: %s", _missile_type) == 0)
	{
		cout<<"Unable to get missile type."<<endl;
		exit(1);
	}

	for (uint i = 0; i < points_count; i++)
	{
		fgets(_buf, __MAX_IMIT_STRING_LENGTH, _hFile);

		int _return_code = sscanf(_buf, "%f %f %f" /* "%d %d %d" */, &coords_array[i].x,
								  &coords_array[i].y, &coords_array[i].h);

		if (_return_code == 0 || _return_code == EOF)
		{
			cout<<"Error while parsing trajectory data."<<endl;
			exit(1);
		}
	}

	fclose(_hFile);
}

uint ImitationEngine::_get_strings_count(void)
{
	uint count = 0;

	_hFile = fopen(__MODELING_DATA_FILE_NAME, "r");

	if (_hFile == NULL)
	{
		cout<<"Unable to open file with model data."<<endl;
		exit(1);
	}

	while (true)
	{
		if (fgets(_buf, __MAX_IMIT_STRING_LENGTH, _hFile) == NULL)
		{
			if (feof(_hFile) == 0)
			{
				cout<<"Wrong input data file format (strings are too long)"<<endl;
				exit(1);
			}
			else
			{
				break;
			}
		}

		count++;
	}

	fclose(_hFile);
	return count;
}

char* ImitationEngine::_get_missile_type(void)
{
	return _missile_type;
}

uint ImitationEngine::_get_time_interval(void)
{
	return _time_interval_ms;
}

double ImitationEngine::_get_azimuth(void)
{
	return _azimuth_angle_deg * __PI / 180.0;
}

float ImitationEngine::_get_screen_size(double distance, uint dimension, float real_size, float camera_angle_dim)
{
	float res = 2 * dimension / camera_angle_dim;

	return res * atan2f(real_size, 2 * distance);
}

float ImitationEngine::_get_target_coords(uint dimension, double camera_angle_dim, double target_angle,
					      double optic_angle, angle_t angle_type)
{
	if (target_angle == optic_angle)
	{
		if (angle_type == AT_ELEVATION_BEARING)
			return __WINDOW_HEIGHT / 2;
		else
			return __WINDOW_WIDTH / 2;
	}

	float pixel_weight = dimension / camera_angle_dim, delta_angle = optic_angle - target_angle;

	return dimension / 2 - delta_angle * pixel_weight;
}

double ImitationEngine::_get_deck_distance(orto_coords *trajectory_points, long tick_count)
{
	float reslt;

	if (trajectory_points[tick_count].x <= 0.0 || trajectory_points[tick_count].y <= 0.0 ||
		trajectory_points[tick_count].h <= 0.0)
		return 0.0;

	reslt = pow(trajectory_points[tick_count].x, 2);
	reslt += pow(trajectory_points[tick_count].y, 2);
	reslt += pow(trajectory_points[tick_count].h, 2);
	return sqrt(reslt);
}

double ImitationEngine::_get_relative_bearing(orto_coords *trajectory_points, long tick_count)
{
	if (trajectory_points[tick_count].x == 0 && trajectory_points[tick_count].y != 0)
	{
		if (trajectory_points[tick_count].y > 0)
		{
			return 0.0;
		}
		else
		{
			return __PI;
		}
	}

	if (trajectory_points[tick_count].x != 0 && trajectory_points[tick_count].y == 0)
	{
		if (trajectory_points[tick_count].x > 0)
		{
			return __PI / 2.0;
		}
		else
		{
			return __PI * 1.5;
		}
	}

	double reslt = (double)atan2(trajectory_points[tick_count].x,
								 trajectory_points[tick_count].y);

	return reslt;
}

double ImitationEngine::_get_elevation_bearing(orto_coords *trajectory_points, long tick_count, double* distance)
{
	float h_tmp = trajectory_points[tick_count].h;

	h_tmp -= __OBSERVATION_POINT_HEIGHT;
	int _sign = ((h_tmp > 0) ? 1 : -1);

	if (h_tmp == 0)
	{
		return 0.0;
	}

	if (trajectory_points[tick_count].x == 0 && trajectory_points[tick_count].y != 0)
	{
		if (trajectory_points[tick_count].y > 0)
		{
			return acos(trajectory_points[tick_count].y / *distance) * _sign;
		}
		else
		{
			return acos(abs(trajectory_points[tick_count].y) / *distance) * _sign;
		}
	}

	if (trajectory_points[tick_count].x != 0 && trajectory_points[tick_count].y == 0)
	{
		if (trajectory_points[tick_count].x > 0)
		{
			return acos(trajectory_points[tick_count].x / *distance) * _sign;
		}
		else
		{
			return acos(abs(trajectory_points[tick_count].x) / *distance) * _sign;
		}
	}

	if (trajectory_points[tick_count].x == 0 && trajectory_points[tick_count].y == 0 &&
			h_tmp != 0)
	{
		if (trajectory_points[tick_count].h > 0)
		{
			return __PI / 2.0;
		}
		else
		{
			return __PI / (-2.0);
		}
	}

	double reslt = atan2((double)h_tmp, (double)sqrt(
				 	 	 (double)pow(trajectory_points[tick_count].x, 2) +
				 	 	 (double)pow(trajectory_points[tick_count].y, 2)));

	return reslt;
}
