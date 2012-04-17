/*
 * video_transfer.h
 *
 *  Created on: Apr 6, 2012
 *      Author: lucer
 */

#ifndef VIDEO_TRANSFER_H_
#define VIDEO_TRANSFER_H_

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
#include <stdlib.h>

using namespace std;

#define MAX_STRING_LENGTH 255
#define __WINDOW_WIDTH 720.0
#define __WINDOW_HEIGHT 576.0
#define __SHARED_SEGMENT_SIZE (__WINDOW_WIDTH*__WINDOW_HEIGHT*3)

class VideoTransfer
{
public:
	VideoTransfer(const char *data_ready, const char *data_accepted,
				  const char *prc_term, key_t key_value);
	~VideoTransfer();
	bool pass_frame(void *data);

private:
	sem_t *_data_ready, *_data_accepted, *_process_termination;
	char *_shared_mem, _data_ready_name[MAX_STRING_LENGTH],
	_data_accepted_name[50], _prc_term_name[50];
	int _shared_mem_id, _sval;
	key_t _key_value;
};

#endif /* VIDEO_TRANSFER_H_ */
