/*
 * video_transfer.cpp
 *
 *  Created on: Apr 6, 2012
 *      Author: lucer
 */

#include "hdr/video_transfer.h"

VideoTransfer::VideoTransfer(const char *data_ready_s,
							 const char *data_accepted_s,
							 const char *prc_term_s, key_t key_value_t)
{
	strcpy(_data_ready_name, data_ready_s);
	strcpy(_data_accepted_name, data_accepted_s);
	strcpy(_prc_term_name, prc_term_s);
	_key_value = key_value_t;

	sem_unlink(_data_ready_name);
	sem_unlink(_data_accepted_name);
	sem_unlink(_prc_term_name);

	_data_ready = sem_open("data_ready_imit_to_interface", O_CREAT | O_EXCL,
						  S_IRWXU | S_IRWXO, 0);
	if (_data_ready == SEM_FAILED)
	{
		cout<<"Unable to open data_ready_imit_to_interface semaphore."
			<<errno<<endl;
		exit(EXIT_FAILURE);
	}

	_data_accepted = sem_open("data_accepted_imit_to_interface", O_CREAT | O_EXCL,
			S_IRWXU | S_IRWXO, 0);
	if (_data_ready == SEM_FAILED)
	{
		cout<<"Unable to open data_accepted_imit_to_interface semaphore."<<endl;
		exit(EXIT_FAILURE);
	}

	_process_termination = sem_open("prc_term_imit_to_interface", O_CREAT | O_EXCL,
			S_IRWXU | S_IRWXO, 0);
	if (_data_ready == SEM_FAILED)
	{
		cout<<"Unable to open prc_term_imit_to_interface."<<endl;
		exit(EXIT_FAILURE);
	}

	_shared_mem_id = shmget(_key_value, __SHARED_SEGMENT_SIZE,
			IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH);
	if (_shared_mem_id == -1)
	{
		if (errno == EEXIST)
		{
			_shared_mem_id = shmget(_key_value, __SHARED_SEGMENT_SIZE,
								   S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH);
			if (_shared_mem_id == -1)
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

	_shared_mem = (char*)shmat(_shared_mem_id, 0, 0);
	if (_shared_mem == (void *)-1)
	{
		cout<<"Unable to attach shared memory segment."<<endl;
		exit(EXIT_FAILURE);
	}

	*((key_t *)_shared_mem) = _key_value;

	sem_post(_data_ready);
}

VideoTransfer::~VideoTransfer()
{
    //release all captured system resources
    sem_unlink(_data_ready_name);
    sem_unlink(_data_accepted_name);
    sem_unlink(_prc_term_name);
    sem_close(_process_termination);
    sem_close(_data_ready);
    sem_close(_data_accepted);
    shmdt(_shared_mem);
    shmctl(_shared_mem_id, IPC_RMID, 0);
}

bool VideoTransfer::pass_frame(void *data)
{
	sem_getvalue(_process_termination, &_sval);

	if (_sval == 1)
	{
		return false;
	}

	sem_wait(_data_accepted);
	memcpy(_shared_mem, data, __SHARED_SEGMENT_SIZE);
	sem_post(_data_ready);
	return true;
}
