#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>

#include <iostream>
#include <algorithm>

#include <cstdlib>
#include <cstdio>

#include <cmath>
#include <ctime>
#include <cstring>

#include <omp.h>

using namespace std;

/**
  * helper routine: check if array is sorted correctly
  */
bool isSortedParallel(int ref[], int data[], const size_t size){
	bool flag = true;
	std::sort(ref, ref + size);
	#pragma omp parallel shared(flag) num_threads(omp_get_max_threads())
	{
		
		#pragma omp for
		for (size_t idx = 0; idx < size; ++idx)
		{
			#pragma omp cancellation point for 
			if (ref[idx] != data[idx]) {
				#pragma omp atomic write
				flag = false;
				#pragma omp cancel for
			}
		}
	}
	
	
	return flag;
}




void MsMergeSequential(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin) {
	long left = begin1;
	long right = begin2;

	long idx = outBegin;

	while (left < end1 && right < end2) {
		if (in[left] <= in[right]) {
			out[idx] = in[left];
			left++;
		} else {
			out[idx] = in[right];
			right++;
		}
		idx++;
	}

	while (left < end1) {
		out[idx] = in[left];
		left++, idx++;
	}

	while (right < end2) {
		out[idx] = in[right];
		right++, idx++;
	}
}


/**
  * sequential MergeSort
  */
// TODO: remember one additional parameter (depth)
// TODO: recursive calls could be taskyfied
// TODO: task synchronization also is required
void MsSequential(int *array, int *tmp, bool inplace, long begin, long end) {
	if (begin < (end - 1)) {
		const long half = (begin + end) / 2;
		MsSequential(array, tmp, !inplace, begin, half);
		MsSequential(array, tmp, !inplace, half, end);
		if (inplace) {
			MsMergeSequential(array, tmp, begin, half, half, end, begin);
		} else {
			MsMergeSequential(tmp, array, begin, half, half, end, begin);
		}
	} else if (!inplace) {
		tmp[begin] = array[begin];
	}
}


/**
  * Serial MergeSort
  */
// TODO: this function should create the parallel region
// TODO: good point to compute a good depth level (cut-off)
void MsSerial(int *array, int *tmp, const size_t size) {

   // TODO: parallel version of MsSequential will receive one more parameter: 'depth' (used as cut-off)
	MsSequential(array, tmp, true, 0, size);
}

/**
  * sequential MergeSort
  */
// TODO: remember one additional parameter (depth)
// TODO: recursive calls could be taskyfied
// TODO: task synchronization also is required
void MsParallelism(int *array, int *tmp, bool inplace, long begin, long end,int depth) {

	const int cutoff = 800;

	

	if (begin < (end - 1)) {
		const long half = (begin + end) / 2;

		if(end - begin <= cutoff || depth <= 0){
			MsParallelism(array, tmp, !inplace, begin, half,depth-1);
			MsParallelism(array, tmp, !inplace, half, end,depth-1);
		}
		else{
			#pragma omp task shared(array,tmp)
			{
				MsParallelism(array, tmp, !inplace, begin, half,depth-1);
			}
			#pragma omp task shared(array,tmp)
			{
				MsParallelism(array, tmp, !inplace, half, end,depth-1);
			}
			#pragma omp taskwait
		}
		
		if (inplace) {
			MsMergeSequential(array, tmp, begin, half, half, end, begin);
		} else {
			MsMergeSequential(tmp, array, begin, half, half, end, begin);
		}


	} else if (!inplace) {
		tmp[begin] = array[begin];
	}
	return;
}


/**
  * Serial MergeSort
  */
// TODO: this function should create the parallel region
// TODO: good point to compute a good depth level (cut-off)
void MsParallel(int *array, int *tmp, const size_t size) {

   // TODO: parallel version of MsSequential will receive one more parameter: 'depth' (used as cut-off)
	// MsSequential(array, tmp, true, 0, size);

	#pragma omp parallel num_threads(omp_get_max_threads())
	{
		#pragma omp single 
		{
			MsParallelism(array, tmp, true, 0, size,4);
		}
	}
}


/** 
  * @brief program entry point
  */
int main(int argc, char* argv[]) {
	// variables to measure the elapsed time
	struct timeval t1, t2, t3, t4;
	double etime1, etime2;

	// expect one command line arguments: array size
	if (argc != 2) {
		printf("Usage: MergeSort.exe <array size> \n");
		printf("\n");
		return EXIT_FAILURE;
	}
	else {
		const size_t stSize = strtol(argv[1], NULL, 10);
		int *data = (int*) malloc(stSize * sizeof(int));
		int *tmp = (int*) malloc(stSize * sizeof(int));
		int *ref = (int*) malloc(stSize * sizeof(int));

		//Initialization
		printf("Initialization...\n");
		srand(95);
		for (size_t idx = 0; idx < stSize; ++idx){
			data[idx] = (int) (stSize * (double(rand()) / RAND_MAX));
		}
		std::copy(data, data + stSize, ref);
		double dSize = (stSize * sizeof(int)) / 1024 / 1024;
		printf("Sorting %zu elements of type int (%f MiB)...\n", stSize, dSize);

		//operate by serial compute
		gettimeofday(&t1, NULL);
		MsSerial(data, tmp, stSize);
		gettimeofday(&t2, NULL);

		etime1 = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
		etime1 = etime1 / 1000;

		printf("done, took %f sec of serial \nVerification...", etime1);
		//Verification
		if (isSortedParallel(ref, data, stSize)) {
			printf(" successful.\n");
		}
		else {
			printf(" FAILED.\n");
		}

		//recreate data for parallel with same seed
		for (size_t idx = 0; idx < stSize; ++idx){
			data[idx] = (int) (stSize * (double(rand()) / RAND_MAX));
		}
		std::copy(data, data + stSize, ref);
		//operate by parallel compute
		gettimeofday(&t3, NULL);
		MsParallel(data, tmp, stSize);
		gettimeofday(&t4, NULL);

		etime2 = (t4.tv_sec - t3.tv_sec) * 1000 + (t4.tv_usec - t3.tv_usec) / 1000;
		etime2 = etime2 / 1000;
	
		printf("done, took %f sec of parrallel.\nVerification...", etime2);

		//Verification
		if (isSortedParallel(ref, data, stSize)) {
			printf(" successful.\n");
		}
		else {
			printf(" FAILED.\n");
		}

		free(data);
		free(tmp);
		free(ref);
	}
	return EXIT_SUCCESS;
}


