#include <Windows.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>

#pragma comment(lib, "cudaRT.lib")
#pragma comment(lib, "winmm.lib")

using namespace std;

std::vector<int> nms_cpp(float* x1s, float* y1s, float* x2s, float* y2s, float* scores,
	int data_size, float thresh, int max_candidate);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ VOID SuppressFunc(float ix1, float iy1, float ix2, float iy2, 
			float* x1s, float* y1s, float* x2s, float* y2s, 
			int iArea, int base_index, int data_size, int* suppressed,
			float thresh) {

	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	int j = base_index + threadID;

	//printf ("data_size : %d\n", data_size);
	//printf ("base_index : %d, blockIdx.x : %d, blockDim.x : %d, threadIdx.x : %d, j : %d\n", base_index, blockIdx.x, blockDim.x, threadIdx.x, j);
	
	if (j < data_size) {
		//printf ("suppressed[j] : %d\n", suppressed[j]);
		if (suppressed[j] == 1)
			return;
		int jArea = (x2s[j] - x1s[j] + 1) * (y2s[j] - y1s[j] + 1);
		float xx1 = max(ix1, x1s[j]);
		float yy1 = max(iy1, y1s[j]);
		float xx2 = min(ix2, x2s[j]);
		float yy2 = min(iy2, y2s[j]);
		float w = max(0.0, xx2 - xx1 + 1);
		float h = max(0.0, yy2 - yy1 + 1);
		float inter = w * h;
		float ovr = (float)inter / float(iArea + jArea - inter);
		if (ovr >= thresh) {
			suppressed[j] = 1;
			//printf("%d is suppressed by %d\n", j, base_index-1);
		}
	}
}

std::vector<int> nms_cuda(float* x1s, float* y1s, float* x2s, float* y2s, float* scores,
		int data_size, float thresh, int max_candidate) {

	std::vector<int> keep;
	int keep_no = 0;
	int suppressed_size = sizeof(int) * data_size;
	int* suppressed = (int*)malloc(suppressed_size);
	memset(suppressed, 0, sizeof(int)*data_size);
	
	int points_size = data_size * sizeof(float);
	float* d_x1s, *d_y1s, *d_x2s, *d_y2s;
	int* d_suppressed;
	cudaMalloc((void **)&d_x1s, points_size);
	cudaMalloc((void **)&d_y1s, points_size);
	cudaMalloc((void **)&d_x2s, points_size);
	cudaMalloc((void **)&d_y2s, points_size);
	cudaMalloc((void **)&d_suppressed, sizeof(int)*data_size);

	cudaMemcpy(d_x1s, x1s, points_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1s, y1s, points_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2s, x2s, points_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2s, y2s, points_size, cudaMemcpyHostToDevice);
	
	for (int _i = 0; _i < data_size; _i++){
		int i = _i;

		if (suppressed[i] == 1)
			continue;

		keep.push_back(i);
		keep_no++;

		if (keep_no == max_candidate)
			break;

		float ix1 = x1s[i];
		float iy1 = y1s[i];
		float ix2 = x2s[i];
		float iy2 = y2s[i];
		int iArea = (ix2 - ix1 + 1) * (iy2 - iy1 + 1);

		cudaMemcpy(d_suppressed, suppressed, suppressed_size, cudaMemcpyHostToDevice);

		int block_size = 100;
		int thread_size = data_size / block_size - (i / block_size);
		if (data_size % block_size != 0)
			thread_size++;		 
		SuppressFunc <<<block_size, thread_size >>>(ix1, iy1, ix2, iy2, 
							d_x1s, d_y1s, d_x2s, d_y2s, 
							iArea, i+1, data_size, d_suppressed,
							thresh);

		cudaMemcpy(suppressed, d_suppressed, suppressed_size, cudaMemcpyDeviceToHost);
		
		//cudaDeviceSynchronize();
		//gpuErrchk( cudaPeekAtLastError() );
	}
	
	free(suppressed);

	cudaFree(d_x1s);
	cudaFree(d_y1s);
	cudaFree(d_x2s);
	cudaFree(d_y2s);
	cudaFree(d_suppressed);

	return keep;
}

int nms_cu_main(void) {
	DWORD dwTime = 0;
	float thresh = 0.2;
	int max_candidate = 123;
	std::vector<float> x1s;
	std::vector<float> y1s;
	std::vector<float> x2s;
	std::vector<float> y2s;
	std::vector<float> scores;

	for (int i = 0; i < 20000; i++) {
		float x1 = rand() % 100;
		float y1 = rand() % 100;
		float x2 = x1 + rand() % 500;
		float y2 = y1 + rand() % 500;
		float score = (float)(rand() % 100) / (float)100;;
		x1s.push_back(x1);
		y1s.push_back(y1);
		x2s.push_back(x2);
		y2s.push_back(y2);
		scores.push_back(score);
	}

	int data_size = x1s.size();

	std::cout << "input rect : " << data_size << std::endl;

	for (int i=0; i<50; i++) {
		dwTime = timeGetTime();

		std::vector<int> keep1 = nms_cpp(x1s.data(), y1s.data(), x2s.data(), y2s.data(), scores.data(), 
						data_size, thresh, max_candidate);

		dwTime = timeGetTime() - dwTime;
		std::cout << "keep size  : " << keep1.size() << std::endl;
		std::cout << "time " << dwTime << "ms." << std::endl;


		dwTime = timeGetTime();

		std::vector<int> keep2 = nms_cuda(x1s.data(), y1s.data(), x2s.data(), y2s.data(), scores.data(), 
						data_size, thresh, max_candidate);

		dwTime = timeGetTime() - dwTime;
		std::cout << "keep2 size  : " << keep2.size() << std::endl;
		std::cout << "time " << dwTime << "ms." << std::endl << std::endl;
	}

	//for (int i = 0; i < rect_size; i++)
	//	std::cout << "suppressed[" << i << "] : " << suppressed[i] << std::endl;

	//for (int i = 0; i < keep.size(); i++)
	//	std::cout << "keep[" << i << "] : " << keep[i] << std::endl;



	return 0;
}
