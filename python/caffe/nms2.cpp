#include <Windows.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>

//#pragma comment(lib, "cudaRT.lib")
#pragma comment(lib, "winmm.lib")

using namespace std;

typedef struct Rect_t {
	Rect_t(int x1_, int y1_, int x2_, int y2_, float score_) {
		x1 = x1_;
		y1 = y1_;
		x2 = x2_;
		y2 = y2_;
		score = score_;
	}
	int x1;
	int y1;
	int x2;
	int y2;
	float score;
} Rect;

void printHaha() {
	printf("Haha\n");
}

std::vector<int> nms(float* x1s, float* y1s, float* x2s, float* y2s, float* scores,
		int data_size, float thresh) {

	std::vector<int> keep;
	int keep_no = 0;
	int* suppressed = (int*)malloc(sizeof(int) * data_size);
	memset(suppressed, 0, sizeof(int)*data_size);
	
	for (int _i = 0; _i < data_size; _i++){
		//int i = order[_i];
		int i = _i;

		if (suppressed[i] == 1)
			continue;

		keep.push_back(i);
		keep_no++;
		
		/*
		printf("x1s[0] = %.3f\n", x1s[0]);
		printf("x1s[1] = %.3f\n", x1s[1]);
		printf("x1s[2] = %.3f\n", x1s[2]);
		
		printf("*(x1s + 0) = %.3f\n", *(x1s + 0));
		printf("*(x1s + 1) = %.3f\n", *(x1s + 1));
		printf("*(x1s + 2) = %.3f\n", *(x1s + 2));
		*/

		float ix1 = x1s[i];
		float iy1 = y1s[i];
		float ix2 = x2s[i];
		float iy2 = y2s[i];
		int iArea = (ix2 - ix1 + 1) * (iy2 - iy1 + 1);

		//printf("ix1=%d, iy1=%d, ix2=%d, iy2=%d, iArea=%d\n", ix1, iy1, ix2, iy2, iArea);

		for (int _j = i + 1; _j < data_size; _j++){
			//int j = order[_j];
			int j = _j;

			if (suppressed[j] == 1)
				continue;
			int jArea = (x2s[j] - x1s[j] + 1) * (y2s[j] - y1s[j] + 1);
			float xx1 = max(ix1, x1s[j]);
			float yy1 = max(iy1, y1s[j]);
			float xx2 = min(ix2, x2s[j]);
			float yy2 = min(iy2, y2s[j]);
			int w = max(0.0, xx2 - xx1 + 1);
			int h = max(0.0, yy2 - yy1 + 1);
			int inter = w * h;
			float ovr = (float)inter / float(iArea + jArea - inter);
			if (ovr >= thresh)
				suppressed[j] = 1;

			//printf("jx1=%d, jy1=%d, jx2=%d, jy2=%d, jArea=%d\n", jRect.x1, jRect.y1, jRect.x2, jRect.y2, jArea);
			//printf("xx1=%d, yy1=%d, xx2=%d, yy2=%d\n", xx1, yy1, xx2, yy2);

		}
	}

	free(suppressed);

	return keep;
}

int nms2_main(void) {
	DWORD dwTime = 0;
	float thresh = 0.2;
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

	dwTime = timeGetTime();

	std::vector<int> keep = nms(x1s.data(), y1s.data(), x2s.data(), y2s.data(), scores.data(), 
					thresh, data_size);

	dwTime = timeGetTime() - dwTime;

	//for (int i = 0; i < rect_size; i++)
	//	std::cout << "suppressed[" << i << "] : " << suppressed[i] << std::endl;

	//for (int i = 0; i < keep.size(); i++)
	//	std::cout << "keep[" << i << "] : " << keep[i] << std::endl;


	std::cout << "input rect : " << data_size << std::endl;
	std::cout << "keep size  : " << keep.size() << std::endl;
	std::cout << "time " << dwTime << "ms." << std::endl;

	return 0;
}
