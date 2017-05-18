/**
* @file Morphology_1.cpp
* @brief Erosion and Dilation sample code
* @author OpenCV team
*/

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <tchar.h>
#include <chrono>
#include <math.h>
#include <thread>
#include <mutex>

#define PI 3.14159265358979

using namespace cv;
using std::mutex;
using std::vector;
using namespace std::chrono;

double calcdistance(Point val1, Point val2);
double calcdist(Point val1, Point val2);
double calculatedistance(float tvec[3]);
void calculatepose(Point2f coords[4], float tvec[3], float rvec[3]);
void getEulerAngles(Mat &rotCamerMatrix, Vec3d &eulerAngles);

milliseconds ms1 = duration_cast< milliseconds >(
	system_clock::now().time_since_epoch()
	);

milliseconds ms2 = duration_cast< milliseconds >(
	system_clock::now().time_since_epoch()
	);

mutex m;
mutex mp;
int detections = 0;

int process_image(Mat frame) {
	printf("\n------Beginning processing------\n");
	for (;;) {
		vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(95);
		mp.lock();
		imwrite("E:\\img.jpg", frame, compression_params);
		mp.unlock();
		const size_t stringSize = 1000;

		STARTUPINFO si;
		PROCESS_INFORMATION pi;
		DWORD exit_code;
		//char commandLine[stringSize] = "C:\\Apriltags\\apriltag_demo.exe C:\\Apriltags\\unknowntag.jpg";
		char commandLine[stringSize] = "E:\\process.bat";
		WCHAR wCommandLine[stringSize];
		mbstowcs(wCommandLine, commandLine, stringSize);

		ZeroMemory(&si, sizeof(si));
		si.cb = sizeof(si);
		ZeroMemory(&pi, sizeof(pi));
		m.lock();
		// Start the child process. 
		if (!CreateProcess(NULL,   // No module name (use command line)
			wCommandLine,   // Command line
			NULL,           // Process handle not inheritable
			NULL,           // Thread handle not inheritable
			FALSE,          // Set handle inheritance to FALSE
			0,              // No creation flags
			NULL,           // Use parent's environment block
			NULL,           // Use parent's starting directory 
			&si,            // Pointer to STARTUPINFO structure
			&pi)           // Pointer to PROCESS_INFORMATION structure
			)
		{
			printf("CreateProcess failed (%d).\n", GetLastError());
			return -1;
		}

		// Wait until child process exits.
		WaitForSingleObject(pi.hProcess, INFINITE);

		GetExitCodeProcess(pi.hProcess, &exit_code);
		printf("the execution of: \"%s\"\nreturns: %d\n", commandLine, exit_code - 242);
		m.unlock();

		// Close process and thread handles. 
		CloseHandle(pi.hProcess);
		CloseHandle(pi.hThread);
		//prms.set_value(exit_code-242);
		//return exit_code - 242;
		//detections = exit_code - 242;
		printf("\n222222222222222222222|detections: %d\n", detections);
	}
}

int main(int, char**) {
	VideoCapture cap(0);//Open default cam
	if (!cap.isOpened())
		return -1;

	Mat edges;
	namedWindow("edges", 1);


	Mat frame;
	Mat display;
	//cap.read(frame);
	cap >> frame;
	frame.copyTo(display);
	std::thread t1(process_image, display);
	float val[24] = {};
	float tvec[3] = { 1, 1, 1 };
	float rvec[3] = { 1, 1, 1 };
	int tags[8];
	float tagtvec[8][3];
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 3; j++)
			tagtvec[i][j] = 0;
	for (;;)
	{
		//cap.read(frame);
		cap >> frame; // get a new frame from camera
		mp.lock();
		frame.copyTo(display);
		mp.unlock();
		if (system(NULL)) puts("Ok");
		else exit(EXIT_FAILURE);

		if (m.try_lock()) {
			//cvtColor(frame, edges, CV_BGR2GRAY);
			//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
			//Canny(edges, edges, 0, 30, 3);
			FILE *F = fopen("E:\\coordapril.txt", "r");
			char buff[255];
			fgets(buff, 255, F);
			detections = atoi(buff);
			printf("\n999999999999999999999999999999999999|detections: %d\n", detections);
			for (int i = 0; i < detections; i++) {
				fgets(buff, 255, F);
				tags[i] = atoi(buff);
				for (int j = 0; j < 8; j++) {
					fgets(buff, 255, F);
					val[j + 8 * i] = ceil(atof(buff));
				}
			}
			m.unlock();
		}

		for (int i = 0; i < detections; i++) {
			Point2f coords[4];
			coords[0] = Point2f(val[4 + i * 8], val[5 + i * 8]);
			coords[1] = Point2f(val[6 + i * 8], val[7 + i * 8]);
			coords[2] = Point2f(val[0 + i * 8], val[1 + i * 8]);
			coords[3] = Point2f(val[2 + i * 8], val[3 + i * 8]);
			calculatepose(coords, tvec, rvec);
			line(frame, Point(val[0 + i * 8], val[1 + i * 8]), Point(val[2 + i * 8], val[3 + i * 8]), (0, 0, 255), 3, 8, 0);
			line(frame, Point(val[2 + i * 8], val[3 + i * 8]), Point(val[4 + i * 8], val[5 + i * 8]), CV_RGB(255, 0, 255), 3, 8, 0);
			line(frame, Point(val[4 + i * 8], val[5 + i * 8]), Point(val[6 + i * 8], val[7 + i * 8]), CV_RGB(0, 255, 0), 3, 8, 0);
			line(frame, Point(val[6 + i * 8], val[7 + i * 8]), Point(val[0 + i * 8], val[1 + i * 8]), CV_RGB(0, 255, 255), 3, 8, 0);
			line(frame, Point((val[0 + i * 8] + val[2 + i * 8]) / 2, (val[1 + i * 8] + val[3 + i * 8]) / 2), Point((val[4 + i * 8] + val[6 + i * 8]) / 2, (val[5 + i * 8] + val[7 + i * 8]) / 2), CV_RGB(0, 0, 255), 3, 8, 0);
			//double dist = calcdistance(Point((val[0] + val[2]) / 2, (val[1] + val[3]) / 2), Point((val[4] + val[6]) / 2, (val[5] + val[7]) / 2));

			line(frame, Point((val[4 + i * 8] + val[6 + i * 8]) / 2, (val[5 + i * 8] + val[7 + i * 8]) / 2), Point(val[6 + i * 8], val[7 + i * 8]), CV_RGB(255, 255, 255), 2, 8, 0);
			line(frame, Point((val[4 + i * 8] + val[6 + i * 8]) / 2, (val[5 + i * 8] + val[7 + i * 8]) / 2), Point(val[6 + i * 8], (val[5 + i * 8] + val[7 + i * 8]) / 2), CV_RGB(255, 0, 255), 2, 8, 0);
			//double angle = acos(calcdist(Point((val[4] + val[6]) / 2, (val[5] + val[7]) / 2), Point(val[6], (val[5] + val[7]) / 2)) / calcdist(Point((val[4] + val[6]) / 2, (val[5] + val[7]) / 2), Point(val[6], val[7])));
			//angle = (8 / 3)*angle * 180 / PI;

			float rvect[3] = { 1,1,1 };
			for (int j = 0; j < 3; j++) {
				rvect[j] = rvec[j];
			}
			Mat rmat = Mat(3, 3, CV_32FC1);
			Mat rmato = Mat(3, 1, CV_32FC1, rvec);
			Rodrigues(rmato, rmat);
			Vec3d eulerAngles;
			//getEulerAngles(rmat, eulerAngles);
			Mat translate = Mat(3, 1, CV_32FC1, tvec);
			rmat = rmat.t();
			Mat rotm = Mat(1, 3, CV_32FC1, rmat.at<float>(0));
			Mat transm = Mat(1, 3, CV_32FC1, tvec);
			rotm = rotm.t();
			Mat mat3x3 = -rotm* transm;
			translate = -rmat * translate;
			printf("\nRVEC|pppppppppppppppppppppppppppppppppppp| %f | %f | %f |\n", rvect[0], rvect[1], rvect[2]);
			printf("\nTVEC|____________________________________| %f | %f | %f |\n", tvec[0], tvec[1], tvec[2]);
			for (int d = 0; d < 3; d++) {
				printf("\nCOMM|wwwwwwwwwwwwwwwwwwwwww| %f | %f | %f |\n", mat3x3.at<float>(d, 0), mat3x3.at<float>(d, 1), mat3x3.at<float>(d, 2));
			}
			//printf("\n8888888888888888888888888888888888888888888888888888888| %f\n", eulerAngles[0]);
			/*float rotation[3] = { 1,1,1 };
			for (int j = 0; j < 3; j++) {
			rotation[i] = rmat.at<float>(0, i);
			printf("\n[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[| %f |n", rotation[i]);
			}*/
			for (int j = 0; j < 3; j++) {
				printf("\nRodrigues matrix| ~~~~~~~~~~~~~~~~| %f | %f | %f |\n", rmat.at<float>(j, 0), rmat.at<float>(j, 1), rmat.at<float>(j, 2));
			}/*
			 printf("\n,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,| %f |,,,,,,,| %f |", rotation[0], tvec[0]);
			 printf("\n,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,| %f |,,,,,,,| %f |", rotation[1], tvec[1]);
			 printf("\n,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,| %f |,,,,,,,| %f |", rotation[2], tvec[2]);
			 float position[3];
			 for (int j = 0; j < 3; j++) {
			 position[j] = rotation[j] * tvec[j];
			 printf("\n,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,| %f |,,,,,,,| %f |", rotation[j], tvec[j]);
			 }*/
			for (int j = 0; j < 3; j++) {
				rvec[j] = rvect[j];
			}
			float tvec2[3];
			for (int j = 0; j < 3; j++) {
				tvec2[j] = translate.at<float>(j);
				printf("\n8888888888888888888888888888888888888888888888888888888| %f\n", translate.at<float>(j));
			}
			double dist = calculatedistance(tvec2);
			String disp = std::to_string(dist) + "Meters";
			putText(frame, disp, Point(val[6 + i * 8], val[7 + i * 8] - 5), FONT_HERSHEY_PLAIN, 5, CV_RGB(255, 0, 0));
			float unit[3] = { 1,0,0 };
			Mat unitMat = Mat(3, 1, CV_32FC1, unit);
			Mat transformed = rmat*unitMat;
			//double angle = acos(rmat.at<float>(0,0));
			//angle = angle* 180 / PI;
			double angle = atan(transformed.at<float>(2) / transformed.at<float>(0)) * 180 / PI;
			printf("\nangle|| %f ||\n", atan(rmat.at<float>(0, 1) / rmat.at<float>(0, 0)) * 180 / PI);
			printf("\nACos(angle)|| %f || %f || %f ||\n", transformed.at<float>(0), transformed.at<float>(1), transformed.at<float>(2));
			printf("\nAngle2|| %f ||\n", atan(transformed.at<float>(2) / transformed.at<float>(0)) * 180 / PI);
			std::string anglep = std::to_string(angle) + "deg";
			putText(frame, anglep, Point(val[2 + i * 8] + 5, val[3 + i * 8]), FONT_HERSHEY_PLAIN, 5, CV_RGB(255, 0, 0));
			//double Camloc[3];
			//for (int g = 0; g < 3; g++) {
			//	int weightedsum;
			//}
			printf("\nTAG DETECTED|| %d\n", tags[i]);
			for (int j = 0; j < 3; j++) {
				tagtvec[tags[i]][j] = tvec[j];
			}
		}
		for (int i = 0; i < 8; i++) {
			printf("\ntotal tvecs| %f | %f | %f |\n", tagtvec[i][0], tagtvec[i][1], tagtvec[i][2]);
		}
		float tagvecLC[8][3];
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 3; j++)
				tagvecLC[i][j] = tagtvec[i][j];
		tagvecLC[0][0] += .6525;
		tagvecLC[4][0] += .26;
		tagvecLC[4][1] += -.645;
		tagvecLC[4][2] += .58;
		if (detections > 0) {
			float avgtvec[3];
			for (int i = 0; i < 3; i++) {
				float runningsum = 0;
				for (int j = 0; j < 8; j++)
					runningsum += tagvecLC[j][i];
				avgtvec[i] = runningsum / detections;
			}
			printf("\nAVERAGEVECTOR| %f | %f | %f |\n", avgtvec[0], avgtvec[1], avgtvec[2]);
		}
		for (int i = 0; i < 8; i++) {
			printf("\nLC tvecs| %f | %f | %f |\n", tagvecLC[i][0], tagvecLC[i][1], tagvecLC[i][2]);
		}

		ms2 = duration_cast< milliseconds >(
			system_clock::now().time_since_epoch()
			);
		printf("\n====================\n%d\n=================", ms2 - ms1);
		imshow("edges", frame);
		ms1 = duration_cast< milliseconds >(
			system_clock::now().time_since_epoch()
			);
		if (waitKey(30) >= 0) break;
	}
	t1.join();
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}


double calcdistance(Point val1, Point val2) {
	double vala = (pow((val1.x - val2.x), 2) + pow((val1.y - val2.y), 2));
	vala = sqrt(vala);
	double val = vala;
	double returnval = .33 * 359 / val;
	return returnval;
}

double calcdist(Point val1, Point val2) {
	double val = (pow((val1.x - val2.x), 2) + pow((val1.y - val2.y), 2));
	val = sqrt(val);
	return val;
}

double calculatedistance(float tvec[3]) {
	return sqrt(pow(tvec[0], 2) + pow(tvec[1], 2) + pow(tvec[2], 2));
}

void calculatepose(Point2f coords[4], float tvec[3], float rvec[3]) {
	float twM = .17414;//tag width in meters
	float cameramatrix[3][3];
	cameramatrix[0][0] = 634.18497;
	cameramatrix[1][0] = 0;
	cameramatrix[2][0] = 0;
	cameramatrix[0][1] = 0;
	cameramatrix[1][1] = 637.61085;
	cameramatrix[2][1] = 0;
	cameramatrix[0][2] = 330.96758;
	cameramatrix[1][2] = 239.64850;
	cameramatrix[2][2] = 1;
	float distort[5] = { -.04388406, 1.528083,  -.004284626, -.003920457, -5.734556 };
	//float objpoints[4][3];
	vector<Point2f> imagepoint;
	for (int i = 0; i < 4; i++) {
		imagepoint.push_back(Point2f(coords[i].x, coords[i].y));
		printf("\nIMAGE POINTS^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^| (%f,%f)\n", coords[i].x, coords[i].y);
	}
	/*float imagepoints[4][2];
	for (int i = 0; i < 4; i++) {
	imagepoints[i][1] = coords[i].x;
	imagepoints[i][0] = coords[i].y;
	}
	objpoints[0][0] = 0;
	objpoints[0][1] = 0;
	objpoints[0][2] = 0;
	objpoints[1][0] = 0;
	objpoints[1][1] = -tagwidthM;
	objpoints[1][2] = 0;
	objpoints[2][0] = 0;
	objpoints[2][1] = -tagwidthM;
	objpoints[2][2] = -tagwidthM;
	objpoints[3][0] = 0;
	objpoints[3][1] = 0;
	objpoints[3][2] = -tagwidthM;*/
	vector<Point3f> objectpoints;
	objectpoints.push_back(Point3f(twM / 2, twM / 2, 0));
	objectpoints.push_back(Point3f(-twM / 2, twM / 2, 0));
	objectpoints.push_back(Point3f(-twM / 2, -twM / 2, 0));
	objectpoints.push_back(Point3f(twM / 2, -twM / 2, 0));
	for (int s = 0; s < 4; s++) {
		printf("\nObjectPoints|______________________________| %f | %f | %f |\n", objectpoints.at(s).x, objectpoints.at(s).y, objectpoints.at(s).z);
	}
	//Mat objinarr = Mat(4, 3, CV_32FC1, objpoints);
	//Mat imginarr = Mat(4, 2, CV_32FC1, imagepoints);
	Mat caminarr = Mat(3, 3, CV_32FC1, cameramatrix);
	Mat distinarr = Mat(1, 5, CV_32FC1, distort);
	Mat rvecoutarr = Mat(3, 1, CV_32FC1, rvec);
	Mat tvecoutarr = Mat(3, 1, CV_32FC1, tvec);
	solvePnP(objectpoints, imagepoint, caminarr, distinarr, rvecoutarr, tvecoutarr);
	printf("\nMatrix type|%d", tvecoutarr.type());
	for (int i = 0; i < 3; i++) {
		tvec[i] = tvecoutarr.at<double>(i);
		printf("\nTVEC ORIG????????????????????????????????????????????????????| %f\n", tvec[i]);
	}
	for (int i = 0; i < 3; i++) {
		rvec[i] = rvecoutarr.at<double>(i);
	}
	/*printf("\n%f|33333333333333333333333333333333333333333\n", tvec[0]);
	printf("\n%f|33333333333333333333333333333333333333333\n", tvec[1]);
	printf("\n%f|33333333333333333333333333333333333333333\n", tvec[2]);
	printf("\n%f|33333333333333333333333333333333333333333\n", tvecoutarr.at<float>(0));
	printf("\n%f|33333333333333333333333333333333333333333\n", tvecoutarr.at<float>(1));
	printf("\n%f|33333333333333333333333333333333333333333\n", tvecoutarr.at<float>(2));
	/*for (int i = 0; i < 3; i++) {
	rvec[i] = rvecoutarr.at<float>(i);
	}*/
}
void getEulerAngles(Mat &rotCamerMatrix, Vec3d &eulerAngles) {

	Mat cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ;
	double* _r = rotCamerMatrix.ptr<double>();
	double projMatrix[12] = { _r[0],_r[1],_r[2],0,
		_r[3],_r[4],_r[5],0,
		_r[6],_r[7],_r[8],0 };

	decomposeProjectionMatrix(Mat(3, 4, CV_64FC1, projMatrix),
		cameraMatrix,
		rotMatrix,
		transVect,
		rotMatrixX,
		rotMatrixY,
		rotMatrixZ,
		eulerAngles);
}
