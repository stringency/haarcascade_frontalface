/*******************************************************
 > File Name: main.cpp
 > Author: admin
 > Mail: ffffffan@foxmail.com
 > Created Time: Fri 24 Apr 2020 09:46:00 CST
 > Modified Time:2020年07月01日 星期三 15时13分11秒
 > Note: No
*******************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <time.h>

using namespace std;

// OpenCV includes
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/face.hpp"

using namespace cv;
using namespace cv::ml;

cv::Ptr<cv::face::FaceRecognizer> recognizer;
CascadeClassifier faceCascade;
char g_moduleFlag = 0;
vector<string> g_ids;
size_t facesNum = 0;
int flag_facesNum = 0;
size_t labelCounts = 1;

Mat preProcessImage(Mat input);
Mat extractFace(Mat input, vector<Rect>* faces);
bool detection(void);
void init(void);
int videoProcess(void);
Mat testvideo(Mat img);
int trainOnce(Mat img);
int updateLabels(void);


//输出灰度图像，直方图均衡，长宽缩小0.5倍
Mat preProcessImage(Mat input)
{
	//cout<<"-->>function preprocessImage"<<endl;
	Mat result;

	resize(input, input, Size(), 0.5, 0.5, INTER_AREA);
	cvtColor(input, result, COLOR_BGR2GRAY);  //转变为灰度图像
	// Equalize the histogram
	equalizeHist(result, result);  //转为均衡化直方图
	return result;
}

//仅支持单脸
//未检测到脸时返回纯黑
//返回灰度脸图，缩小了1.1倍
Mat extractFace(Mat input, vector<Rect>* faces)
{
	//cout<<"-->>function extractFace"<<endl;
	Mat result = preProcessImage(input);

	Mat zero = Mat::zeros(result.rows, result.cols, CV_8UC1);

	//识别器参数
	faceCascade.detectMultiScale(result, *faces, \
		1.1, 2, 0 | CV_HAL_CMP_GE, Size(20, 20), Size(100, 100));

	if ((*faces).size() == 0)
		return zero;

	//Mat grayc = result.clone();
	//grayc = grayc((*faces)[0]);//

	Rect faceRect = (*faces)[0];
	Mat grayc = result(faceRect);



	return grayc;
}

//检测人脸
Mat detection(Mat img, size_t& facesNum)
{
	//cout<<"-->>function detection"<<endl;
	vector<Rect> faces;		//用于存储人脸位置信息的向量

	Mat grayt = extractFace(img, &faces);

	Scalar he = sum(grayt);	//查看是否没有人脸

	resize(img, img, Size(), 0.5, 0.5, INTER_AREA); //将图片缩小一半
	if (he[0] == 0)
	{
		return img;
	}
	//resize(img, img, Size(), 0.5, 0.5, INTER_AREA);

	for (size_t i = 0; i < faces.size(); ++i) {
		// 处理第 i 张人脸，faces[i] 包含了当前人脸的位置信息
		// ...
		rectangle(img, faces[i], Scalar(0, 0, 255), 2);
	}
	if (facesNum != faces.size())
	{
		facesNum = faces.size();
		flag_facesNum = 1;
	}

	if (flag_facesNum == 1)
	{
		cout << "脸的数量：" << facesNum << endl;
		flag_facesNum = 0;
	}

	//rectangle(img, faces[0], Scalar(0, 0, 255), 2);

	return img;
}


int updateLabels(void)
{
	cout << "-->>function updateLabels" << endl;

	int inputlabel;

	cout << "请输入标签号：" << endl;
	//cin >> inputlabel;
	//cin.ignore();
	////g_ids.push_back(to_string(inputlabel));
	inputlabel = labelCounts++;
	cout << inputlabel << endl;

	cout << "-->>function updateLabels end" << endl;
	return inputlabel;
}

int trainOnce(Mat img)
{
	cout << "-->>function trainOnce" << endl;

	std::vector<cv::Mat> referenceImages;	//存放人脸信息
	std::vector<int> labels;	//存放对应的人脸信息标签
	// vectors of reference image and their labels

	vector<Rect> faces;
	Mat frame;
	int inputlabel;

	frame = extractFace(img, &faces);
	Scalar he = sum(frame);
	if (he[0] == 0)
	{
		cout << " No face" << endl;
		return -1;
	}



	inputlabel = updateLabels(); //手动输入标签
	referenceImages.push_back(frame);
	labels.push_back(inputlabel);

	if (g_moduleFlag == 0)//是否有模型
	{
		recognizer->train(referenceImages, labels);
		cout << "g_moduleFlag == 0!" << endl;
		g_moduleFlag = 1;
	}
	else
		recognizer->update(referenceImages, labels);

	cout << "trained!" << endl;

	recognizer->write("./trainer.yml");

	return 0;
}

//摄像头测试
Mat testvideo(Mat img)
{
	//cout<<"-->>function testvideo"<<endl;
	int predictedLabel = -1;
	double confidence = 0.0;

	vector<Rect> faces;
	Mat zero = Mat::zeros(img.rows, img.cols, CV_8UC1);

	if (g_moduleFlag == 0)
	{
		cout << "未训练，无法测试" << endl;
		return zero;
	}

	Mat grayt = extractFace(img, &faces);

	Scalar he = sum(grayt);
	if (he[0] == 0)
	{
		//cout<<"图片中无人脸"<<endl;
		resize(img, img, Size(), 0.5, 0.5, INTER_AREA);
		return img;
	}

	resize(img, img, Size(), 0.5, 0.5, INTER_AREA);

	//rectangle(img, faces[0], Scalar(255, 255, 255), 2);

	for (size_t i = 0; i < faces.size(); ++i) {
		// 处理第 i 张人脸，faces[i] 包含了当前人脸的位置信息
		// ...
		rectangle(img, faces[i], Scalar(255, 255, 255), 2);

		// predict the label of this image
		recognizer->predict(grayt,     // face image 
			predictedLabel, // predicted label of this image 
			confidence);    // confidence of the prediction

		string name;

		if (confidence < 80)
		{
			name = to_string(predictedLabel);
		}
		else {
			name = g_ids[0];
			confidence = 0;
		}

		stringstream ss;
		ss << (int)confidence;
		putText(img, ss.str().c_str(), Point2d(faces[i].x, faces[i].y + 30), \
			FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);

		putText(img, name, Point2d(faces[i].x, faces[i].y + faces[i].height), \
			FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, LINE_AA);
	}

	if (facesNum != faces.size())
	{
		facesNum = faces.size();
		flag_facesNum = 1;
	}

	if (flag_facesNum == 1)
	{
		cout << "脸的数量：" << facesNum << endl;
		flag_facesNum = 0;
	}



	return img;
}

int videoProcess(void)
{
	cout << "-->>function videoProcess" << endl;

	Mat camera;
	vector<Rect> faces;
	char ch = 48;	//相机功能
	char sh = ch;	//目前相机状态
	int count = 0;

	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cout << "No video!" << endl;
		return -1;
	}

	cout << "请在图片中选择摄像头模式：" << endl;
	cout << "	0 显示视频" << endl;
	cout << "	1 训练人脸" << endl;
	cout << "	2 持续测试人脸" << endl;


	while (true)
	{
		cap >> camera;
		flip(camera, camera, 1); //水平翻转图像

		if (ch == 48) {//0	普通加载摄像头
			camera = detection(camera, facesNum);  //识别图像并且框出打印人脸数量
			//resize(camera, camera, Size(), 0.5, 0.5, INTER_AREA);
			imshow("Camera", camera);
			sh = ch;
			ch = waitKey(1);
		}
		else if (ch == 48 + 1) {//1	训练人脸
			trainOnce(camera);
			resize(camera, camera, Size(), 0.5, 0.5, INTER_AREA);
			imshow("Camera", camera);
			ch = 48 + 2;
		}
		else if (ch == 48 + 2) {//2		持续测试人脸
			camera = testvideo(camera);

			Scalar he = sum(camera);
			if (he[0] == 0)	//人脸未训练
				ch = 48;
			else {
				imshow("Camera", camera);
				sh = ch;
				ch = waitKey(1);
			}
		}
		else if (ch == 27)//esc
			break;
		else
			ch = sh;
	}
	cap.release();

	return 0;
}

void init(void)
{
	cout << "-->>function init" << endl;
	string faceCascadeName = "./haarcascade_frontalface_alt2.xml";
	if (!faceCascade.load(faceCascadeName))
	{
		cerr << "Error loading cascade file. Exiting!" << endl;
		exit(0);
	}

	g_ids.push_back("Unknown"); //将字符串放到g_ids的末尾，是一个标记
	recognizer = cv::face::LBPHFaceRecognizer::create(1, 8, 8, 8, 200.); //创建一个LBPH（Local Binary Patterns Histograms）人脸识别器对象
}

int main(int argc, char* argv[])
{
	cout << "-->>function main" << endl;
	Mat camera;

	init();
	videoProcess();

	destroyAllWindows();
	cout << "End of program" << endl;
	return 0;
}


