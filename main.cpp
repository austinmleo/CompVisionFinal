//==========================================
//	Austin Leo and Stacia Near
//	Computer Vision, Fall 2017
//	Final Project
//
//==========================================

// This code uses the template from Lab08.

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <windows.h> // For Sleep()
//#include "stdafx.h"
#include <sapi.h>

// Length of one side of a square marker.
const float markerLength = 2.0;

using namespace cv;
using namespace std;

//Global flags
bool useAruco = true;
bool doImageWrite = false;
bool waitForUser = true;
bool useCamera = false;
bool drawMarkers = false;

Mat readImage(String filename) {
	Mat image = imread(filename, CV_LOAD_IMAGE_UNCHANGED);

	if (image.empty())
	{
		cout << "Image not loaded";
		exit(1);
	}
	else {
		return image;
	}
}

void showImage(Mat image, String windowname) {
	namedWindow(windowname, CV_WINDOW_AUTOSIZE);
	imshow(windowname, image);
	if (waitForUser) waitKey(0);
}

void showImageResize(Mat image, String windowname) {
	namedWindow(windowname, CV_WINDOW_NORMAL);
	//resizeWindow(windowname, image.size().width*scale, image.size().height*scale);
	imshow(windowname, image);
	if (waitForUser) waitKey(0);
}

bool detectAruco(Mat image, vector<Point2f>& markerCornersSingle) {
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);
	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

	vector< int > markerIds;
	vector< vector<Point2f> > markerCorners, rejectedCandidates;

	aruco::detectMarkers(
		image, // input image
		dictionary, // type of markers that will be searched for
		markerCorners, // output vector of marker corners
		markerIds, // detected marker IDs
		detectorParams, // algorithm parameters
		rejectedCandidates);
	if (markerIds.size() > 0) {
		// Draw all detected markers.
		if (drawMarkers) {

			aruco::drawDetectedMarkers(image, markerCorners, markerIds);


			for (unsigned int i = 0; i < (int)markerCorners[0].size(); i++) {
				int fontFace = 1;
				double fontScale = 5;
				Scalar color = 255;
				putText(image, to_string(i + 1), markerCorners[0][i], fontFace, fontScale, color, 3);
			}
		}

		//only one aruco marker should be found
		markerCornersSingle = markerCorners[0];
		return true; //success
	}
	return false;
}

void drawBoundingRect(Mat image, vector<Rect> boundRect) {
	RNG rng(12345);

	//Mat drawing = Mat::zeros(binaryImage.size(), CV_8UC3);
	int bound = boundRect.size();
	for (int i = 0; i< bound; i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(drawing, contours_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(image, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		putText(image, to_string(i + 1), boundRect[i].tl(), 1, 2, color, 2);
		//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
	}
	showImageResize(image, "Bounding boxes.");
}

//Sources: https://stackoverflow.com/questions/25529646/sorting-vectorpoints-from-top-to-bottom-then-from-left-to-right-using-stl-in-c
//http://answers.opencv.org/question/31515/sorting-contours-from-left-to-right-and-top-to-bottom/
//https://stackoverflow.com/questions/29630052/ordering-coordinates-from-top-left-to-bottom-right

struct text_order_sorter
{
	bool operator ()(const Rect ra, const Rect rb) {
		//get the top left point of each rectangle
		//the first step is to "warp" the points only during the sort
		//so that they fall into matching gridlines
		//by dividing by a constant to make the number smaller,
		//then rounding it, then multiplying it back up,
		//it will normalize the coordinates to fall into an even grid
		//after that we can employ a simple method to sort by y then x
		int m = 90;
		int n = 90;
		int xa = int(ra.tl().x / n)*n;
		int ya = int(ra.tl().y / m)*m;
		int xb = int(rb.tl().x / n)*n;
		int yb = int(rb.tl().y / m)*m;

		// scale factor for y should be larger than img.width
		//this scale factor allows the different axes to order points correctly
		int scale = 2000;

		return ((xa + scale * ya) > (xb + scale * yb));
	}
};

vector<Rect> sortBoundingRect(vector<Rect> boundRect) {
	//vector<Rect> sortedRect(boundRect.size());
	//margin of error, other letters must be +-margin close to the "y" that this line of letters are located on
	//int marginY = 5;
	sort(boundRect.begin(), boundRect.end(), text_order_sorter());

	//this sort returns the vector in exactly opposite order
	//flip to make it match text order
	reverse(boundRect.begin(), boundRect.end());
	
	return boundRect;

}


// get bounds of letters
void getBoundingRect(Mat templateImage, vector<Rect>& returnRect) {

	int arucoMinX;
	int arucoMaxX;
	int arucoMinY;
	int arucoMaxY;
	vector<Point2f> markerCorners;

	if (useAruco) {
		//Find aruco marker
		detectAruco(templateImage, markerCorners);
		//top left, bottom right
		//Need to know this so any contours within aruco marker region
		//are not detected, only bounding boxes of letters are detected
		Point tl = markerCorners[1];
		Point br = markerCorners[3];
		//x increases in a counterintuitive direction
		arucoMinX = br.x;
		arucoMaxX = tl.x;
		arucoMinY = tl.y;
		arucoMaxY = br.y;
	}

	Mat binaryImage;
	Mat gray;
	int thresh = 0;
	if (useAruco) thresh = 110;
	int const max_BINARY_value = 255;
	int threshold_type = THRESH_BINARY;


	cvtColor(templateImage, gray, CV_BGR2GRAY);
	threshold(gray, binaryImage, thresh, max_BINARY_value, threshold_type);

	//imshow("Binary Image", binaryImage);

	//Find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//binaryImage, input image (is destroyed)
	findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	Rect bigRect;
	//check all black areas to find biggest
	//so we know where aruco marker is
	//if aruco is not enabled
	if (!useAruco) {
		vector<Point> contour_poly;
		int maxArea = 0;
		int maxAreaIndex = 0;

		for (unsigned int i = 0; i < (int)contours.size(); i++) {
			if (contourArea(contours[i], true) > 0) {
				//cout << "black" << endl;
				int area = contourArea(contours[i]);
				if (area > maxArea) {
					maxArea = area;
					maxAreaIndex = i;
				}
			}
			else {
				//cout << "white" << endl;
				//ignore white
				continue;
			}
		}

		approxPolyDP(Mat(contours[maxAreaIndex]), contour_poly, 3, true);
		bigRect = boundingRect(Mat(contour_poly));

		arucoMinX = bigRect.tl().x;
		arucoMaxX = arucoMinX + bigRect.width;
		arucoMinY = bigRect.tl().y;
		arucoMaxY = arucoMinY + bigRect.height;

		Mat biggestRegion = templateImage.clone();
		drawBoundingRect(biggestRegion, { bigRect });
	}

	
	//Source for below code:
	//https://docs.opencv.org/3.3.0/da/d0c/tutorial_bounding_rects_circles.html

	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect; //last is bounding rect of aruco mark
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	Mat coloredBounds = templateImage.clone();


	for (unsigned int i = 0; i < (int)contours.size(); i++) {

		if (contourArea(contours[i], true) > 0) {
			//cout << "black" << endl;
			//check to make sure its not part of the aruco marker
			//to do that, we must know the centroid of current contour
			//Sources: https://docs.opencv.org/3.3.1/dd/d49/tutorial_py_contour_features.html
			//https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html

			Moments M = moments(contours[i]);
			int cx = int(M.m10 / M.m00);
			int cy = int(M.m01 / M.m00);
			Point centroid;
			centroid.x = cx;
			centroid.y = cy;

			//ignore anything within the aruco marker's bounds
			//this should also work if just using the largest region to decide which is the aruco marker
			if ((cx > arucoMinX && cx < arucoMaxX) && (cy > arucoMinY && cy < arucoMaxY)) {
				continue;
			}

			drawMarker(coloredBounds, centroid, 255);
		}
		else {
			//cout << "white" << endl;
			//ignore white
			continue;
		}

		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect.push_back(boundingRect(Mat(contours_poly[i])));
		//minEnclosingCircle(contours_poly[i], center[i], radius[i]);
	}


	if (useAruco) {
		//Insert aruco marker as #27 (index 26)
		boundRect.push_back(boundingRect(markerCorners));
	}
	else {
		boundRect.push_back(bigRect);
	}

	drawBoundingRect(coloredBounds, boundRect);
	vector<Rect> sortedRect = sortBoundingRect(boundRect);

	Mat sortedBounds = templateImage.clone();
	drawBoundingRect(sortedBounds, sortedRect);

	//return sortedRect;
	returnRect = sortedRect;
}


void showImageVector(vector<Mat>& imageVec) {
	int letter = 65;

	for (int i = 0; i < (int)imageVec.size(); i++) {
		string charStr;
		charStr = (char)letter;
		showImage(imageVec[i], "Cropped " + charStr);
		letter++;
	}
	
}

//Source: https://stackoverflow.com/questions/14365411/opencv-crop-image

void cropLetters(Mat image, vector<Rect> boundRect, vector<Mat>& letters) {
	//65 = A, 90 = Z
	int letter = 65;

	for (int i = 0; i < (int)boundRect.size(); i++) {
		//int startX = 200, startY = 200, width = 100, height = 100;
		int whitespaceY = 4;
		int whitespaceX = 8;
		int startX = boundRect[i].tl().x - whitespaceX;
		int startY = boundRect[i].tl().y - whitespaceY;
		int width = boundRect[i].width + whitespaceX * 2;
		int height = boundRect[i].height + whitespaceY * 2;

		Mat cropPixels(image, Rect(startX, startY, width, height));
		Mat croppedImage;
		// Copy the data into new matrix
		cropPixels.copyTo(croppedImage);

		//add to image array
		letters.push_back(croppedImage);

		string charStr;
		charStr = (char)letter;

		letter++;
	}
}

void initializeTemplates(Mat templateImage, vector<Mat>& letters) {
	vector<Rect> boundRect;
	getBoundingRect(templateImage, boundRect);
	cropLetters(templateImage, boundRect, letters);
	//showImageVector(letters);
}

void erodeDilate(Mat& image, int erosion_size) {

	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));

	Mat output;

	erode(image, image, element);
	dilate(image, image, element);
}

bool transformImage(Mat image, Mat& transformedImage) {
	int erosion_size = 3;

	vector<Point2f> detectedCorners;

	vector<Point2f> correspondingCorners = { Point2f(200.0, 20.0), Point2f(560.0, 20.0), Point2f(560.0, 380.0), Point2f(200.0, 380.0) };
	showImageResize(image, "detectAruco");
	if (detectAruco(image, detectedCorners)) {

		Mat lambda = Mat::zeros(image.rows, image.cols, image.type());
		lambda = findHomography(detectedCorners, correspondingCorners);
		warpPerspective(image, transformedImage, lambda, Size(793, 1122));

		Mat grayScale, bin;
		cvtColor(transformedImage, grayScale, CV_BGR2GRAY);


		erodeDilate(transformedImage, erosion_size);
		
		return true;
	}
	else {
		printf("No aruco marker found.");
		return false;
	}
	

}

char matchLetter(Mat testLetter, vector<Mat> trainingLetters) {
	int match = -MAXINT;
	char letter;
	float correllation = 0;

	for (int j = 0; j < trainingLetters.size(); j++) {
		Mat templ = trainingLetters[j];
		Mat result;

		if (templ.size().height > testLetter.size().height && templ.size().width > testLetter.size().width && \
			templ.size().width * 0.40 < testLetter.size().width && templ.size().height * 0.40 < testLetter.size().height) {
			matchTemplate(testLetter, templ, result, CV_TM_CCORR_NORMED);


			double minVal; double maxVal; Point minLoc; Point maxLoc;
			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
			if (maxVal > correllation) {
				match = j;
				correllation = maxVal;
			}
		}

	}
	if (match != -MAXINT) {
		// If character is a .
		if (match == 26) {
			match = -19;
		}
		// If character is aruco marker
		if (match == 27) {
			match = -65;
		}
		letter = char(65 + match);
	}
	else {
		letter = '-';
	}


	return letter;
}

string makeSentence(vector<string> words) {
	string sentence = "";

	for (int i = 0; i < words.size(); i++) {
		if (words[i] != ".") {
			sentence.append(" ");
		}

		sentence.append(words[i].c_str());
	}

	return sentence;
}


String readScaledText(Mat testImage, vector<Mat>& trainingLetters) {
	int currentX = -MAXINT;
	int currentY = -MAXINT;
	int lastX = -MAXINT;
	int lastY = -MAXINT;

	vector<Rect> boundRect;
	getBoundingRect(testImage, boundRect);

	vector<string> words;
	string word;

	for(int i = 0; i < boundRect.size(); i++) {
		currentX = boundRect[i].x;
		currentY = boundRect[i].y;

		Mat testLetter = Mat(testImage, boundRect[i]);

		char match = matchLetter(testLetter, trainingLetters);

		if (currentY > lastY + 20 || currentX > lastX + 45) {
			if (word != "") {
				words.push_back(word);
			}
			word = "";
			word.push_back(match);
		}
		else {
			word.push_back(match);
		}

		lastX = currentX + boundRect[i].width;
		lastY = currentY;
	}
	words.push_back(word);

	printf("\n%s\n", makeSentence(words).c_str());
	return makeSentence(words).c_str();

	/*if (waitForUser) 
		waitKey(0);*/
}

Mat streamFromCamera() {
	VideoCapture stream1("outputLonely.avi");   //0 is the id of video device.0 if you have only one camera.

	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	}

	//unconditional loop
	while (true) {
		Mat cameraFrame;
		stream1.read(cameraFrame);
		vector<Point2f> markerCorners;
		detectAruco(cameraFrame, markerCorners);
		vector<Rect> boundRect;
		//getBoundingRect(cameraFrame, boundRect, 30);
		imshow("Unregistered HyperCam 2", cameraFrame);
		if (waitKey(30) >= 0)
			return cameraFrame;
	}
}

//https://stackoverflow.com/a/27296/8711488
std::wstring s2ws(const std::string& s)
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}

int sayWithSAPI(String sayText) {
	std::wstring stemp = s2ws(sayText);
	LPCWSTR result = stemp.c_str();

	ISpVoice * pVoice = NULL;

	if (FAILED(::CoInitialize(NULL)))
		return FALSE;

	HRESULT hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void **)&pVoice);
	if (SUCCEEDED(hr))
	{
		hr = pVoice->Speak((LPCWSTR)result, 0, NULL);
		pVoice->Release();
		pVoice = NULL;
	}

	::CoUninitialize();
	return TRUE;
}

// MAIN
int main(int argc, char* argv[])
{
	//sayWithSAPI("Microsoft text to speech test.");

	Mat trainingImage = readImage("training_with_scale_ARUCO.bmp");
	Mat inputImage = readImage("test2.bmp");

	Mat transformedImage;

	vector<Mat> letters;
	initializeTemplates(trainingImage, letters);

	if (useCamera) {
		Mat cameraImg = streamFromCamera();
		showImage(cameraImg, "chosen frame");

		if (transformImage(cameraImg, transformedImage)) {
			String sayText = readScaledText(transformedImage, letters);
			sayWithSAPI(sayText);
		}
	}
	else {
		if (transformImage(inputImage, transformedImage)) {
			String sayText = readScaledText(transformedImage, letters);
			sayWithSAPI(sayText);
			waitKey();
		}
	}

	//Don't dissappear until I confirm I saw the text output
}
