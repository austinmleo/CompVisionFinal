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

// Length of one side of a square marker.
const float markerLength = 2.0;

using namespace cv;
using namespace std;

Mat readImage(String filename) {
	cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);

	if (image.empty())
	{
		cout << "Image not loaded";
		std::exit(1);
	}
	else {
		return image;
	}
}

void showImage(Mat image, String windowname) {
	cv::namedWindow(windowname, CV_WINDOW_AUTOSIZE);
	cv::imshow(windowname, image);
	cv::waitKey(0);
}

void showImageResize(Mat image, String windowname) {
	cv::namedWindow(windowname, CV_WINDOW_NORMAL);
	//cv::resizeWindow(windowname, image.size().width*scale, image.size().height*scale);
	cv::imshow(windowname, image);
	cv::waitKey(0);
}

vector<Point2f> detectAruco(Mat image) {
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);
	cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();

	std::vector< int > markerIds;
	std::vector< std::vector<cv::Point2f> > markerCorners, rejectedCandidates;
	cv::aruco::detectMarkers(
		image, // input image
		dictionary, // type of markers that will be searched for
		markerCorners, // output vector of marker corners
		markerIds, // detected marker IDs
		detectorParams, // algorithm parameters
		rejectedCandidates);
	if (markerIds.size() > 0) {
		// Draw all detected markers.
		cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);
	}

	for (unsigned int i = 0; i < (int)markerCorners[0].size(); i++) {
		int fontFace = 1;
		double fontScale = 5;
		Scalar color = 255;
		putText(image, to_string(i+1), markerCorners[0][i], fontFace, fontScale, color, 3);
	}
	showImageResize(image, "Aruco markers");
	//only one aruco marker should be found
	return markerCorners[0];
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
		int m = 100;
		int n = 100;
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
	std::reverse(boundRect.begin(), boundRect.end());
	
	return boundRect;

}

// initializeTemplates
void initializeTemplates(Mat templateImage, std::vector<Mat>& letters) {
	//Find aruco marker
	vector<Point2f> markerCorners = detectAruco(templateImage);
	//top left, bottom right
	//Need to know this so any contours within aruco marker region
	//are not detected, only bounding boxes of letters are detected
	Point tl = markerCorners[1];
	Point br = markerCorners[3];
	//x increases in a counterintuitive direction
	int arucoMinX = br.x;
	int arucoMaxX = tl.x;
	int arucoMinY = tl.y;
	int arucoMaxY = br.y;

	//Mat A = imread("A.png", CV_LOAD_IMAGE_COLOR);
	//letters.push_back(A);
	Mat binaryImage;
	Mat gray;
	int thresh = 0;
	int const max_BINARY_value = 255;
	int threshold_type = THRESH_BINARY;

	cv::cvtColor(templateImage, gray, CV_BGR2GRAY);
	threshold(gray, binaryImage, thresh, max_BINARY_value, threshold_type);
	//showImage(binaryImage, "Binary image");

	//Find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//binaryImage, input image (is destroyed)
	findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	//Source for below code:
	//https://docs.opencv.org/3.3.0/da/d0c/tutorial_bounding_rects_circles.html

	int numChars = 26;
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(numChars+1); //last is bounding rect of aruco mark
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	Mat coloredBounds = templateImage.clone();

	int lettersCounter = 0;

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
			if ((cx > arucoMinX && cx < arucoMaxX)&&(cy > arucoMinY && cy < arucoMaxY)) {
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
		boundRect[lettersCounter] = boundingRect(Mat(contours_poly[i]));
		lettersCounter++;
		//minEnclosingCircle(contours_poly[i], center[i], radius[i]);
	}

	cout << "# letters: " << lettersCounter;
	//Insert aruco marker as #27 (index 26)
	boundRect[lettersCounter] = boundingRect(markerCorners);

	drawBoundingRect(coloredBounds, boundRect);
	vector<Rect> sortedRect = sortBoundingRect(boundRect);

	Mat sortedBounds = templateImage.clone();
	drawBoundingRect(sortedBounds, sortedRect);
}

// MAIN
int main(int argc, char* argv[])
{
	Mat trainingImage = readImage("training_with_scale_ARUCO.bmp");
	//showImage(trainingImage, "Template image");

	vector<Mat> letters;
	initializeTemplates(trainingImage, letters);

	/*
	printf("This program detects ArUco markers.\n");
	printf("Hit the ESC key to quit.\n");

	// Camera intrinsic matrix
	double K_[3][3] =
	{ { 675, 0, 320 },
	{ 0, 675, 240 },
	{ 0, 0, 1 } };
	Mat K = Mat(3, 3, CV_64F, K_).clone();

	// Distortion coeffs (fill in your actual values here).
	double dist_[] = { 0, 0, 0, 0, 0 };
	Mat distCoeffs = Mat(5, 1, CV_64F, dist_).clone();

	// Open video from file.
	VideoCapture cap("hw4.avi");

	if (!cap.isOpened()) { // check if we succeeded
		printf("error - can't open the camera or video; hit any key to quit\n");
		system("PAUSE");
		return EXIT_FAILURE;
	}

	// See what the image size is from this file.
	double WIDTH = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	double HEIGHT = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	printf("Image width=%f, height=%f\n", WIDTH, HEIGHT);


	std::vector<Mat> letters; 
	initializeTemplates(letters);
	

	// Allocate image.
	Mat image;
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_100);
	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();


	// Run an infinite loop until user hits the ESC key.
	while (1) {
		cap >> image; // get image from camera

		if (image.empty()) break;

		std::vector< int > markerIds;
		std::vector< std::vector<Point2f> > markerCorners, rejectedCandidates;
		aruco::detectMarkers(
			image,				// input image
			dictionary,			// type of markers that will be searched for
			markerCorners,		// output vector of marker corners
			markerIds,			// detected marker IDs
			detectorParams,		// algorithm parameters
			rejectedCandidates);

		if (markerIds.size() > 0) {
			// Draw all detected markers.
			aruco::drawDetectedMarkers(image, markerCorners, markerIds);
			std::vector< Vec3d > rvecs, tvecs;

			aruco::estimatePoseSingleMarkers(
				markerCorners,	// vector of already detected markers corners
				markerLength,	// length of the marker's side
				K,				// input 3x3 floating-point instrinsic camera matrix K
				distCoeffs,		// vector of distortion coefficients of 4, 5, 8 or 12 elements
				rvecs,			// array of output rotation vectors
				tvecs);			// array of output translation vectors

								// Points of button relative to marker.
			Point3d marker0Button = Point3d(2.5, -2.0, -1.0);
			Point3d marker1Button = Point3d(-2.5, -2.0, -5.0);

			for (unsigned int i = 0; i < markerIds.size(); i++) {
				// Display pose for the detected marker with id=0.
				if (markerIds[i] == 0) {
					Vec3d r = rvecs[i];
					Vec3d t = tvecs[i];

					// Draw coordinate axes.
					aruco::drawAxis(image,
						K, distCoeffs, // camera parameters
						r, t, // marker pose
						0.5*markerLength); // length of the axes to be drawn
										   // Draw a symbol in the upper right corner of the detected marker.

										   // Draw a circle on the button relative to marker 0.
					std::vector<Point3d> pointsInterest;
					pointsInterest.push_back(marker0Button);
					std::vector<Point2d> p;
					projectPoints(pointsInterest, rvecs[i], tvecs[i], K, distCoeffs, p);
					circle(image,
						p[0],						// image point,
						15,							// radius
						Scalar(0, 255, 255),	// color
						2);							// thickness
				}

				// Display pose for the detected marker with id=1.
				if (markerIds[i] == 1) {
					Vec3d r = rvecs[i];
					Vec3d t = tvecs[i];

					// Draw coordinate axes.
					aruco::drawAxis(image,
						K, distCoeffs,		// camera parameters
						r, t,				// marker pose
						0.5*markerLength);	// length of the axes to be drawn
											// Draw a symbol in the upper right corner of the detected marker.

											// Draw a circle on the button relative to marker 1.
					std::vector<Point3d> pointsInterest;
					pointsInterest.push_back(marker1Button);
					std::vector<Point2d> p;
					projectPoints(pointsInterest, rvecs[i], tvecs[i], K, distCoeffs, p);
					circle(image,
						p[0],						// image point,
						15,							// radius
						Scalar(0, 255, 255),	// color
						2);							// thickness
				}
			}
		}
		imshow("Image", image); // show image
									// Wait for x ms (0 means wait until a keypress).
									// Returns -1 if no key is hit.

		imshow("A", letters[0]);
		char key = waitKey(1);
		if (key == 27) break; // ESC is ascii 27
	}
	return EXIT_SUCCESS;
	*/
}
