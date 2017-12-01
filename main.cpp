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

// initializeTemplates
void initializeTemplates(std::vector<Mat>& letters) {

	Mat A = imread("A.png", CV_LOAD_IMAGE_COLOR);

	letters.push_back(A);

}




// MAIN
int main(int argc, char* argv[])
{
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
}
