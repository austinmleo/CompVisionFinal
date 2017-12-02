#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <windows.h> // For Sleep()
// Length of one side of a square marker.
const float markerLength = 2.0;

using namespace std;

int main(int argc, char* argv[])
{
	std::printf("This program detects ArUco markers.\n");
	std::printf("Hit the ESC key to quit.\n");

	string videoName = "myVideo.avi";
	boolean doResize = false;

	cv::Mat K;
	cv::Mat distCoeffs;

	/*
	// Camera intrinsic matrix (fill in your actual values here).
	double K_[3][3] =
	{ { 1287.67061797194,	0,	0 },
	{ 29.4619513828229,	1223.49459640499,	0 },
	{ 841.008456053631,	464.374869118092,	1} };*/

	double K_[3][3] =
	{ {1282.68112545420,	0,	0},
	{17.0686634073436,	1199.40203358629,	0},
	{792.466601176545,	424.647109808187,	1}};


	K = cv::Mat(3, 3, CV_64F, K_).clone();

	// Distortion coeffs (fill in your actual values here).
	//double dist_[] = { 0.4853, - 3.1046, -0.0091, 0.0083, 6.6024 };
	double dist_[] = { 0,0,0,0,0 };

	//RadialDistortion: [0.3048 - 1.6326 2.0325]
	//TangentialDistortion : [0.0040 0.0064]

	distCoeffs = cv::Mat(5, 1, CV_64F, dist_).clone();


	//cv::VideoCapture cap(0); // open the camera

	cv::VideoCapture cap(videoName); // or open the video file

	if (!cap.isOpened()) { // check if we succeeded
		std::printf("error - can't open the camera or video; hit any key to quit\n");
		system("PAUSE");
		return EXIT_FAILURE;
	}
	// Let's just see what the image size is from this camera or file.
	double WIDTH = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	double HEIGHT = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	std::printf("Image width=%f, height=%f\n", WIDTH, HEIGHT);

	int shrinkFactor = 4;

	//Shrink and rotate from original vid
	if (doResize) {
		double temp = WIDTH;
		WIDTH = (int)HEIGHT / shrinkFactor;
		HEIGHT = (int)temp / shrinkFactor;
	}

	//Prepare output video
	const cv::String fnameOut("output.avi");
	cv::VideoWriter outputVideo(fnameOut,
		cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),
		30.0,               // fps
		cv::Size(WIDTH, HEIGHT),
		true);              // true (default): output color

	// Allocate image.
	cv::Mat image;
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);
	cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
	// Run an infinite loop until user hits the ESC key.
	while (1) {
		cap >> image; // get image from camera
		if (image.empty()) break;
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
			std::vector< cv::Vec3d > rvecs, tvecs;
			cv::aruco::estimatePoseSingleMarkers(
				markerCorners, // vector of already detected markers corners
				markerLength, // length of the marker's side
				K, // input 3x3 floating-point instrinsic camera matrix K
				distCoeffs, // vector of distortion coefficients of 4, 5, 8 or 12 elements
				rvecs, // array of output rotation vectors
				tvecs); // array of output translation vectors
						// Display pose for the detected marker with id=0.
			for (unsigned int i = 0; i < markerIds.size(); i++) {
				cv::Vec3d r = rvecs[i];
				cv::Vec3d t = tvecs[i];
				// Draw coordinate axes.
				cv::aruco::drawAxis(image,
					K, distCoeffs, // camera parameters
					r, t, // marker pose
					0.5*markerLength); // length of the axes to be drawn
				// Draw a symbol in the upper right corner of the detected marker.
				std::vector<cv::Point3d> pointsInterest;

				double cm = markerLength / 7;

				cv::Point3d handleCoord = cv::Point3d(19.5*cm, -44 * cm, 0);
				cv::Point3d hingeCoord = cv::Point3d(18 * cm, 2 * cm, -3.5*cm);

				pointsInterest.push_back(handleCoord);
				pointsInterest.push_back(hingeCoord);

				if (markerIds[i] == 110)
				{
					//yellow handle
					std::vector<cv::Point2d> p;
					cv::projectPoints(pointsInterest, rvecs[i], tvecs[i], K, distCoeffs, p);
					cv::circle(image,
						p[0], // image point
						20, //radius
						cv::Scalar(0, 255, 255), // color
						3); // thickness
				}
				else if (markerIds[i] == 111) {
					//green hinge
					std::vector<cv::Point2d> p;
					cv::projectPoints(pointsInterest, rvecs[i], tvecs[i], K, distCoeffs, p);
					cv::circle(image,
						p[1], // image point
						20, //radius
						cv::Scalar(0, 255, 0), // color
						3); // thickness
				}
			}


		}

		if (doResize) {
			cv::resize(image, image, cv::Size(image.cols / shrinkFactor, image.rows / shrinkFactor)); // to half size or even smaller
			//Rotate 90
			cv::transpose(image, image);
			cv::flip(image, image, 1);
			cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
		}

		cv::imshow("Image", image); // show image
									// Wait for x ms (0 means wait until a keypress).
									// Returns -1 if no key is hit.
									//Print to output video

		outputVideo << image;

		char key = cv::waitKey(1);
		if (key == 27) break; // ESC is ascii 27
	}
	return EXIT_SUCCESS;
}
