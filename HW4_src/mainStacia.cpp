#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <windows.h> // For Sleep()
// Length of one side of a square marker.
const float markerLength = 2.0;

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	std::printf("This program detects ArUco markers.\n");
	std::printf("Hit the ESC key to quit.\n");

	string img = "training_with_scale_ARUCO.bmp";

	cv::Mat K;
	cv::Mat distCoeffs;

	// Camera intrinsic matrix (fill in your actual values here).
	double K_[3][3] =
	{ { 675, 0, 320 },
	{ 0, 675, 240 },
	{ 0, 0, 1 } };
	K = cv::Mat(3, 3, CV_64F, K_).clone();

	// Distortion coeffs (fill in your actual values here).
	double dist_[] = { 0, 0, 0, 0, 0 };
	distCoeffs = cv::Mat(5, 1, CV_64F, dist_).clone();

	//cv::VideoCapture cap(0); // open the camera

	Mat image = imread(img);
	double WIDTH = image.size().width;
	double HEIGHT = image.size().width;
	std::printf("Image width=%f, height=%f\n", WIDTH, HEIGHT);

	// Allocate image.
	//cv::Mat image;
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
	cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
	// Run an infinite loop until user hits the ESC key.

	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
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

			cv::Point3d circleCoord;


			double inch = markerLength / 2;

			if (markerIds[i] == 0) {
				circleCoord = cv::Point3d(2.5*inch, -2.0*inch, -1.0*inch);
			}
			else if (markerIds[i] == 1) {
				circleCoord = cv::Point3d(-2.5*inch, -2.0*inch, -5.0*inch);
			}
			pointsInterest.push_back(circleCoord);
			std::vector<cv::Point2d> p;
			cv::projectPoints(pointsInterest, rvecs[i], tvecs[i], K, distCoeffs, p);
			cv::circle(image,
				p[0], // image point
				10, //radius
				cv::Scalar(0, 255, 255), // color
				2); // thickness

		}

		imshow("Image", image); // show image
									// Wait for x ms (0 means wait until a keypress).
									// Returns -1 if no key is hit.
		Sleep(1000);
	}
	return EXIT_SUCCESS;
}
