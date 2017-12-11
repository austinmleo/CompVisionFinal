#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <windows.h> // For Sleep()

using namespace std;

int main(int argc, char* argv[])
{
	boolean doResize = false;

	cv::VideoCapture cap(0); // open the camera


	if (!cap.isOpened()) { // check if we succeeded
		std::printf("error - can't open the camera or video; hit any key to quit\n");
		system("PAUSE");
		return EXIT_FAILURE;
	}
	// Let's just see what the image size is from this camera or file.
	double WIDTH = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	double HEIGHT = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	std::printf("Image width=%f, height=%f\n", WIDTH, HEIGHT);

	/*
	int shrinkFactor = 4;

	//Shrink and rotate from original vid
	if (doResize) {
		double temp = WIDTH;
		WIDTH = (int)HEIGHT / shrinkFactor;
		HEIGHT = (int)temp / shrinkFactor;
	}*/

	//Prepare output video
	const cv::String fnameOut("outputPangram.avi");
	cv::VideoWriter outputVideo(fnameOut,
		cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),
		30.0,               // fps
		cv::Size(WIDTH, HEIGHT),
		true);              // true (default): output color

	// Allocate image.
	cv::Mat image;

	// Run an infinite loop until user hits the ESC key.
	while (1) {
		cap >> image; // get image from camera
		if (image.empty()) break;

	/*
		if (doResize) {
			cv::resize(image, image, cv::Size(image.cols / shrinkFactor, image.rows / shrinkFactor)); // to half size or even smaller
																									  //Rotate 90
			cv::transpose(image, image);
			cv::flip(image, image, 1);
			cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
		}
		*/

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
