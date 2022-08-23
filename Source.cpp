#include<iostream>
#include<opencv2/highgui.hpp>
#include<opencv2/objdetect.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/calib3d.hpp>
#include<opencv2/dnn/all_layers.hpp>
#include<opencv2/dnn.hpp>
#include<opencv2/opencv.hpp>
#include<fstream>
#include<cmath>


int main() {

	cv::Mat img;

	cv::Mat maskImg = cv::imread("E:/MaskingFace/maskTemplate/surgical_blue.png");

	// Loading pre-trained model
	cv::dnn::Net faceDetection = cv::dnn::readNetFromModelOptimizer("E:/MaskingFace/models/faceDetection/face-detection-retail-0005.xml", "E:/MaskingFace/models/faceDetection/face-detection-retail-0005.bin");
	cv::dnn::Net landmarkDetection = cv::dnn::readNet("E:/MaskingFace/models/facialLandmark/facial-landmarks-35-adas-0002.bin", "E:/MaskingFace/models/facialLandmark/facial-landmarks-35-adas-0002.xml");

	// Declaring mask points 
	std::vector<cv::Point2i> maskPoints{
		// {122, 90}, {405, 7}, {686, 79}, {406, 509}, {653, 311}, {165, 323} cloth mask points
		{10, 97}, {307, 22}, {600, 99}, {295, 470}, {600, 323}, {45, 322} // surgical_blue mask points
	};

	// Declaring world corrdinates
	std::vector<cv::Point3d> model_points{
	{0.0, 0.0, 0.0}, {0.0, -330.0, -65.0}, {-225.0, 170.0, -135.0}, {225.0, 170.0, -135.0}, {-150.0, -150.0, -125.0}, {150.0, -150.0, -125.0}
	};

	cv::VideoCapture cap(0);

	if (!cap.isOpened()) {

		std::cout << "cannot open camera";

	}

	while (true) {

		cap >> img;

		// Detecting face in an image
		cv::Mat detectionBlob = cv::dnn::blobFromImage(img, 1.0, cv::Size(300, 300));

		faceDetection.setInput(detectionBlob);

		cv::Mat detectedFaces = faceDetection.forward();

		// Croping image of detected face with 5%+ scaling frame size
		int x1 = static_cast<int>((detectedFaces.at<float>(0, 3) * img.cols) - (img.cols * 0.05));
		int y1 = static_cast<int>((detectedFaces.at<float>(0, 4) * img.rows) - (img.rows * 0.05));
		int x2 = static_cast<int>((detectedFaces.at<float>(0, 5) * img.cols) + (img.cols * 0.05));
		int y2 = static_cast<int>((detectedFaces.at<float>(0, 6) * img.rows) + (img.rows * 0.05));
		
		// If face not detected throw warning 
		if (x1 < 0 || y1 < 0 || x2 > img.cols || y2 > img.rows) {
			cv::putText(img, "Stay in front of the camera", cv::Point(15, 45), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
			cv::imshow("Mask Filter", img);
			cv::waitKey(1);
		}

		else {
			cv::Mat faceImg = img(cv::Range(y1, y2), cv::Range(x1, x2));

			// Detecting facial landmark from image
			cv::Mat landmarkBlob = cv::dnn::blobFromImage(faceImg, 1, cv::Size(60, 60));

			landmarkDetection.setInput(landmarkBlob);
			cv::Mat normOutput = landmarkDetection.forward();

			// Selecting specific landmarks for estimating euler angles
			std::vector <cv::Point2f> image_points{
				{x1 + normOutput.at<float>(0, 8) * faceImg.cols, y1 + normOutput.at<float>(0, 9) * faceImg.rows}, // Nose tip
				{x1 + normOutput.at<float>(0, 52) * faceImg.cols, y1 + normOutput.at<float>(0, 53) * faceImg.rows}, // Chin
				{x1 + normOutput.at<float>(0, 6) * faceImg.cols, y1 + normOutput.at<float>(0, 7) * faceImg.rows}, // Left eye left
				{x1 + normOutput.at<float>(0, 2) * faceImg.cols, y1 + normOutput.at<float>(0, 3) * faceImg.rows}, // Right eye right
				{x1 + normOutput.at<float>(0, 18) * faceImg.cols, y1 + normOutput.at<float>(0, 19) * faceImg.rows}, // Left Mouth corner
				{x1 + normOutput.at<float>(0, 16) * faceImg.cols, y1 + normOutput.at<float>(0, 17) * faceImg.rows}, // Right mouth corner
			};

			// Approximating camera parameters (We can also calibrate actual camara parameters instead of approximating)
			double focal_length = img.cols;
			cv::Point2d center = cv::Point2d(img.cols / 2, img.rows / 2);
			cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
			cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

			// Estimating the pose of a calibrated camera
			cv::Mat rotation_vector, translation_vector;
			cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

			// Estimating euler oriantation angles
			cv::Mat rotaionMatrix;
			cv::Rodrigues(rotation_vector, rotaionMatrix);

			cv::Mat R, Q, x, y, z;

			cv::Vec3d angles = cv::RQDecomp3x3(rotaionMatrix, R, Q, x, y, z);

			double yaw = angles[1];
			double pitch = angles[0];
			double roll = angles[2];

			// Ploting direction based on euler angles
			std::string text;

			if (yaw > 15) {
				text = "Looking Right";
			}
			else if (yaw < -15) {
				text = "Looking Left";
			}
			else if (pitch > 10) {
				text = "Looking Down";
			}
			else if (pitch < -10) {
				text = "Looking Up";
			}
			else if (roll > 10) {
				text = "Rolling Left";
			}
			else if (roll < -10) {
				text = "Rolling Right";
			}
			else {
				text = "Looking Straight";
			}
			
			cv::putText(img, text, cv::Point(15, 45), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(60, 20, 228), 2, cv::LINE_AA);

			// Ploting mid points for both eyes
			cv::Point2f ll = cv::Point2f(x1 + normOutput.at<float>(0, 6) * faceImg.cols, y1 + normOutput.at<float>(0, 7) * faceImg.rows);
			cv::Point2f lr = cv::Point2f(x1 + normOutput.at<float>(0, 4) * faceImg.cols, y1 + normOutput.at<float>(0, 5) * faceImg.rows);
			cv::Point2f lelftEyeMid = (ll + lr) / 2;

			cv::Point2f rl = cv::Point2f(x1 + normOutput.at<float>(0, 0) * faceImg.cols, y1 + normOutput.at<float>(0, 1) * faceImg.rows);
			cv::Point2f rr = cv::Point2f(x1 + normOutput.at<float>(0, 2) * faceImg.cols, y1 + normOutput.at<float>(0, 3) * faceImg.rows);
			cv::Point2f rightEyeMid = (rl + rr) / 2;

			cv::circle(img, lelftEyeMid, 4, cv::Scalar(255, 0, 0), 1);
			cv::circle(img, rightEyeMid, 4, cv::Scalar(255, 0, 0), 1);

			// Selecting specific landmarks for mask agumentation
			std::vector<cv::Point2f> facePoints{
				{x1 + normOutput.at<float>(0, 40) * faceImg.cols, y1 + normOutput.at<float>(0, 41) * faceImg.rows},
				{x1 + normOutput.at<float>(0, 8) * faceImg.cols, y1 + normOutput.at<float>(0, 9) * faceImg.rows},
				{x1 + normOutput.at<float>(0, 64) * faceImg.cols, y1 + normOutput.at<float>(0, 65) * faceImg.rows},
				{x1 + normOutput.at<float>(0, 52) * faceImg.cols, y1 + normOutput.at<float>(0, 53) * faceImg.rows},
				{x1 + normOutput.at<float>(0, 58) * faceImg.cols, y1 + normOutput.at<float>(0, 59) * faceImg.rows},
				{x1 + normOutput.at<float>(0, 46) * faceImg.cols, y1 + normOutput.at<float>(0, 47) * faceImg.rows}
			};

			// Calculating homography between selected mask points and facial landmarks
			cv::Mat homoMatrix = cv::findHomography(maskPoints, facePoints);
			cv::Mat transformedMask;
			cv::warpPerspective(maskImg, transformedMask, homoMatrix, img.size());

			// Agumentation of mask on face
			cv::Mat mask, maskInv, imgBg, imgFg, addedImg;
			cv::cvtColor(transformedMask, mask, cv::COLOR_BGR2GRAY);
			cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);

			cv::bitwise_not(mask, maskInv);
			cv::bitwise_and(img, img, imgBg, maskInv);
			cv::bitwise_and(transformedMask, transformedMask, imgFg, mask);

			cv::add(imgBg, imgFg, addedImg);

			cv::imshow("Mask Filter", addedImg);
			cv::waitKey(1);
		}
	}
	cv::destroyAllWindows();

	return 0;
}
