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

	cv::Mat maskImg = cv::imread("E:/MaskTheFace/maskTemplate/cloth.png");

	cv::dnn::Net faceDetection = cv::dnn::readNetFromModelOptimizer("E:/MaskTheFace/models/faceDetection/face-detection-retail-0005.xml", "E:/MaskTheFace/models/faceDetection/face-detection-retail-0005.bin");
	cv::dnn::Net landmarkDetection = cv::dnn::readNet("E:/MaskTheFace/models/facialLandmark/facial-landmarks-35-adas-0002.bin", "E:/MaskTheFace/models/facialLandmark/facial-landmarks-35-adas-0002.xml");

	std::vector<cv::Point2i> maskPoints{
		//{131, 93}, {400, 10}, {678, 83}, {413, 499}, {406, 509}, {165, 323}
		{122, 90}, {405, 7}, {686, 79}, {406, 509}, {653, 311}, {165, 323}
	};

	std::vector<cv::Point3d> model_points{
	{0.0, 0.0, 0.0}, {0.0, -330.0, -65.0}, {-225.0, 170.0, -135.0}, {225.0, 170.0, -135.0}, {-150.0, -150.0, -125.0}, {150.0, -150.0, -125.0}
	};

	cv::VideoCapture cap(0);

	if (!cap.isOpened()) {

		std::cout << "cannot open camera";

	}

	while (true) {

		cap >> img;

		cv::Mat detectionBlob = cv::dnn::blobFromImage(img, 1, cv::Size(300, 300));

		faceDetection.setInput(detectionBlob);
		cv::Mat detectedFaces = faceDetection.forward();

		int x1 = static_cast<int>((detectedFaces.at<float>(0, 3) * img.cols) - (img.cols * 0.05));
		int y1 = static_cast<int>((detectedFaces.at<float>(0, 4) * img.rows) - (img.rows * 0.05));
		int x2 = static_cast<int>((detectedFaces.at<float>(0, 5) * img.cols) + (img.cols * 0.05));
		int y2 = static_cast<int>((detectedFaces.at<float>(0, 6) * img.rows) + (img.rows * 0.05));


		cv::Mat faceImg = img(cv::Range(y1, y2), cv::Range(x1, x2));

		cv::Mat landmarkBlob = cv::dnn::blobFromImage(faceImg, 1, cv::Size(60, 60));

		landmarkDetection.setInput(landmarkBlob);
		cv::Mat normOutput = landmarkDetection.forward();

		std::vector <cv::Point2f> image_points{
			{x1 + normOutput.at<float>(0, 8) * faceImg.cols, y1 + normOutput.at<float>(0, 9) * faceImg.rows}, // Nose tip
			{x1 + normOutput.at<float>(0, 52) * faceImg.cols, y1 + normOutput.at<float>(0, 53) * faceImg.rows}, // Chin
			{x1 + normOutput.at<float>(0, 6) * faceImg.cols, y1 + normOutput.at<float>(0, 7) * faceImg.rows}, // Left eye left
			{x1 + normOutput.at<float>(0, 2) * faceImg.cols, y1 + normOutput.at<float>(0, 3) * faceImg.rows}, // Right eye right
			{x1 + normOutput.at<float>(0, 18) * faceImg.cols, y1 + normOutput.at<float>(0, 19) * faceImg.rows}, // Left Mouth corner
			{x1 + normOutput.at<float>(0, 16) * faceImg.cols, y1 + normOutput.at<float>(0, 17) * faceImg.rows}, // Right mouth corner
		};


		double focal_length = img.cols;
		cv::Point2d center = cv::Point2d(img.cols / 2, img.rows / 2);
		cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
		cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

		cv::Mat rotation_vector, translation_vector;
		cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

		cv::Mat rotaionMatrix;
		cv::Rodrigues(rotation_vector, rotaionMatrix);

		cv::Mat R, Q, x, y, z;

		cv::Vec3d angles = cv::RQDecomp3x3(rotaionMatrix, R, Q, x, y, z);

		cv::putText(img, std::to_string(angles[0]), cv::Point(20, 25), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
		cv::putText(img, std::to_string(angles[1]), cv::Point(20, 55), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
		cv::putText(img, std::to_string(angles[2]), cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

		std::vector<cv::Point2f> facePoints{
			{x1 + normOutput.at<float>(0, 40) * faceImg.cols, y1 + normOutput.at<float>(0, 41) * faceImg.rows},
			{x1 + normOutput.at<float>(0, 8) * faceImg.cols, y1 + normOutput.at<float>(0, 9) * faceImg.rows},
			{x1 + normOutput.at<float>(0, 64) * faceImg.cols, y1 + normOutput.at<float>(0, 65) * faceImg.rows},
			{x1 + normOutput.at<float>(0, 52) * faceImg.cols, y1 + normOutput.at<float>(0, 53) * faceImg.rows},
			{x1 + normOutput.at<float>(0, 58) * faceImg.cols, y1 + normOutput.at<float>(0, 59) * faceImg.rows},
			{x1 + normOutput.at<float>(0, 46) * faceImg.cols, y1 + normOutput.at<float>(0, 47) * faceImg.rows}
		};

		cv::Mat homoMatrix = cv::findHomography(maskPoints, facePoints);
		cv::Mat transformedMask;
		cv::warpPerspective(maskImg, transformedMask, homoMatrix, img.size());

		cv::Mat mask, maskInv, imgBg, imgFg, addedImg;
		cv::cvtColor(transformedMask, mask, cv::COLOR_BGR2GRAY);
		cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);

		cv::bitwise_not(mask, maskInv);
		cv::bitwise_and(img, img, imgBg, maskInv);
		cv::bitwise_and(transformedMask, transformedMask, imgFg, mask);

		cv::add(imgBg, imgFg, addedImg);

		cv::imshow("0", addedImg);
		cv::imshow("1", transformedMask);

		cv::waitKey(1);

	}
	return 0;
}

