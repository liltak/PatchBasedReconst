#pragma once

#include <iostream>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <Eigen/Eigen>

struct NormalMapTuple {
	const Eigen::MatrixXd NX;
	const Eigen::MatrixXd NY;
	const Eigen::MatrixXd NZ;
	const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> V;
	NormalMapTuple(const Eigen::MatrixXd& NX, const Eigen::MatrixXd& NY, const Eigen::MatrixXd& NZ, const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& V)
		: NX(NX), NY(NY), NZ(NZ), V(V)
	{
	}
};

std::vector<cv::Mat> PatchBasedNormalEstimation(cv::Mat input, cv::Mat Mask_img);

std::vector<cv::Mat> DataOpen(int datatype, int number, std::string dataname);

cv::Mat cvt_data(const cv::Mat& Data, const cv::Mat& Mask);

std::vector<cv::Mat> PatchBasedNormalEstimation_iteration(cv::Mat Input, cv::Mat Mask_img, NormalMapTuple estimated_maps);

