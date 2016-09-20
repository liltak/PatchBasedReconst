//
//  process.hpp
//  real_portraits
//
//  Created by yu-bun on 2016/04/07.
//  Copyright © 2016年 yu-bun. All rights reserved.
//

#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <math.h>
//#include <unistd.h>

using namespace cv;
using namespace std;
using namespace Eigen;

Mat image_preparation(Mat& image);
Mat denoisefilter(cv::Mat src, cv::Mat tar);
void myBlending(Mat src, Mat dst, Point p, Mat& output);
void mySeamlessClone(Mat tile, Mat& dst, Point pt, int off = 1);
void conpute_luminanceave(Mat& image, int M, vector<double>& input_luminance, vector<Point2f>& patch_center);
void draw_point( Mat& img, Point2f fp, Scalar color );
Point2f sub_vector( Point2f a, Point2f b );
void makingmodel( Mat& img, Subdiv2D& subdiv, Scalar delaunay_color, vector<Point2f>& points, vector<vector<int>>& pointslist);
void makingmesh (Mat& tar, Scalar delaunay_color, vector<Point2f>& points_tar, vector<vector<int>>& pointslist );
void meshmatch(vector<float>& min, vector<int>& minID, int a, int ii, int M, Mat& img,Mat& img_gray, Mat& tar_gray,
               vector<Point2f>& points,vector<Point2f>& points_tar, vector<vector<int>>& pointslist,
               Point2f& P, vector<vector<Point2f>>& Pd, vector<double>& input_luminance, vector<float>& patch_dis);