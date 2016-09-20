#pragma once
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/image_processing.h"
#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include "dlib/opencv.h"

#include "process.hpp"

#include "SimplePoissonDepthReconstructor.h"

class pb3d_tsukasa
{
public:
    pb3d_tsukasa();
    static void getLandmark(cv::Mat & img, std::vector<dlib::full_object_detection>& shapes, dlib::frontal_face_detector & detector, dlib::shape_predictor & pose_model, float ratio);
    static void cross(cv::Mat & dst, cv::Point pt, double length, cv::Scalar color, double angle, double thickness);
    static cv::Point ptd2cv(dlib::full_object_detection shape, unsigned long i);
    static int draw_shapes(cv::Mat & dst, dlib::full_object_detection shape);
    static void cvtDlib2cv(std::vector<cv::Point>& pts, dlib::full_object_detection & shape);
    static std::vector<cv::Mat> makemask(cv::Mat & Input);
    static cv::Mat cvt_data(const cv::Mat & Data, const cv::Mat & Mask);
    static void maketxt(std::string input, dlib::full_object_detection shape, std::string name);

	static std::vector<float> face_rot(cv::Point eye_R, cv::Point eye_L);


	static cv::Mat Normalize_rou(cv::Mat input_img, std::vector<float>& mat_set);

	~pb3d_tsukasa();
};

