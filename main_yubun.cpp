//
//  main.cpp
//  real_portraits
//
//  Created by yu-bun on 2016/04/07.
//  Copyright © 2016年 yu-bun. All rights reserved.
//
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/image_processing.h"
#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include "dlib/opencv.h"

#include "process.hpp"

#include "SimplePoissonDepthReconstructor.h"
#include "PatchBasedNormalEstimation.h"
#include "pb3d_tsukasa.h"

using namespace dlib;
using namespace std;

int i = 0;
int img_size = 300;



//struct NormalMapTuple {
//    const Eigen::MatrixXd NX;
//    const Eigen::MatrixXd NY;
//    const Eigen::MatrixXd NZ;
//    const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> V;
//    NormalMapTuple(const Eigen::MatrixXd& NX, const Eigen::MatrixXd& NY, const Eigen::MatrixXd& NZ, const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& V)
//        : NX(NX), NY(NY), NZ(NZ), V(V)
//    {
//    }
//};


/**
* @fn
* 法線マップ画像(RGB)を、XYZ各成分についてのEigen::MatrixXd型のマップに変換する。また、指定した背景色に相当するかどうかの判定を行い前景フラグマップを生成する。
* @param img 法線を計算したい画像
* @param bgColor 背景色
* @param coord 座標系
* @return 各法線マップおよび前景フラグマップをメンバとして持つ構造体
*/
const NormalMapTuple createNormalMapsFromImage(const cv::Mat_<cv::Vec3b>& img, const cv::Vec3b& bgColor, const int coord) {
    typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXBool;


    const int rows = img.rows;
    const int cols = img.cols;

    Eigen::MatrixXd normalX = Eigen::MatrixXd::Zero(rows, cols);
    Eigen::MatrixXd normalY = Eigen::MatrixXd::Zero(rows, cols);
    Eigen::MatrixXd normalZ = Eigen::MatrixXd::Zero(rows, cols);
    MatrixXBool mask; mask.setConstant(rows, cols, false);


    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cv::Vec3b bgr = img(i, j);

            //領域判定,法線の取得
            if (bgr != bgColor) {
                mask(i, j) = true;

                Eigen::Vector3d n;
                n.x() = bgr[2] * 2.0 / 255.0 - 1.0;
                n.y() = bgr[1] * 2.0 / 255.0 - 1.0;
                n.z() = bgr[0] * 2.0 / 255.0 - 1.0;

                //座標系に合わせて符号反転
                if (coord & 1) n.x() = -n.x();
                if (coord & 2) n.y() = -n.y();
                if (coord & 4) n.z() = -n.z();

                n.normalize();
                normalX(i, j) = n.x();
                normalY(i, j) = n.y();
                normalZ(i, j) = n.z();

            }


        }
    }

    NormalMapTuple nmt(normalX, normalY, normalZ, mask);
    return nmt;
}


NormalMapTuple reconst_yubun( int argc, const char* argv[])
{
	//char *cwd ;
	//cwd=getcwd(NULL, 0); ;
	//printf("CWD:%s\n", cwd);
	//free(cwd);
	
	//input information----------
	string input, databasename_dir;
	input = argv[1];	//inputname(bmp)
	std::string input_dir = argv[2];
	std::string output_dir = argv[3];
	//databasename = "database_Asian/-15";	// DB name
	databasename_dir = argv[4];
	
	std::cout << input_dir << std::endl;
	std::cout << output_dir << std::endl;
	std::cout << databasename_dir << std::endl;

	int data = 16;		//number of DB
	int M = 6;	//patch size
    int ratio = 3;
	string csv;
	csv = ".csv";
	ofstream ofs;
	//std::string output_dir = argv[3];
	ofs.open(output_dir + input + csv, std::ios::out);

	//int maskflag = 0;	//if you want to make mask, you should make mask file name is input name.
	
	
	// Read in the image----------
	string inputimage;
	inputimage = input_dir + input;
	string patchsize;
	patchsize = M;
	Mat img = imread(inputimage + "_transfer_result.jpg");
	if (img.empty()) std::cout << "error";
	//cvtColor(img, img, CV_8UC3);
	cv::Mat Input_ori = cv::imread(inputimage + ".jpg", 1);

	std::vector<dlib::full_object_detection> shapes;
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::proxy_deserialize hoge = dlib::deserialize("shape_predictor_68_face_landmarks.dat");
	hoge >> pose_model;

	pb3d_tsukasa::getLandmark(img, shapes, detector, pose_model, ratio);
    pb3d_tsukasa::maketxt(input_dir, shapes[0], input);

    //マスクの作成（dlib_Result[0] : fp_image, dlib_Result[1] : mask_img ）
    std::vector<cv::Mat> dlib_Result = pb3d_tsukasa::makemask(img);
	
	if (img.empty()) {
		cout << "image not found " << input << endl;
		exit(0);
	} else {
		cout << "image loaded " << input << endl;
		//imshow("Loaded Image " + input, img);
	}

	Mat imgcopy = img.clone();
	Mat img_gray = image_preparation(img);
	Mat result = Mat(img.rows, img.cols, CV_8UC3,Scalar(0,0,0));
	Mat result_nor = Mat(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));
	//result = img.clone();
	
	cout<< "start setup" << endl;
	std::vector<Point2f> points;
	ifstream ifs(inputimage + ".txt");
	int x, y;
	while(ifs >> x >> y)
	{
		points.push_back(Point2f(x,y));
	}
	
	// Define colors for drawing.
	Scalar delaunay_color(255,255,255), points_color(0, 0, 255);
	
	// Rectangle to be used with Subdiv2D
	Size size = img.size();
	Rect rect(0, 0, size.width, size.height);
	
	// Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);
	
	// Insert points into subdiv----------
	for( std::vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
	{
		subdiv.insert(*it);
	}
	
	//making input image model(pointslist)----------
	std::vector<std::vector<int>> pointslist(1000, std::vector<int>(3,-1)) ; // featurepointsID set of mesh
	makingmodel( img, subdiv, delaunay_color, points, pointslist);
	
	//preparation for path synthesis----------
	std::vector<Point2f> patch_center;
	
	std::vector<double> input_luminance;
	//std::vector<std::vector<double>> input_luminance(std::vector<std::vector<double>> (img.cols/M,std::vector<double>(img.rows/M,0)));
	conpute_luminanceave(img_gray, M, input_luminance, patch_center);
	
	//correspondence between input and DB----------
	//std::vector<Mat> tar(data);
	
	cout<<"loading DataBase" <<endl;
	std::vector<Mat> tar(data);
	std::vector<Mat> tar_gray(data);

	std::vector<Mat> tar_nor(data);

	for(int a = 0; a < data; a++){
		//load database
		stringstream database_img;
		database_img << databasename_dir << "tex/" << a << ".bmp";
		tar[a] = imread(database_img.str());
		tar_gray[a] = image_preparation(tar[a]);

		stringstream database_img_nor;
		database_img_nor << databasename_dir << "normal/" << a << ".bmp";
		tar_nor[a] = imread(database_img_nor.str());
	}
	cout<<"finish loading"<<endl;
	
	cout<<"start making offsetmap"<<endl;
	std::vector<std::vector<int>> offsetmap;
	for(int ii = 0; ii < patch_center.size(); ii ++){
		//paramater for patch correspondence
		std::vector<float> patch_dis(data,0.0);
		std::vector<float> min (1,100000.0);
		std::vector<int> minID (1, -1);
		std::vector<std::vector<Point2f>> Pd(patch_center.size(),std::vector<Point2f>(data));	//correspondence point
		
		for(int a = 0; a < data; a++){
			std::vector<Point2f> points_tar;
			
			stringstream database_txt;
			database_txt << databasename_dir << "fp/" << a << ".txt";
			ifstream ifs_tar(database_txt.str());
			int x1, y1;
			while(ifs_tar >> x1 >> y1)
			{
				points_tar.push_back(Point2f(x1,y1));
			}
			
			//std::vector<Point2f> test(100.0, 150.0);	//correspondence point
			
			Mat target_gray = tar_gray[a];
			//makingmesh(tar[a], delaunay_color, points_tar ,pointslist);   //show mesh
			meshmatch(min, minID, a, ii, M, img, img_gray, target_gray, points, points_tar, pointslist, patch_center[ii], Pd, input_luminance, patch_dis);
			
			//circle(tar[a], Pd[ii], 3, Scalar(0,255,255), CV_FILLED, CV_AA, 0 );
			
			/*for( std::vector<Point2f>::iterator it = points_tar.begin(); it != points_tar.end(); it++)
			 {
			 draw_point(tar[a], *it, points_color);
			 }*/
			/*imshow("detabase mesh" + to_string(a), tar[a])    ;
			 waitKey(5);*/
			//imwrite("./result_DB/detabase mesh" + to_string(a) + ".bmp",tar[a]);
			
			/*if(a == data-1){
			 imshow("detabase mesh" + to_string(a),tar[a]);
			 }*/
		}
		
		if(minID[0] == -1){
		}else{
			cout << minID[0] << endl;
			int ID = minID[0];
			
			string idnumber;
			stringstream ss;
			ss << ID;
			ss >> idnumber;
			ofs << idnumber;
			ofs << std::endl;
			
			std::vector<int> offsetmap_temp(4);
			offsetmap_temp[0] = ID;
			offsetmap_temp[1] = ii;
			offsetmap_temp[2] = (int)Pd[ii][minID[0]].x;
			offsetmap_temp[3] = (int)Pd[ii][minID[0]].y;
			offsetmap.push_back(offsetmap_temp);
		}
	}
	
	cout << "complete making offsetmap!" << endl;
	cout << "texture synthesis" << endl;
	for(int ii = 0; ii < offsetmap.size(); ii ++){
		stringstream databaseID;
		databaseID << databasename_dir <<  "tex/" << offsetmap[ii][0] << ".bmp";
		Mat tar_temp = imread(databaseID.str());
		Mat patch = Mat (M+1,M+1,CV_8UC3,Scalar(0,0,0));
		int patchnumber = offsetmap[ii][1];

		stringstream databaseID_normal;
		databaseID_normal << databasename_dir << "normal/" << offsetmap[ii][0] << ".bmp";
		Mat tar_temp_nor = imread(databaseID_normal.str());
		Mat patch_nor = Mat(M + 1, M + 1, CV_8UC3, Scalar(0, 0, 0));
		int patchnumber_nor = offsetmap[ii][1];
		
		//patch tiling----------
		//case of criterion is patch left top
		/*for(int y = 0; y <= M; y++){
		 for(int x = 0; x <= M; x++){
		 Vec3b getpixel = tar_temp.at<Vec3b>(y + offsetmap[ii][3], x + offsetmap[ii][2]);
		 patch.at<Vec3b>(y, x) = getpixel;
		 }
		 }
		 
		 myBlending(patch, result, Point(patch_center[patchnumber].x, patch_center[patchnumber].y),result);*/
		
		
		//case of criterion is patch center
		for(int y = -M/2; y <= M/2; y++){
			for(int x = -M/2; x <= M/2; x++){
				Vec3b getpixel = tar_temp.at<Vec3b>(y + offsetmap[ii][3], x + offsetmap[ii][2]);
				patch.at<Vec3b>(y + M/2, x + M/2) = getpixel;

				Vec3b getpixel_nor = tar_temp_nor.at<Vec3b>(y + offsetmap[ii][3], x + offsetmap[ii][2]);
				patch_nor.at<Vec3b>(y + M / 2, x + M / 2) = getpixel_nor;
			}
		}
		
		myBlending(patch, result, Point(patch_center[patchnumber].x - (M/2), patch_center[patchnumber].y - (M/2)),result);
		myBlending(patch_nor, result_nor, Point(patch_center[patchnumber_nor].x - (M / 2), patch_center[patchnumber_nor].y - (M / 2)), result_nor);
		
	}
	
	for (int i=0; i<offsetmap.size(); i++) {
		stringstream databaseID;
		databaseID << databasename_dir <<  "tex/" << offsetmap[i][0] << ".bmp";
		Mat tar_temp = imread(databaseID.str());
		Mat patch = Mat (M+1,M+1,CV_8UC3,Scalar(0,0,0));
		int patchnumber = offsetmap[i][1];

		stringstream databaseID_nor;
		databaseID_nor << databasename_dir << "normal/" << offsetmap[i][0] << ".bmp";
		Mat tar_temp_nor = imread(databaseID.str());
		Mat patch_nor = Mat(M + 1, M + 1, CV_8UC3, Scalar(0, 0, 0));
		int patchnumber_nor = offsetmap[i][1];
		
		//case of criterion is patch left top
		/*for(int y = 0; y <= M; y++){
		 for(int x = 0; x <= M; x++){
		 Vec3b getpixel = tar_temp.at<Vec3b>(y + offsetmap[i][3], x + offsetmap[i][2]);
		 patch.at<Vec3b>(y, x) = getpixel;
		 }
		 }
		 Point pt(patch_center[patchnumber].x, patch_center[patchnumber].y);
		 
		 mySeamlessClone(patch, result, pt);*/
		
		//case of criterion is patch center
		for(int y = -M/2; y <= M/2; y++){
			for(int x = -M/2; x <= M/2; x++){
				Vec3b getpixel = tar_temp.at<Vec3b>(y + offsetmap[i][3], x + offsetmap[i][2]);
				patch.at<Vec3b>(y + M/2, x + M/2) = getpixel;
				//result.at<Vec3b>(y + M/2, x + M/2) = getpixel;

				Vec3b getpixel_nor = tar_temp_nor.at<Vec3b>(y + offsetmap[i][3], x + offsetmap[i][2]);
				patch_nor.at<Vec3b>(y + M / 2, x + M / 2) = getpixel_nor;
			}
		}
		Point pt(patch_center[patchnumber].x - M/2, patch_center[patchnumber].y - M/2);
		Point pt_nor(patch_center[patchnumber_nor].x - M / 2, patch_center[patchnumber_nor].y - M / 2);
		
		mySeamlessClone(patch, result, pt);
		mySeamlessClone(patch_nor, result_nor, pt);
	}
	ofs.close();
	result = denoisefilter(result, imgcopy);
   // result_nor = denoisefilter(result_nor, imgcopy);
	
	// Draw points-----------
	/* for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
	 {
	 draw_point(img, *it, points_color);
	 }*/
	
	//imshow("input", imgcopy);
	//imshow( "input mesh", img);
	//imwrite("./output/" + input  +"_mesh.bmp" ,img);
	cv::Mat res2;
    //境界線でカット
    result_nor = pb3d_tsukasa::cvt_data(result_nor, dlib_Result[1]);

	cv::bilateralFilter(result_nor, res2, 5, 20, 200);
    res2 = pb3d_tsukasa::cvt_data(res2, dlib_Result[1]);

	//imshow("result", result);
	//imshow("result_nor", result_nor);
	//imshow("result_nor_blur", res2);
	imwrite(output_dir + input + "_tex.bmp",result);
	imwrite(output_dir + input + "_normal_result.bmp", result_nor);
	imwrite(output_dir + input + "_normal_blur_result.bmp", res2);
	cv::imwrite(output_dir + input + ".bmp", Input_ori);
	cout << "end" << endl;

	//最小二乗誤差最小化
	cv::Mat_<cv::Vec3b> img3 = result_nor;
	const auto maps2 = createNormalMapsFromImage(img3, cv::Vec3b(0, 0, 0), 2);

	SimplePoissonDepthReconstructor rec2;
	Eigen::MatrixXd depthMap2 = rec2.fromNormals(maps2.NX, maps2.NY, maps2.NZ, maps2.V);

	Eigen::IOFormat csvFmt2(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", "");
	//std::ofstream ofs("Output/[" + name + "]_depth_output.csv");
	std::ofstream ofs3(output_dir + input + "_ori.csv");
	ofs3 << depthMap2.format(csvFmt2);
  
	
	//最小二乗誤差最小化
	cv::Mat_<cv::Vec3b> img2 = res2;
	const auto maps = createNormalMapsFromImage(img2, cv::Vec3b(0, 0, 0), 2);

	//SimplePoissonDepthReconstructor rec;
	//Eigen::MatrixXd depthMap = rec.fromNormals(maps.NX, maps.NY, maps.NZ, maps.V);
	//Eigen::IOFormat csvFmt(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", "");
	////std::ofstream ofs("Output/[" + name + "]_depth_output.csv");
	//std::ofstream ofs2(output_dir + input + ".csv");
	//ofs2 << depthMap.format(csvFmt);

	return maps;
	
}

NormalMapTuple reconst_tsukasa(int argc, const char* argv[], NormalMapTuple estimated_maps)
{

	//入力画像の読み込み
	std::string Input_dir = argv[1];
	std::string Output_dir = argv[2];
	std::string name = argv[3];
	cv::Mat Input = cv::imread(Input_dir + name + "_transfer_result.jpg", 1);
	cv::Mat Input_ori = cv::imread(Input_dir + name + ".jpg", 1);

	int ratio = 3;

	//入力画像の正規化
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	proxy_deserialize hoge = deserialize("shape_predictor_68_face_landmarks.dat");
	hoge >> pose_model;

	std::vector<dlib::full_object_detection> shapes;
	pb3d_tsukasa::getLandmark(Input, shapes, detector, pose_model, ratio);

	std::vector<cv::Point> pts;
	pb3d_tsukasa::cvtDlib2cv(pts, shapes[0]);


	cv::Mat img_normalized = pb3d_tsukasa::Normalize_rou(Input, pb3d_tsukasa::face_rot(pts[39], pts[42]));

	std::vector<dlib::full_object_detection> shapes_2;
	pb3d_tsukasa::getLandmark(img_normalized, shapes_2, detector, pose_model, ratio);

	std::vector<cv::Point> pts_2;
	pb3d_tsukasa::cvtDlib2cv(pts_2, shapes_2[0]);

	//目頭位置をあわせる
	int dif_x = ((pts_2[42].x + pts_2[39].x) / 2) - (img_size / 2);
	int dif_y = 110 - ((pts_2[42].y + pts_2[39].y) / 2);

	//cv::Mat New_Tex(cv::Size(size, size), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat New_Tex(cv::Size(img_size, img_size), CV_8UC3, cv::Scalar(0, 0, 0));

	for (int j = 0; j < img_size; j++)
	{
		for (int i = 0; i < img_size; i++)
		{
			//if(0 <= j-dif_y && j-dif_y < img_normalized.rows && 0 <= i-dif_x && i-dif_x < img_normalized.cols)
			{
				//cv::Vec3b BGR = temp6.at<cv::Vec3b>(j-dif_y, i + dif_x);
				//New_Tex.at<cv::Vec3b>(j, i) = BGR;
				cv::Vec3b bgr = img_normalized.at<cv::Vec3b>(j - dif_y, i + dif_x);
				New_Tex.at<cv::Vec3b>(j, i) = bgr;
			}

		}
	}

	//cv::imshow("input", Input);
	//cv::imshow("normlize", img_normalized);
	//cv::imshow("normlize_cut", New_Tex);


	//マスクの作成（dlib_Result[0] : fp_image, dlib_Result[1] : mask_img ）
	std::vector<cv::Mat> dlib_Result = pb3d_tsukasa::makemask(New_Tex);




	//パッチタイリング結果（Result[0] : RGB, Result[1] : Normal ）
	std::vector<cv::Mat> Result = PatchBasedNormalEstimation_iteration(New_Tex, dlib_Result[1], estimated_maps);

	cv::cvtColor(New_Tex, New_Tex, CV_RGB2GRAY);

	////////結果確認
	//cv::imshow("input", New_Tex);
	//cv::imshow("RGB", Result[0]);
	//cv::imshow("Normal", Result[1]);
	//cv::imshow("Mask", dlib_Result[1]);
	//cv::imshow("Feature Points", dlib_Result[0]);

	cv::imwrite(Output_dir + name + ".bmp", Input_ori);
	cv::imwrite(Output_dir + name + "_Normalized_Input.bmp", New_Tex);
	cv::imwrite(Output_dir + name + "_RGB_output.bmp", Result[0]);
	cv::imwrite(Output_dir + name + "_Normal_output.bmp", Result[1]);
	cv::imwrite(Output_dir + name + "_Mask_output.bmp", dlib_Result[1]);
	cv::imwrite(Output_dir + name + "_FeaturePoints_output.bmp", dlib_Result[0]);

	//最小二乗誤差最小化
	cv::Mat_<cv::Vec3b> img = Result[1];
	const auto maps = createNormalMapsFromImage(img, cv::Vec3b(0, 0, 0), 2);

	//SimplePoissonDepthReconstructor rec;
	//Eigen::MatrixXd depthMap = rec.fromNormals(maps.NX, maps.NY, maps.NZ, maps.V);
	//Eigen::IOFormat csvFmt(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", "");
	//std::ofstream ofs(Output_dir + name + ".csv");
	//ofs << depthMap.format(csvFmt);

}

//法線マップから3次元形状を推定する
void reconst_from_norm(NormalMapTuple maps, const char* argv[])
{
	std::string Output_dir = argv[2];
	std::string name = argv[3];

	SimplePoissonDepthReconstructor rec;
	Eigen::MatrixXd depthMap = rec.fromNormals(maps.NX, maps.NY, maps.NZ, maps.V);

	Eigen::IOFormat csvFmt(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", "");
	std::ofstream ofs(Output_dir + name + ".csv");
	ofs << depthMap.format(csvFmt);
}

void main(int argc, const char* argv[])
{
	//ゆーぶん手法で初期形状の推定
	NormalMapTuple estimated_maps = reconst_yubun(argc, argv);

	//つかさの手法で密な対応を推定、これをイテレーション回して推定結果を徐々に修正
	//NormalMapTuple estimated_maps = reconst_tsukasa(argc, argv, estimated_maps);
	//NormalMapTuple estimated_maps = reconst_tsukasa(argc, argv, estimated_maps);
	//NormalMapTuple estimated_maps = reconst_tsukasa(argc, argv, estimated_maps);

	//三次元形状を法線情報から復元
	reconst_from_norm(estimated_maps, argv);
}