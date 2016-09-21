#include "pb3d_tsukasa.h"



pb3d_tsukasa::pb3d_tsukasa()
{
}


void pb3d_tsukasa::getLandmark(cv::Mat& img,
    std::vector<dlib::full_object_detection>& shapes,
    dlib::frontal_face_detector& detector,
    dlib::shape_predictor& pose_model,
    float ratio)
{
    cv::Mat temp;
    cv::resize(img, temp, cv::Size(), 1.0 / ratio, 1.0 / ratio);

    dlib::cv_image<dlib::bgr_pixel> cimg_t(temp);
    dlib::cv_image<dlib::bgr_pixel> cimg(img);

    // Detect faces //
    std::vector<dlib::rectangle> faces = detector(cimg_t);

    for (unsigned long i = 0; i < faces.size(); ++i) {
        faces[i].set_top(faces[i].top() * ratio);
        faces[i].set_left(faces[i].left() * ratio);
        faces[i].set_right(faces[i].right() * ratio);
        faces[i].set_bottom(faces[i].bottom() * ratio);
    }

    // Find the pose of each face.
    //std::vector<full_object_detection> shapes;
    for (unsigned long i = 0; i < faces.size(); ++i)
        shapes.push_back(pose_model(cimg, faces[i]));

};

void pb3d_tsukasa::cross(cv::Mat& dst, cv::Point pt, double length, cv::Scalar color, double angle = 45.0, double thickness = 1)
{

    double theta = M_PI / 180.0 * angle;
    double lx = length / 2.0 * cos(theta);
    double ly = length / 2.0 * sin(theta);

    cv::Point pt1(pt.x + lx, pt.y + ly);
    cv::Point pt2(pt.x - lx, pt.y - ly);
    cv::line(dst, pt1, pt2, color);

    pt1 = cv::Point(pt.x - lx, pt.y + ly);
    pt2 = cv::Point(pt.x + lx, pt.y - ly);
    cv::line(dst, pt1, pt2, color, thickness);

};


cv::Point pb3d_tsukasa::ptd2cv(dlib::full_object_detection shape, unsigned long i)
{
    return cv::Point(shape.part(i).x(), shape.part(i).y());
};


int pb3d_tsukasa::draw_shapes(cv::Mat &dst, dlib::full_object_detection shape)
{

    const int size = shape.num_parts();

    cv::Scalar blue(255, 0, 0);
    cv::Scalar green(0, 255, 0);
    cv::Scalar red(0, 0, 255);

    if (size == 68) {
        cv::Point pt1, pt2;

        for (unsigned long i = 0; i < 16; ++i)
            cv::line(dst, ptd2cv(shape, i), ptd2cv(shape, i + 1), green);

        for (unsigned long i = 27; i < 30; ++i)
            cv::line(dst, ptd2cv(shape, i), ptd2cv(shape, i + 1), green);

        for (unsigned long i = 17; i < 21; ++i)
            cv::line(dst, ptd2cv(shape, i), ptd2cv(shape, i + 1), green);

        for (unsigned long i = 22; i < 26; ++i)
            cv::line(dst, ptd2cv(shape, i), ptd2cv(shape, i + 1), green);

        for (unsigned long i = 30; i < 35; ++i)
            cv::line(dst, ptd2cv(shape, i), ptd2cv(shape, i + 1), green);
        cv::line(dst, ptd2cv(shape, 35), ptd2cv(shape, 30), green);

        for (unsigned long i = 36; i < 41; ++i)
            cv::line(dst, ptd2cv(shape, i), ptd2cv(shape, i + 1), green);
        cv::line(dst, ptd2cv(shape, 41), ptd2cv(shape, 36), green);

        for (unsigned long i = 42; i < 47; ++i)
            cv::line(dst, ptd2cv(shape, i), ptd2cv(shape, i + 1), green);
        cv::line(dst, ptd2cv(shape, 47), ptd2cv(shape, 42), green);

        for (unsigned long i = 48; i < 59; ++i)
            cv::line(dst, ptd2cv(shape, i), ptd2cv(shape, i + 1), green);
        cv::line(dst, ptd2cv(shape, 59), ptd2cv(shape, 48), green);

        for (unsigned long i = 60; i < 67; ++i)
            cv::line(dst, ptd2cv(shape, i), ptd2cv(shape, i + 1), green);
        cv::line(dst, ptd2cv(shape, 67), ptd2cv(shape, 60), green);

        // landmarks
        for (unsigned long i = 0; i <= 67; ++i)
            cross(dst, ptd2cv(shape, i), 10, red);

    }
    else {
        std::cout << "the dat cannot be applied" << std::endl;
    }

    return 0;
};

void pb3d_tsukasa::cvtDlib2cv(std::vector<cv::Point>& pts, dlib::full_object_detection& shape)
{

    for (int i = 0; i<68; i++) {
        pts.push_back(ptd2cv(shape, i));
    }

};


std::vector<cv::Mat> pb3d_tsukasa::makemask(cv::Mat &Input)
{
    std::vector<cv::Mat> dlib_result;

    try
    {
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor pose_model;
        dlib::proxy_deserialize hoge = dlib::deserialize("shape_predictor_68_face_landmarks.dat");
        hoge >> pose_model;

        std::vector<dlib::full_object_detection> shapes;
        getLandmark(Input, shapes, detector, pose_model, 3);

        std::vector<cv::Point> pts;
        cvtDlib2cv(pts, shapes[0]);

        cv::Mat dst = Input.clone();
        draw_shapes(dst, shapes[0]);

        cv::Point vec0 = (pts[27] - pts[8]);
        cv::Point vec0_new = cv::Point(vec0.x / 6, vec0.y / 6);

        cv::Point vecyoko = (pts[39] - pts[42]);
        cv::Point vecyoko_new = cv::Point(vecyoko.x / 4, vecyoko.y / 4);

        cv::Point vectate_new = cv::Point(vec0.x / 12, vec0.y / 12);

        std::vector<cv::Point> addpts;
        addpts.push_back(pts[26] + vec0_new);
        addpts.push_back(pts[24] + vec0_new);
        //addpts.push_back(pts[27] + vec0/3);
        addpts.push_back(pts[19] + vec0_new);
        addpts.push_back(pts[17] + vec0_new);

        cv::Scalar green(0, 255, 0);
        for (int i = 0; i<addpts.size(); i++)
            cross(dst, addpts[i], 10, green);

        std::vector<cv::Point> maskPoly;
        for (int i = 0; i<17; i++)
        {
            if (0 <= i  && i <= 5)
            {
                maskPoly.push_back(pts[i] - vecyoko_new);

            }
            else if (6 <= i && i < 11)
            {
                maskPoly.push_back(pts[i] + vectate_new);
            }
            else if (11 <= i)
            {
                maskPoly.push_back(pts[i] + vecyoko_new);
            }

        }
        for (int i = 0; i<addpts.size(); i++) maskPoly.push_back(addpts[i]);

        cv::Mat dst2 = cv::Mat(Input.size(), CV_8UC3, cv::Scalar::all(0));
        cv::fillConvexPoly(dst2, maskPoly.data(), (int)maskPoly.size(), cv::Scalar(255, 255, 255));

        std::vector<cv::Point> right_eye_Poly;
        for (int i = 36; i<42; i++) right_eye_Poly.push_back(pts[i]);
        //for (int i=0; i<addpts.size(); i++) right_eye_Poly.push_back(addpts[i]);

        //cv::Mat dst2 = cv::Mat(img.size(), CV_8UC3, cv::Scalar::all(0));
        //cv::fillConvexPoly(dst2, right_eye_Poly.data(), (int)right_eye_Poly.size(), cv::Scalar(0,0,0));

        std::vector<cv::Point> left_eye_Poly;
        for (int i = 42; i<47; i++) left_eye_Poly.push_back(pts[i]);
        //for (int i=0; i<addpts.size(); i++) maskPoly.push_back(addpts[i]);

        //cv::Mat dst2 = cv::Mat(img.size(), CV_8UC3, cv::Scalar::all(0));
        //cv::fillConvexPoly(dst2, left_eye_Poly.data(), (int)left_eye_Poly.size(), cv::Scalar(0,0,0));

        //std::string out = "out";
        //for (unsigned long i = 0; i < shapes.size(); ++i) {
        //	//my::draw_shapes(dst, shapes[i]);
        //	maketxt(n, shapes[i], out);
        //}


        //cv::imshow("input", Input);
        //cv::imshow("dst", dst);
        //cv::imshow("mask", dst2);
        //cv::imwrite("out/fp_" + FileName0.str() + ".bmp", dst);
        //cv::imwrite("out/mask_eye_" + FileName0.str() + ".bmp", dst2);
        dlib_result.push_back(dst);
        dlib_result.push_back(dst2);
        //cv::waitKey(15);
    }
    catch (dlib::serialization_error& e)
    {
        std::cout << "You need dlib's default face landmarking model file to run this example." << std::endl;
        std::cout << "You can get it from the following URL: " << std::endl;
        std::cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << std::endl;
        std::cout << std::endl << e.what() << std::endl;
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }


    return dlib_result;
}

//マスク処理
cv::Mat pb3d_tsukasa::cvt_data(const cv::Mat& Data, const cv::Mat& Mask)
{
    cv::Vec3b BLACK(0, 0, 0);
    cv::Vec3b WHITE(1.0, 1.0, 1.0);

    cv::Mat New_Data;
    Data.copyTo(New_Data);

    for (int j = 0; j < Data.rows; j++)
    {
        for (int i = 0; i < Data.cols; i++)
        {
            cv::Vec3b Mask_color = Mask.at<cv::Vec3b>(j, i);

            if (Mask_color == BLACK)
            {
                if (Data.channels() == 3)
                {
                    New_Data.at<cv::Vec3b>(j, i) = BLACK;
                }
                else
                {
                    New_Data.at<uchar>(j, i) = 0;
                }

            }
            else
            {
                continue;
            }

        }
    }

    return New_Data;
}

void pb3d_tsukasa::maketxt(std::string input, dlib::full_object_detection shape, std::string name) {

    std::stringstream   FileName0;
    FileName0 << input << name;

    std::string txt = ".txt";
    std::string txtname = FileName0.str();
    //string txtname;
    //stringstream ss;
    //ss << a;
    //ss >> txtname;

    std::ofstream ofs;
    //if(a == 99999){
    //ofs.open("out/" + name + txt, std::ios::out);
    //}else{
    ofs.open(txtname + txt, std::ios::out);
    //}
    for (unsigned long i = 0; i <= 67; ++i) {
        //cout<<shape.part(i).x()<<" "<< shape.part(i).y()<<endl;

        ofs << shape.part(i).x() << " " << shape.part(i).y();
        ofs << std::endl;
    }
    if (!ofs.is_open()) {
        std::cout << "faild" << std::endl;
    }
    ofs.close();

};

//回転行列
std::vector<float> pb3d_tsukasa::face_rot(cv::Point eye_R, cv::Point eye_L)
{

	std::vector<float> mat;
	double eyes_d, eyes_d_x, eyes_d_y;
	eyes_d = hypot(eye_R.x - eye_L.x, eye_R.y - eye_L.y);
	// eyes_d_xは絶対に正である必要がある
	eyes_d_x = eye_L.x - eye_R.x;
	// eyes_d_yはこちらから見て右目が下の時は－，右目が上の時は＋になっている必要がある
	eyes_d_y = eye_R.y - eye_L.y;

	Eigen::MatrixXd temp_pre(2, 1), temp_aft(2, 1);  // 回転行列，一時的にx,y座標を格納する行列
	Eigen::MatrixXd rot(2, 2);  //回転行列
	rot << eyes_d_x, -eyes_d_y, eyes_d_y, eyes_d_x;  //OKAOの座標軸は通常と逆なのでそのまま回転行列をかけると逆回転をさせたことになる
	rot = rot.array() / eyes_d;  // 斜辺の長さで割ることによりcos,sinにする

	float kakudo = acos(rot(0, 0));

	if (eyes_d_y < 0)
	{
		kakudo = kakudo / M_PI * 180.0;
	}
	else
	{
		kakudo = -1 * kakudo / M_PI * 180.0;
	}
	mat.push_back(kakudo);
	mat.push_back(eyes_d);
	//return rot;
	return mat;
}



cv::Mat pb3d_tsukasa::Normalize_rou(cv::Mat input_img, std::vector<float> &mat_set)
{
	int d = 50;

	cv::Mat normalize_img;

	//auto mat_set = mat(src_x, src_y);

	// 回転, スケーリング
	float angle = mat_set[0];
	float scale = d / mat_set[1];
	//float scale = 1.0;
	// 中心：画像中心
	cv::Point2f center(input_img.cols*0.5, input_img.rows*0.5);
	// 以上の条件から2次元の回転行列を計算
	const cv::Mat affine_matrix = cv::getRotationMatrix2D(center, angle, scale);

	//cv::Mat temp2_New;
	cv::warpAffine(input_img, normalize_img, affine_matrix, input_img.size());

	return normalize_img;

}

pb3d_tsukasa::~pb3d_tsukasa()
{
}
