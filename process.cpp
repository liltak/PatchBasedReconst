//
//  process.cpp
//  real_portraits
//
//  Created by yu-bun on 2016/04/07.
//  Copyright © 2016年 yu-bun. All rights reserved.
//

#include "process.hpp"


Mat image_preparation(Mat& image)
{
    Mat image_gray( Size(image.rows, image.cols), CV_8UC1, 1 );
    //imshow("",image);
    //cvWaitKey();
    
    cvtColor(image, image_gray,CV_BGR2GRAY);
    GaussianBlur(image_gray, image_gray, cv::Size(15,15), 0.5, 0.5);
    image_gray.convertTo(image_gray,CV_32F,1/255.0);
    
    return image_gray;
}


Mat denoisefilter(cv::Mat src, cv::Mat tar)
{
    cv::Mat dst = cv::Mat ( src.rows, src.cols, CV_8UC3, cv::Scalar(0,0,0));
    for(int y = 0; y < src.rows; y++){
        for(int x = 0; x < src.cols; x++){
            int r = 0, g = 0, b = 0;
            r = src.at<cv::Vec3b>(y,x)[2];
            g = src.at<cv::Vec3b>(y,x)[1];
            b = src.at<cv::Vec3b>(y,x)[0];
            if(r == 0 && g == 0 && b == 0){
                dst.at<cv::Vec3b>(y,x)[2] = tar.at<cv::Vec3b>(y,x)[2];
                dst.at<cv::Vec3b>(y,x)[1] = tar.at<cv::Vec3b>(y,x)[1];
                dst.at<cv::Vec3b>(y,x)[0] = tar.at<cv::Vec3b>(y,x)[0];
            }else{
                dst.at<cv::Vec3b>(y,x)[2] = src.at<cv::Vec3b>(y,x)[2];
                dst.at<cv::Vec3b>(y,x)[1] = src.at<cv::Vec3b>(y,x)[1];
                dst.at<cv::Vec3b>(y,x)[0] = src.at<cv::Vec3b>(y,x)[0];
            }
            
        }
    }
    return dst;
}


//make target image for seamlessblending
void myBlending(Mat src, Mat dst, Point p, Mat& output) {
    src.convertTo(src, CV_32FC3, 1/255.0);
    dst.convertTo(dst, CV_32FC3, 1/255.0);
    output = dst.clone();
    for (int i=0; i<src.rows; i++) {
        for (int j=0; j<src.cols; j++) {
            int y = i + p.y;
            int x = j + p.x;
            if (x < 0 || x >= dst.cols || y < 0 || y >= dst.rows) continue;
            
            cv::Vec3f src_rgb = src.at<cv::Vec3f>(i,j);
            cv::Vec3f dst_rgb = dst.at<cv::Vec3f>(y,x);
            cv::Vec3f out_rgb;
            if (dst.at<cv::Vec3f>(y,x) == cv::Vec3f(0,0,0)) {
                out_rgb = src_rgb;
            }
            else {
                out_rgb = src_rgb * 0.5 + dst_rgb * 0.5;
            }
            
            output.at<cv::Vec3f>(y,x) = out_rgb;
        }
    }
    output.convertTo(output, CV_8UC3, 255);
}

//seamlessblending
void mySeamlessClone(Mat tile, Mat &dst, Point pt, int off) {
    cv::Rect rect(pt, tile.size());
    cv::Mat roi = dst(rect);
cv:Mat mask(tile.rows+off, tile.cols+off, CV_8U, cv::Scalar::all(255));
    
    seamlessClone(tile, roi, mask, cv::Point(roi.cols/2, roi.rows/2), roi, NORMAL_CLONE);
}

//conpute input luminance average
void conpute_luminanceave(Mat& image,int M, vector<double>& input_luminance, vector<Point2f>& patch_center)
{
    for(int y = 0; y < image.rows/M; y++){
        for(int x = 0; x < image.cols/M; x++){
            float luminance = 0.0;
            
            for(int yy = y*M; yy < (y+1)*M; yy++){
                for(int xx = x*M; xx < (x+1)*M; xx++){
                    float l = image.at<float>(yy,xx);
                    luminance += l;
                }
            }
            //patch_center.push_back(Point(x*M,y*M));   //patch left top points
            patch_center.push_back(Point(x*M+M/2,y*M+M/2));   //patch center points.
            input_luminance.push_back((double) luminance /(double) (M*M));
            //input_luminance[y][x] = (double) luminance /(double) (M*M);
            //cout << "x=" << x << ",y=" << y << ",ïΩãœÇÕ"<< input_luminance[y][x] << endl;
        }
    }
}

// Draw a single point
void draw_point( Mat& img, Point2f fp, Scalar color )
{
    circle( img, fp, 2, color, CV_FILLED, CV_AA, 0 );
}


Point2f sub_vector( Point2f a, Point2f b )
{
    Point ret;
    ret.x = a.x - b.x;
    ret.y = a.y - b.y;
    return ret;
}

// making mesh model using delaunay triangle
void makingmodel( Mat& img, Subdiv2D& subdiv, Scalar delaunay_color, vector<Point2f>& points, vector<vector<int>>& pointslist )
{
    //triangleList of delaunay
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    
    //for select featurepoints from triangleList
    vector<vector<int>> pointcheck(points.size(),vector<int>(2));
    for(int i = 0; i < points.size(); i ++){
        pointcheck[i][0] = cvRound(points[i].x);
        pointcheck[i][1] = cvRound(points[i].y);
    }
    
    //parameter for using subdiv
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);
    
    //draw mesh and make mesh model-----------
    vector<Point> pt(3);   //points of triangeList(input)
    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
        for(int k =0; k < points.size(); k++){
            if(pt[0].x == pointcheck[k][0] && pt[0].y == pointcheck[k][1])
                pointslist[i][0] = k;
            if(pt[1].x == pointcheck[k][0] && pt[1].y == pointcheck[k][1])
                pointslist[i][1] = k;
            if(pt[2].x == pointcheck[k][0] && pt[2].y == pointcheck[k][1])
                pointslist[i][2] = k;
        }
        
        // Draw rectangles completely inside the image.
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
            circle(img, pt[0], 5, Scalar(0,0,255), -1, CV_AA);
        }
        
    }
    //save for inout mesh model.
    //imshow("",img);
    //imwrite("./process/ta_mesh.bmp",img);
    //waitKey();
    
}

void makingmesh (Mat& tar, Scalar delaunay_color, vector<Point2f>& points_tar, vector<vector<int>>& pointslist )
{
    Size size_tar = tar.size();
    Rect rect_tar(0,0, size_tar.width, size_tar.height);
    
    //draw DB mesh using input mesh model-----------
    vector<Point> pt1(3);   //points of triangeList(DB)
    for (int i = 0; i < pointslist.size(); i++) {
        if(pointslist[i][0] != -1 && pointslist[i][1] != -1 && pointslist[i][2] != -1){
            pt1[0].x = cvRound(points_tar[pointslist[i][0]].x); pt1[0].y = cvRound(points_tar[pointslist[i][0]].y);
            pt1[1].x = cvRound(points_tar[pointslist[i][1]].x); pt1[1].y = cvRound(points_tar[pointslist[i][1]].y);
            pt1[2].x = cvRound(points_tar[pointslist[i][2]].x); pt1[2].y = cvRound(points_tar[pointslist[i][2]].y);
            
            if ( rect_tar.contains(pt1[0]) && rect_tar.contains(pt1[1]) && rect_tar.contains(pt1[2]) )
            {
                line(tar, pt1[0], pt1[1], delaunay_color, 1, CV_AA, 0);
                line(tar, pt1[1], pt1[2], delaunay_color, 1, CV_AA, 0);
                line(tar, pt1[2], pt1[0], delaunay_color, 1, CV_AA, 0);
                circle(tar, pt1[0], 5, Scalar(0,0,255), -1, CV_AA);
            }
            
        }
    }
    //database mesh
    //imshow("", tar);
    //imwrite("./process/tar_mesh2.bmp",tar);
    //waitKey();
}

void meshmatch(vector<float>& min, vector<int>& minID, int a, int ii, int M, Mat& img,Mat& img_gray, Mat& tar_gray,
               vector<Point2f>& points,vector<Point2f>& points_tar, vector<vector<int>>& pointslist,
               Point2f& P, vector<vector<Point2f>>& Pd, vector<double>& input_luminance, vector<float>& patch_dis)
{
    //P.x = 370.0;
    //P.y = 350.0;
    //circle( img, P, 3, Scalar(0,255,255), CV_FILLED, CV_AA, 0 );
    Scalar edge_color(0,255,0);
    
    //searching mesh----------
    int meshC = 0;
    bool flag = false;
    for(int i = 0; i < pointslist.size(); i ++){
        Point2f A,B,C;
        if(pointslist[i][0] != -1 && pointslist[i][1] != -1 && pointslist[i][2] != -1){
            Point2f P_temp;
            P_temp.x = P.x + 1.0; P_temp.y = P.y + 1.0;
            A.x = cvRound(points[pointslist[i][0]].x); A.y = cvRound(points[pointslist[i][0]].y);
            B.x = cvRound(points[pointslist[i][1]].x); B.y = cvRound(points[pointslist[i][1]].y);
            C.x = cvRound(points[pointslist[i][2]].x); C.y = cvRound(points[pointslist[i][2]].y);
            
            Point2f AB = sub_vector(B, A);
            Point2f BP = sub_vector(P, B);
            
            Point2f BC = sub_vector(C, B);
            Point2f CP = sub_vector(P, C);
            
            Point2f CA = sub_vector(A, C);
            Point2f AP = sub_vector(P, A);
            
            double c1 = AB.x * BP.y - AB.y * BP.x;
            double c2 = BC.x * CP.y - BC.y * CP.x;
            double c3 = CA.x * AP.y - CA.y * AP.x;
            
            Point2f BP_temp = sub_vector(P_temp, B);
            Point2f CP_temp = sub_vector(P_temp, C);
            Point2f AP_temp = sub_vector(P_temp, A);
            
            double c1_temp = AB.x * BP_temp.y - AB.y * BP_temp.x;
            double c2_temp = BC.x * CP_temp.y - BC.y * CP_temp.x;
            double c3_temp = CA.x * AP_temp.y - CA.y * AP_temp.x;
            
            if( ( c1 > 0 && c2 > 0 && c3 > 0 ) || ( c1 < 0 && c2 < 0 && c3 < 0 ) || ( c1_temp > 0 && c2_temp > 0 && c3_temp > 0 ) || ( c1_temp < 0 && c2_temp < 0 && c3_temp < 0 )) {
                 line(img, A, B, edge_color, 2, CV_AA, 0);
                 line(img, B, C, edge_color, 2, CV_AA, 0);
                 line(img, C, A, edge_color, 2, CV_AA, 0);
                meshC = i;
                flag = true;
                //imshow("",img);
                //imwrite("./process/ta_correspondence.bmp",img);
            }
        }
    }
    
    //compute correspondence point-----------
    if(flag == false){
    }else{
        Point2f A,B,C;	//input active mesh
        A.x = cvRound(points[pointslist[meshC][0]].x); A.y = cvRound(points[pointslist[meshC][0]].y);
        B.x = cvRound(points[pointslist[meshC][1]].x); B.y = cvRound(points[pointslist[meshC][1]].y);
        C.x = cvRound(points[pointslist[meshC][2]].x); C.y = cvRound(points[pointslist[meshC][2]].y);
        
        Point2f AA,BB,CC;	//DB active mesh
        AA.x = cvRound(points_tar[pointslist[meshC][0]].x); AA.y = cvRound(points_tar[pointslist[meshC][0]].y);
        BB.x = cvRound(points_tar[pointslist[meshC][1]].x); BB.y = cvRound(points_tar[pointslist[meshC][1]].y);
        CC.x = cvRound(points_tar[pointslist[meshC][2]].x); CC.y = cvRound(points_tar[pointslist[meshC][2]].y);
        
        Point2f AB = sub_vector(B, A);
        Point2f AC = sub_vector(C, A);
        Point2f AP = sub_vector(P, A);
        
        Matrix2f ABC;
        ABC << AB.x, AC.x, AB.y, AC.y;
        Vector2f b, x;
        b << AP.x, AP.y;
        FullPivLU<Matrix2f> solver(ABC);
        x = solver.solve(b);
        
        Point2f ABd = sub_vector(BB, AA);
        Point2f ACd = sub_vector(CC, AA);
        
        Pd[ii][a].x = (x(0)*ABd.x + x(1)*ACd.x)+AA.x;
        Pd[ii][a].y = (x(0)*ABd.y + x(1)*ACd.y)+AA.y;
        //cout << Pd.x << "," << Pd.y << endl;
        
        //draw correspondence mesh
         //line(tar, AA, BB, edge_color, 2, CV_AA, 0);
         //line(tar, BB, CC, edge_color, 2, CV_AA, 0);
         //line(tar, CC, AA, edge_color, 2, CV_AA, 0);
         //circle(tar, Pd[ii][a], 5, Scalar(0,255,255), -1, CV_AA);
        
        //get result of correspondence point
        //imshow("",tar);
        //imwrite("./process/db1_correspondence.bmp",tar);
        
        //start patch synthesis----------(patch left top)
        //compute DB luminance average
        //	float luminance_temp = 0.0;
        //	for(int y = 0; y < M; y++){
        //		for(int x = 0; x < M; x++){
        //			float l = tar_gray.at<float>((int)Pd[ii][a].y+y,(int)Pd[ii][a].x+x);
        //			luminance_temp += l;
        //		}
        //	}
        
        //	//luminance fitting
        //	float luminance_sum = luminance_temp/(M*M);
        //	for(int y = 0; y < M; y++){
        //		for(int x = 0; x < M; x++){
        //			float l = tar_gray.at<float>((int)Pd[ii][a].y+y,(int)Pd[ii][a].x+x);
        //			float luminance_dis = luminance_sum - input_luminance[ii];
        
        //			if(l + luminance_dis < 0.0)
        //				tar_gray.at<float>((int)Pd[ii][a].y+y,(int)Pd[ii][a].x+x) = 0.0;
        //			if(l + luminance_dis > 1.0)
        //				tar_gray.at<float>((int)Pd[ii][a].y+y,(int)Pd[ii][a].x+x) = 1.0;
        //			else
        //				tar_gray.at<float>((int)Pd[ii][a].y+y,(int)Pd[ii][a].x+x) = l + luminance_dis;
        
        //		}
        //	}
        
        //	//compute SSD
        //	for(int y = 0; y < M; y++){
        //		for(int x = 0; x < M; x++){
        //			int xx = x + (int)P.x, yy = y + (int)P.y, xxx = x + (int)Pd[ii][a].x, yyy = y + (int)Pd[ii][a].y;
        //			float l_input = img_gray.at<float>(yy, xx);
        //			float l_tar = tar_gray.at<float>(yyy, xxx);
        
        //			patch_dis[a] = patch_dis[a] + abs((l_input - l_tar)*(l_input - l_tar));
        //			//cout << patch_dis[a] << endl;
        //		}
        //	}
        
        //	if( min[0] > patch_dis[a] )
        //	{
        //		min[0] = patch_dis[a];
        //		minID[0] = a;
        //	}
        //	//cout << minID[0] << endl;
        //}
        
        //start patch synthesis----------(patch center)
        //compute DB luminance average
        float luminance_temp = 0.0;
        for(int y = -M/2; y <= M/2; y++){
        	for(int x = -M/2; x <= M/2; x++){
        		float l = tar_gray.at<float>((int)Pd[ii][a].y+y,(int)Pd[ii][a].x+x);
        		luminance_temp += l;
        	}
        }
        
        //luminance fitting
        double luminance_sum = luminance_temp/(M*M);
        for(int y = -M/2; y <= M/2; y++){
        	for(int x = -M/2; x <= M/2; x++){
        		float l = tar_gray.at<float>((int)Pd[ii][a].y+y,(int)Pd[ii][a].x+x);
        		float luminance_dis = luminance_sum - input_luminance[ii];
        
        		if(l + luminance_dis < 0.0)
        			tar_gray.at<float>((int)Pd[ii][a].y+y,(int)Pd[ii][a].x+x) = 0.0;
        		if(l + luminance_dis > 1.0)
        			tar_gray.at<float>((int)Pd[ii][a].y+y,(int)Pd[ii][a].x+x) = 1.0;
        		else
        			tar_gray.at<float>((int)Pd[ii][a].y+y,(int)Pd[ii][a].x+x) = l + luminance_dis;
        	}
        }
        
        //compute SSD
        for(int y = -M/2; y <= M/2; y++){
            for(int x = -M/2; x <= M/2; x++){
                int xx = x + (int)P.x, yy = y + (int)P.y, xxx = x + (int)Pd[ii][a].x, yyy = y + (int)Pd[ii][a].y;
                float l_input = img_gray.at<float>(yy, xx);
                float l_tar = tar_gray.at<float>(yyy, xxx);
                
                patch_dis[a] = patch_dis[a] + (l_input - l_tar)*(l_input - l_tar);
                //cout << patch_dis[a] << endl;
            }
        }
        
        //updata the suitable patch
        if( min[0] > patch_dis[a] )
        {
            min[0] = patch_dis[a];
            minID[0] = a;
        }
        //cout << minID[0] << endl;
    }
    
}

