#include "PatchBasedNormalEstimation.h"


////パッチサイズ
int Totalnumber = 16;
int PatchSize = 10;     //←右辺の数字を変更
int Overlap = PatchSize - 1;
int Difference = PatchSize - Overlap;

//重み
float g = 1;
float l_RGB = 0.01;
float l_Normal = 0.01;

//探索範囲
int Range = 5;
int range = Range + 1;

//黒，白
cv::Vec3b black(0, 0, 0);
cv::Vec3b white(255, 255, 255);

int type[3] = {1, 3, 0};


//オプション
bool cos_sim = true;

//データの読み込み
std::vector<cv::Mat> DataOpen(int datatype, int number, std::string dataname)
{
	std::vector<cv::Mat> Data;


	if(datatype == 1)
	{

		for(int i = 1; i <= number; i++ )
		{

			std::stringstream   FileName0;
			FileName0 << std::setw( 3 ) << std::setfill( '0' ) << i << ".bmp";
			cv::Mat data = cv::imread("Database/man/rgb_img/" + dataname + FileName0.str());

			cv::Mat data_gray(data.rows, data.cols, CV_8UC1);
			cvtColor(data, data_gray, CV_RGB2GRAY);
			Data.push_back( data_gray );

			//cv::Mat data_float;
			//data_gray.convertTo(data_float, CV_32F, 1.0/255);
			//Data.push_back( data_float );

		}

	}
	else if(datatype == 3)
	{

		for(int i = 1; i <= number; i++ )
		{

			std::stringstream   FileName0;
			FileName0 << std::setw( 3 ) << std::setfill( '0' ) << i << ".bmp";
			cv::Mat data = cv::imread("Database/man/normal_img/" + dataname + FileName0.str(), 1 );

			Data.push_back( data );

			//cv::Mat data_float;
			//data.convertTo(data_float, CV_32FC3, 1.0/255);
			//Data.push_back( data_float );

		}

	}
	else
	{

		for(int i = 1; i <= number; i++ )
		{

			std::stringstream   FileName0;
			FileName0 << std::setw( 3 ) << std::setfill( '0' ) << i << ".png";
			cv::Mat data = cv::imread("Database/man/Mask_img/" + dataname + FileName0.str(), 1 );

			Data.push_back( data );

			//cv::Mat data_float;
			//data.convertTo(data_float, CV_32FC3, 1.0/255);
			//Data.push_back( data_float );

		}

	}

	return Data;
}


//マスク処理
cv::Mat cvt_data(const cv::Mat& Data, const cv::Mat& Mask)
{
	cv::Vec3b BLACK(0, 0, 0);
	cv::Vec3b WHITE(1.0, 1.0, 1.0);

	cv::Mat New_Data;
	Data.copyTo(New_Data);

	for(int j = 0; j < Data.rows; j++)
	{
		for(int i = 0; i < Data.cols; i++)
		{
			cv::Vec3b Mask_color = Mask.at<cv::Vec3b>(j, i);

			if(Mask_color == black)
			{
				if(Data.channels() == 3)
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

//struct NormalMapTuple {
//	const Eigen::MatrixXd NX;
//	const Eigen::MatrixXd NY;
//	const Eigen::MatrixXd NZ;
//	const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> V;
//	NormalMapTuple(const Eigen::MatrixXd& NX, const Eigen::MatrixXd& NY, const Eigen::MatrixXd& NZ, const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& V)
//		: NX(NX), NY(NY), NZ(NZ), V(V)
//	{
//	}
//};


std::vector<cv::Mat> PatchBasedNormalEstimation(cv::Mat Input, cv::Mat Mask_img)
{

	std::vector<cv::Mat> Result;


	//dataを取り込み
	std::vector<cv::Mat> Data_L = DataOpen(type[0], Totalnumber, "");
	std::vector<cv::Mat> Data_N = DataOpen(type[1], Totalnumber, "");
	std::vector<cv::Mat> Data_Mask = DataOpen(type[3], Totalnumber, "");


	//Input
	//Mask_Image
	//cv::Mat Mask_img = cv::imread("Input/man_input_mask_1.png", 1);	

	//RGB_Image
	//cv::Mat Input_img_color = cv::imread("Input/man_input_1.bmp", 1);
	cv::Mat input_img = Input;
	cv::Mat input;
	cv::cvtColor(input_img, input,CV_RGB2GRAY);


	input = cvt_data(input, Mask_img);




	////出力
	cv::Mat dst(cv::Size(input.cols, input.rows), CV_8UC1, cv::Scalar(0, 0, 0));
	cv::Mat dst2(cv::Size(input.cols, input.rows), CV_8UC1, cv::Scalar(0, 0, 0));
	cv::Mat dst3(cv::Size(input.cols, input.rows), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat dst4(cv::Size(input.cols, input.rows), CV_8UC3, cv::Scalar(0, 0, 0));
	//cv::Mat dst5(cv::Size(input.cols, input.rows), CV_8UC3, cv::Scalar(0, 0, 0));
	//cv::Mat dst6(cv::Size(input.cols, input.rows), CV_8UC3, cv::Scalar(0, 0, 0));


	cv::Mat PatchNewtate(cv::Size(Overlap, PatchSize), CV_8UC1, cv::Scalar(0, 0, 0));
	cv::Mat PatchNewyoko(cv::Size(PatchSize, Overlap), CV_8UC1, cv::Scalar(0, 0, 0));

	cv::Mat PatchNewtate2(cv::Size(Overlap, PatchSize), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat PatchNewyoko2(cv::Size(PatchSize, Overlap), CV_8UC3, cv::Scalar(0, 0, 0));

	//cv::Mat PatchNewtate3(cv::Size(Overlap, PatchSize), CV_8UC3, cv::Scalar(0, 0, 0));
	//cv::Mat PatchNewyoko3(cv::Size(PatchSize, Overlap), CV_8UC3, cv::Scalar(0, 0, 0));


	//////パッチタイリング/////////////////////////////////////////////////////////////////////////////////////////


	////入力のパッチ
	for (int j = 0; j < input.rows/Difference; j++)
	{
		if(j*Difference + PatchSize > input.rows) continue;
		std::cout << j << std::endl;

		for (int i = 0; i < input.cols/Difference; i++)
		{
			if(i*Difference + PatchSize > input.cols) continue;

			int Black = 0;

			for(int jj = 0; jj < PatchSize; jj++)
			{
				for(int ii = 0; ii < PatchSize; ii++)
				{
					cv::Mat Patch_I(input, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));
					Black += (int)Patch_I.at<uchar>(jj, ii);
				}
			}

			//if(0 <= Black && Black < 892) continue;
			if(Black == 0) continue;

			dst.copyTo(dst2);
			dst3.copyTo(dst4);
			//dst5.copyTo(dst6);
			float min_d = 1000000000.0; 

			//input
			cv::Mat Patch_I(input, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));
			//cv::Mat Patch_I_DOG(DOG_input, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));

			//output1
			cv::Mat RGBdata_Patch_dst(dst, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));
			cv::Mat RGBdata_Patch_dst_over_tate(dst, cv::Rect(i*Difference, j*Difference, Overlap, PatchSize));
			cv::Mat RGBdata_Patch_dst_over_yoko(dst, cv::Rect(i*Difference, j*Difference, PatchSize, Overlap));

			//output2
			cv::Mat RGBdata_Patch_dst2(dst2, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));
			cv::Mat RGBdata_Patch_dst2_over_tate(dst2, cv::Rect(i*Difference, j*Difference, Overlap, PatchSize));
			cv::Mat RGBdata_Patch_dst2_over_yoko(dst2, cv::Rect(i*Difference, j*Difference, PatchSize, Overlap));

			//output3
			cv::Mat Normal_Patch_dst3(dst3, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));
			cv::Mat Normal_Patch_dst3_over_tate(dst3, cv::Rect(i*Difference, j*Difference, Overlap, PatchSize));
			cv::Mat Normal_Patch_dst3_over_yoko(dst3, cv::Rect(i*Difference, j*Difference, PatchSize, Overlap));

			//output4
			cv::Mat Normal_Patch_dst4(dst4, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));
			cv::Mat Normal_Patch_dst4_over_tate(dst4, cv::Rect(i*Difference, j*Difference, Overlap, PatchSize));
			cv::Mat Normal_Patch_dst4_over_yoko(dst4, cv::Rect(i*Difference, j*Difference, PatchSize, Overlap));


			for (int n = 0; n < Totalnumber; n++)
			{
				//if (n == N)
				//{
				//	continue;
				//}	



				for (int left = 0; left < range; left++)
				{
					if(i*Difference - left  < 0) continue;

					if(left == 0)
					{

						for (int right = 0; right < range; right++)
						{

							if(i*Difference + PatchSize + right > input.cols) continue;

							for (int up = 0; up < range; up++)
							{

								if(j*Difference - up < 0 ) continue;

								if(up == 0)
								{
									for (int down = 0; down < range; down++)
									{

										if(j*Difference + PatchSize + down > input.rows ) continue;


										////データベース中のｎ番目の写真
										////大域
										cv::Mat RGBdata_Patch(Data_L[n], cv::Rect(i*Difference + right, j*Difference + down, PatchSize, PatchSize));
										////局所
										cv::Mat RGBdata_Patch_over_tate(Data_L[n], cv::Rect(i*Difference + right, j*Difference + down, Overlap, PatchSize));
										cv::Mat RGBdata_Patch_over_yoko(Data_L[n], cv::Rect(i*Difference + right, j*Difference + down, PatchSize, Overlap));

										////データベース中のｎ番目の法線マップ
										////大域
										cv::Mat Normal_Patch(Data_N[n], cv::Rect(i*Difference + right, j*Difference + down, PatchSize, PatchSize));
										////局所
										cv::Mat Normal_Patch_over_tate(Data_N[n], cv::Rect(i*Difference + right, j*Difference + down, Overlap, PatchSize));
										cv::Mat Normal_Patch_over_yoko(Data_N[n], cv::Rect(i*Difference + right, j*Difference + down, PatchSize, Overlap));

					
										float d_g = 0;
										float d_tate = 0;
										float d_yoko = 0;

										float d_tate_M = 0;
										float d_yoko_M = 0;

										//global
										for (int j_g = 0; j_g < PatchSize; j_g++)
										{
											for (int i_g = 0; i_g < PatchSize; i_g++)
											{

												uchar data_g =RGBdata_Patch.at<uchar>(j_g, i_g);
												uchar input_g =Patch_I.at<uchar>(j_g, i_g);

												int global = (int)data_g - (int)input_g;
												d_g += global*global*3;

						

											}
										}

										//tate
										for (int j_tate = 0; j_tate < PatchSize; j_tate++)
										{

											if (i == 0)continue;

											for (int i_tate = 0; i_tate < Overlap; i_tate++)
											{

												if (i == 0)continue;


												uchar data_tate = RGBdata_Patch_over_tate.at<uchar>(j_tate, i_tate);
												uchar out_tate = RGBdata_Patch_dst2_over_tate.at<uchar>(j_tate, i_tate);

												int tate = (int)data_tate - (int)out_tate;
												d_tate += tate*tate*3;

												cv::Vec3b data_tate_M =Normal_Patch_over_tate.at<cv::Vec3b>(j_tate, i_tate);
												cv::Vec3b out_tate_M =Normal_Patch_dst4_over_tate.at<cv::Vec3b>(j_tate, i_tate);

												int tate_b = (int)data_tate_M[0] - (int)out_tate_M[0];
												int tate_g = (int)data_tate_M[1] - (int)out_tate_M[1];
												int tate_r = (int)data_tate_M[2] - (int)out_tate_M[2];

												d_tate_M += tate_b*tate_b + tate_g*tate_g + tate_r*tate_r;


											}
										}

										//yoko
										for (int j_yoko = 0; j_yoko < Overlap; j_yoko++)
										{

											if (j == 0)continue;

											for (int i_yoko = 0; i_yoko < PatchSize; i_yoko++)
											{

												if (j == 0)continue;


												uchar data_yoko =RGBdata_Patch_over_yoko.at<uchar>(j_yoko, i_yoko);
												uchar out_yoko =RGBdata_Patch_dst2_over_yoko.at<uchar>(j_yoko, i_yoko);

												int yoko = (int)data_yoko - (int)out_yoko;
												d_yoko += yoko*yoko*3;

												cv::Vec3b data_yoko_M =Normal_Patch_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);
												cv::Vec3b out_yoko_M =Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);

												int yoko_b = (int)data_yoko_M[0] - (int)out_yoko_M[0];
												int yoko_g = (int)data_yoko_M[1] - (int)out_yoko_M[1];
												int yoko_r = (int)data_yoko_M[2] - (int)out_yoko_M[2];

												d_yoko_M += yoko_b*yoko_b + yoko_g*yoko_g + yoko_r*yoko_r;


											}
										}
								if (min_d > g*d_g + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M))
								//if (min_d > g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog))
									//if(min_d > g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog) )
								{
									//int number = n;

									min_d = g*d_g + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M);
									//min_d = g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
									//min_d = g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
											RGBdata_Patch.copyTo(RGBdata_Patch_dst);
											Normal_Patch.copyTo(Normal_Patch_dst3);
											//DOG_Patch.copyTo(DOG_Patch_dst5);

											//縦補完
											for (int k_tate = 0; k_tate < PatchSize; k_tate++)
											{

												if (i == 0)continue;

												for (int l_tate = 0; l_tate < Overlap; l_tate++)
												{

													if (i == 0)continue;

													int tate_blend = ((int)RGBdata_Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) + (int)RGBdata_Patch_over_tate.at<uchar>(k_tate, l_tate)) / 2;
													//std::cout << (int)Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) << (int)Patch_over_tate.at<uchar>(k_tate, l_tate) << tate_blend << std::endl;
													PatchNewtate.at<uchar>(k_tate, l_tate) = tate_blend;
													PatchNewtate.copyTo(RGBdata_Patch_dst_over_tate);

													cv::Vec3b tate_out_M = Normal_Patch_dst4_over_tate.at<cv::Vec3b>(k_tate, l_tate);
													cv::Vec3b tate_D_M = Normal_Patch_over_tate.at<cv::Vec3b>(k_tate, l_tate);
													cv::Vec3b tate_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
													tate_blend_M[0] = (((int)tate_D_M[0])+((int)tate_out_M[0]))/2;
													tate_blend_M[1] = (((int)tate_D_M[1])+((int)tate_out_M[1]))/2;
													tate_blend_M[2] = (((int)tate_D_M[2])+((int)tate_out_M[2]))/2;

													PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate) = tate_blend_M;
													PatchNewtate2.copyTo(Normal_Patch_dst3_over_tate);

													//cv::Vec3b tate_out_dog = DOG_Patch_dst6_over_tate.at<cv::Vec3b>(k_tate, l_tate);
													//cv::Vec3b tate_D_dog = DOG_Patch_over_tate.at<cv::Vec3b>(k_tate, l_tate);
													//cv::Vec3b tate_blend_dog;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
													//tate_blend_dog[0] = (((int)tate_D_dog[0])+((int)tate_out_dog[0]))/2;
													//tate_blend_dog[1] = (((int)tate_D_dog[1])+((int)tate_out_dog[1]))/2;
													//tate_blend_dog[2] = (((int)tate_D_dog[2])+((int)tate_out_dog[2]))/2;

													//PatchNewtate3.at<cv::Vec3b>(k_tate, l_tate) = tate_blend_dog;
													//PatchNewtate3.copyTo(DOG_Patch_dst5_over_tate);


												}
											}

											for (int k_yoko = 0; k_yoko < Overlap; k_yoko++)
											{

												if (j == 0)continue;

												for (int l_yoko = 0; l_yoko < PatchSize; l_yoko++)
												{

													if (j == 0)continue;

													int yoko_blend	= ((int)RGBdata_Patch_dst2_over_yoko.at<uchar>(k_yoko, l_yoko) + (int)RGBdata_Patch_over_yoko.at<uchar>(k_yoko, l_yoko)) / 2;
													PatchNewyoko.at<uchar>(k_yoko, l_yoko) = yoko_blend;                                    
													PatchNewyoko.copyTo(RGBdata_Patch_dst_over_yoko);

													cv::Vec3b yoko_out_M = Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
													cv::Vec3b yoko_D_M = Normal_Patch_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
													cv::Vec3b yoko_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
													yoko_blend_M[0] = (((int)yoko_D_M[0])+((int)yoko_out_M[0]))/2;
													yoko_blend_M[1] = (((int)yoko_D_M[1])+((int)yoko_out_M[1]))/2;
													yoko_blend_M[2] = (((int)yoko_D_M[2])+((int)yoko_out_M[2]))/2;

													PatchNewyoko2.at<cv::Vec3b>(k_yoko, l_yoko) = yoko_blend_M;
													PatchNewyoko2.copyTo(Normal_Patch_dst3_over_yoko);

												}
											}
										}	

									}//d


								}//if up= 0 end
								else
								{



									////データベース中のｎ番目の写真
									////大域
									cv::Mat RGBdata_Patch(Data_L[n], cv::Rect(i*Difference + right, j*Difference - up, PatchSize, PatchSize));
									////局所
									cv::Mat RGBdata_Patch_over_tate(Data_L[n], cv::Rect(i*Difference + right, j*Difference - up, Overlap, PatchSize));
									cv::Mat RGBdata_Patch_over_yoko(Data_L[n], cv::Rect(i*Difference + right, j*Difference - up, PatchSize, Overlap));

									////データベース中のｎ番目の法線マップ
									////大域
									cv::Mat Normal_Patch(Data_N[n], cv::Rect(i*Difference + right, j*Difference - up, PatchSize, PatchSize));
									////局所
									cv::Mat Normal_Patch_over_tate(Data_N[n], cv::Rect(i*Difference + right, j*Difference - up, Overlap, PatchSize));
									cv::Mat Normal_Patch_over_yoko(Data_N[n], cv::Rect(i*Difference + right, j*Difference - up, PatchSize, Overlap));

									//////データベース中のｎ番目のDOG
									//////大域
									//cv::Mat DOG_Patch(Data_HF[n], cv::Rect(i*Difference + right, j*Difference - up, PatchSize, PatchSize));
									//////局所
									//cv::Mat DOG_Patch_over_tate(Data_HF[n], cv::Rect(i*Difference + right, j*Difference - up, Overlap, PatchSize));
									//cv::Mat DOG_Patch_over_yoko(Data_HF[n], cv::Rect(i*Difference + right, j*Difference - up, PatchSize, Overlap));

									float d_g = 0;
									float d_tate = 0;
									float d_yoko = 0;

									float d_tate_M = 0;
									float d_yoko_M = 0;

									//float d_dog = 0;
									//float d_tate_dog = 0;
									//float d_yoko_dog = 0;

									//global
									for (int j_g = 0; j_g < PatchSize; j_g++)
									{
										for (int i_g = 0; i_g < PatchSize; i_g++)
										{

											uchar data_g =RGBdata_Patch.at<uchar>(j_g, i_g);
											uchar input_g =Patch_I.at<uchar>(j_g, i_g);

											int global = (int)data_g - (int)input_g;
											d_g += global*global*3;


										}
									}

									//tate
									for (int j_tate = 0; j_tate < PatchSize; j_tate++)
									{

										if (i == 0)continue;

										for (int i_tate = 0; i_tate < Overlap; i_tate++)
										{

											if (i == 0)continue;


											uchar data_tate = RGBdata_Patch_over_tate.at<uchar>(j_tate, i_tate);
											uchar out_tate = RGBdata_Patch_dst2_over_tate.at<uchar>(j_tate, i_tate);

											int tate = (int)data_tate - (int)out_tate;
											d_tate += tate*tate*3;

											cv::Vec3b data_tate_M =Normal_Patch_over_tate.at<cv::Vec3b>(j_tate, i_tate);
											cv::Vec3b out_tate_M =Normal_Patch_dst4_over_tate.at<cv::Vec3b>(j_tate, i_tate);

											int tate_b = (int)data_tate_M[0] - (int)out_tate_M[0];
											int tate_g = (int)data_tate_M[1] - (int)out_tate_M[1];
											int tate_r = (int)data_tate_M[2] - (int)out_tate_M[2];

											d_tate_M += tate_b*tate_b + tate_g*tate_g + tate_r*tate_r;

										}
									}

									//yoko
									for (int j_yoko = 0; j_yoko < Overlap; j_yoko++)
									{

										if (j == 0)continue;

										for (int i_yoko = 0; i_yoko < PatchSize; i_yoko++)
										{

											if (j == 0)continue;


											uchar data_yoko =RGBdata_Patch_over_yoko.at<uchar>(j_yoko, i_yoko);
											uchar out_yoko =RGBdata_Patch_dst2_over_yoko.at<uchar>(j_yoko, i_yoko);

											int yoko = (int)data_yoko - (int)out_yoko;
											d_yoko += yoko*yoko*3;

											cv::Vec3b data_yoko_M =Normal_Patch_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);
											cv::Vec3b out_yoko_M =Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);

											int yoko_b = (int)data_yoko_M[0] - (int)out_yoko_M[0];
											int yoko_g = (int)data_yoko_M[1] - (int)out_yoko_M[1];
											int yoko_r = (int)data_yoko_M[2] - (int)out_yoko_M[2];

											d_yoko_M += yoko_b*yoko_b + yoko_g*yoko_g + yoko_r*yoko_r;


										}
									}

								if (min_d > g*d_g + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M))
								//if (min_d > g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog))
									//if(min_d > g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog) )
								{
									//int number = n;

									min_d = g*d_g + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M);
									//min_d = g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
									//min_d = g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
										RGBdata_Patch.copyTo(RGBdata_Patch_dst);
										Normal_Patch.copyTo(Normal_Patch_dst3);
										//DOG_Patch.copyTo(DOG_Patch_dst5);

										//縦補完
										for (int k_tate = 0; k_tate < PatchSize; k_tate++)
										{

											if (i == 0)continue;

											for (int l_tate = 0; l_tate < Overlap; l_tate++)
											{

												if (i == 0)continue;

												int tate_blend = ((int)RGBdata_Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) + (int)RGBdata_Patch_over_tate.at<uchar>(k_tate, l_tate)) / 2;
												//std::cout << (int)Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) << (int)Patch_over_tate.at<uchar>(k_tate, l_tate) << tate_blend << std::endl;
												PatchNewtate.at<uchar>(k_tate, l_tate) = tate_blend;
												PatchNewtate.copyTo(RGBdata_Patch_dst_over_tate);

												cv::Vec3b tate_out_M = Normal_Patch_dst4_over_tate.at<cv::Vec3b>(k_tate, l_tate);
												cv::Vec3b tate_D_M = Normal_Patch_over_tate.at<cv::Vec3b>(k_tate, l_tate);
												cv::Vec3b tate_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
												tate_blend_M[0] = (((int)tate_D_M[0])+((int)tate_out_M[0]))/2;
												tate_blend_M[1] = (((int)tate_D_M[1])+((int)tate_out_M[1]))/2;
												tate_blend_M[2] = (((int)tate_D_M[2])+((int)tate_out_M[2]))/2;

												PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate) = tate_blend_M;
												PatchNewtate2.copyTo(Normal_Patch_dst3_over_tate);



											}
										}

										for (int k_yoko = 0; k_yoko < Overlap; k_yoko++)
										{

											if (j == 0)continue;

											for (int l_yoko = 0; l_yoko < PatchSize; l_yoko++)
											{

												if (j == 0)continue;

												int yoko_blend	= ((int)RGBdata_Patch_dst2_over_yoko.at<uchar>(k_yoko, l_yoko) + (int)RGBdata_Patch_over_yoko.at<uchar>(k_yoko, l_yoko)) / 2;
												PatchNewyoko.at<uchar>(k_yoko, l_yoko) = yoko_blend;                                    
												PatchNewyoko.copyTo(RGBdata_Patch_dst_over_yoko);

												cv::Vec3b yoko_out_M = Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
												cv::Vec3b yoko_D_M = Normal_Patch_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
												cv::Vec3b yoko_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
												yoko_blend_M[0] = (((int)yoko_D_M[0])+((int)yoko_out_M[0]))/2;
												yoko_blend_M[1] = (((int)yoko_D_M[1])+((int)yoko_out_M[1]))/2;
												yoko_blend_M[2] = (((int)yoko_D_M[2])+((int)yoko_out_M[2]))/2;

												PatchNewyoko2.at<cv::Vec3b>(k_yoko, l_yoko) = yoko_blend_M;
												PatchNewyoko2.copyTo(Normal_Patch_dst3_over_yoko);


											}
										}
									}	

								}//if up != 0 end
							}//u
						}//r

					}//if left = 0 end
					else
					{



						for (int up = 0; up < range; up++)
						{
							if(j*Difference - up < 0 ) continue;

							if(up == 0)
							{
								for (int down = 0; down < range; down++)
								{

									if(j*Difference + PatchSize + down > input.rows ) continue;


									////データベース中のｎ番目の写真
									////大域
									cv::Mat RGBdata_Patch(Data_L[n], cv::Rect(i*Difference - left, j*Difference + down, PatchSize, PatchSize));
									////局所
									cv::Mat RGBdata_Patch_over_tate(Data_L[n], cv::Rect(i*Difference - left, j*Difference + down, Overlap, PatchSize));
									cv::Mat RGBdata_Patch_over_yoko(Data_L[n], cv::Rect(i*Difference - left, j*Difference + down, PatchSize, Overlap));

									////データベース中のｎ番目の法線マップ
									////大域
									cv::Mat Normal_Patch(Data_N[n], cv::Rect(i*Difference - left, j*Difference + down, PatchSize, PatchSize));
									////局所
									cv::Mat Normal_Patch_over_tate(Data_N[n], cv::Rect(i*Difference - left, j*Difference + down, Overlap, PatchSize));
									cv::Mat Normal_Patch_over_yoko(Data_N[n], cv::Rect(i*Difference - left, j*Difference + down, PatchSize, Overlap));


									float d_g = 0;
									float d_tate = 0;
									float d_yoko = 0;

									float d_tate_M = 0;
									float d_yoko_M = 0;

									//float d_dog = 0;
									//float d_tate_dog = 0;
									//float d_yoko_dog = 0;

									//global
									for (int j_g = 0; j_g < PatchSize; j_g++)
									{
										for (int i_g = 0; i_g < PatchSize; i_g++)
										{

											uchar data_g =RGBdata_Patch.at<uchar>(j_g, i_g);
											uchar input_g =Patch_I.at<uchar>(j_g, i_g);

											int global = (int)data_g - (int)input_g;
											d_g += global*global*3;



										}
									}

									//tate
									for (int j_tate = 0; j_tate < PatchSize; j_tate++)
									{

										if (i == 0)continue;

										for (int i_tate = 0; i_tate < Overlap; i_tate++)
										{

											if (i == 0)continue;


											uchar data_tate = RGBdata_Patch_over_tate.at<uchar>(j_tate, i_tate);
											uchar out_tate = RGBdata_Patch_dst2_over_tate.at<uchar>(j_tate, i_tate);

											int tate = (int)data_tate - (int)out_tate;
											d_tate += tate*tate*3;

											cv::Vec3b data_tate_M =Normal_Patch_over_tate.at<cv::Vec3b>(j_tate, i_tate);
											cv::Vec3b out_tate_M =Normal_Patch_dst4_over_tate.at<cv::Vec3b>(j_tate, i_tate);

											int tate_b = (int)data_tate_M[0] - (int)out_tate_M[0];
											int tate_g = (int)data_tate_M[1] - (int)out_tate_M[1];
											int tate_r = (int)data_tate_M[2] - (int)out_tate_M[2];

											d_tate_M += tate_b*tate_b + tate_g*tate_g + tate_r*tate_r;



										}
									}

									//yoko
									for (int j_yoko = 0; j_yoko < Overlap; j_yoko++)
									{

										if (j == 0)continue;

										for (int i_yoko = 0; i_yoko < PatchSize; i_yoko++)
										{

											if (j == 0)continue;


											uchar data_yoko =RGBdata_Patch_over_yoko.at<uchar>(j_yoko, i_yoko);
											uchar out_yoko =RGBdata_Patch_dst2_over_yoko.at<uchar>(j_yoko, i_yoko);

											int yoko = (int)data_yoko - (int)out_yoko;
											d_yoko += yoko*yoko*3;

											cv::Vec3b data_yoko_M =Normal_Patch_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);
											cv::Vec3b out_yoko_M =Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);

											int yoko_b = (int)data_yoko_M[0] - (int)out_yoko_M[0];
											int yoko_g = (int)data_yoko_M[1] - (int)out_yoko_M[1];
											int yoko_r = (int)data_yoko_M[2] - (int)out_yoko_M[2];

											d_yoko_M += yoko_b*yoko_b + yoko_g*yoko_g + yoko_r*yoko_r;


										}
									}

								if (min_d > g*d_g + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M))
								//if (min_d > g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog))
									//if(min_d > g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog) )
								{
									//int number = n;

									min_d = g*d_g + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M);
									//min_d = g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
									//min_d = g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
										RGBdata_Patch.copyTo(RGBdata_Patch_dst);
										Normal_Patch.copyTo(Normal_Patch_dst3);
										//DOG_Patch.copyTo(DOG_Patch_dst5);

										//縦補完
										for (int k_tate = 0; k_tate < PatchSize; k_tate++)
										{

											if (i == 0)continue;

											for (int l_tate = 0; l_tate < Overlap; l_tate++)
											{

												if (i == 0)continue;

												int tate_blend = ((int)RGBdata_Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) + (int)RGBdata_Patch_over_tate.at<uchar>(k_tate, l_tate)) / 2;
												//std::cout << (int)Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) << (int)Patch_over_tate.at<uchar>(k_tate, l_tate) << tate_blend << std::endl;
												PatchNewtate.at<uchar>(k_tate, l_tate) = tate_blend;
												PatchNewtate.copyTo(RGBdata_Patch_dst_over_tate);

												cv::Vec3b tate_out_M = Normal_Patch_dst4_over_tate.at<cv::Vec3b>(k_tate, l_tate);
												cv::Vec3b tate_D_M = Normal_Patch_over_tate.at<cv::Vec3b>(k_tate, l_tate);
												cv::Vec3b tate_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
												tate_blend_M[0] = (((int)tate_D_M[0])+((int)tate_out_M[0]))/2;
												tate_blend_M[1] = (((int)tate_D_M[1])+((int)tate_out_M[1]))/2;
												tate_blend_M[2] = (((int)tate_D_M[2])+((int)tate_out_M[2]))/2;

												PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate) = tate_blend_M;
												PatchNewtate2.copyTo(Normal_Patch_dst3_over_tate);


											}
										}

										for (int k_yoko = 0; k_yoko < Overlap; k_yoko++)
										{

											if (j == 0)continue;

											for (int l_yoko = 0; l_yoko < PatchSize; l_yoko++)
											{

												if (j == 0)continue;

												int yoko_blend	= ((int)RGBdata_Patch_dst2_over_yoko.at<uchar>(k_yoko, l_yoko) + (int)RGBdata_Patch_over_yoko.at<uchar>(k_yoko, l_yoko)) / 2;
												PatchNewyoko.at<uchar>(k_yoko, l_yoko) = yoko_blend;                                    
												PatchNewyoko.copyTo(RGBdata_Patch_dst_over_yoko);

												cv::Vec3b yoko_out_M = Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
												cv::Vec3b yoko_D_M = Normal_Patch_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
												cv::Vec3b yoko_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
												yoko_blend_M[0] = (((int)yoko_D_M[0])+((int)yoko_out_M[0]))/2;
												yoko_blend_M[1] = (((int)yoko_D_M[1])+((int)yoko_out_M[1]))/2;
												yoko_blend_M[2] = (((int)yoko_D_M[2])+((int)yoko_out_M[2]))/2;

												PatchNewyoko2.at<cv::Vec3b>(k_yoko, l_yoko) = yoko_blend_M;
												PatchNewyoko2.copyTo(Normal_Patch_dst3_over_yoko);


											}
										}
									}	

								}//d


							}//if up= 0 end
							else
							{


								////データベース中のｎ番目の写真
								////大域
								cv::Mat RGBdata_Patch(Data_L[n], cv::Rect(i*Difference - left, j*Difference - up, PatchSize, PatchSize));
								////局所
								cv::Mat RGBdata_Patch_over_tate(Data_L[n], cv::Rect(i*Difference - left, j*Difference - up, Overlap, PatchSize));
								cv::Mat RGBdata_Patch_over_yoko(Data_L[n], cv::Rect(i*Difference - left, j*Difference - up, PatchSize, Overlap));

								////データベース中のｎ番目の法線マップ
								////大域
								cv::Mat Normal_Patch(Data_N[n], cv::Rect(i*Difference - left, j*Difference - up, PatchSize, PatchSize));
								////局所
								cv::Mat Normal_Patch_over_tate(Data_N[n], cv::Rect(i*Difference - left, j*Difference - up, Overlap, PatchSize));
								cv::Mat Normal_Patch_over_yoko(Data_N[n], cv::Rect(i*Difference - left, j*Difference - up, PatchSize, Overlap));

								float d_g = 0;
								float d_tate = 0;
								float d_yoko = 0;

								float d_tate_M = 0;
								float d_yoko_M = 0;

								//global
								for (int j_g = 0; j_g < PatchSize; j_g++)
								{
									for (int i_g = 0; i_g < PatchSize; i_g++)
									{

										uchar data_g =RGBdata_Patch.at<uchar>(j_g, i_g);
										uchar input_g =Patch_I.at<uchar>(j_g, i_g);

										int global = (int)data_g - (int)input_g;
										d_g += global*global*3;


									}
								}

								//tate
								for (int j_tate = 0; j_tate < PatchSize; j_tate++)
								{

									if (i == 0)continue;

									for (int i_tate = 0; i_tate < Overlap; i_tate++)
									{

										if (i == 0)continue;


										uchar data_tate = RGBdata_Patch_over_tate.at<uchar>(j_tate, i_tate);
										uchar out_tate = RGBdata_Patch_dst2_over_tate.at<uchar>(j_tate, i_tate);

										int tate = (int)data_tate - (int)out_tate;
										d_tate += tate*tate*3;

										cv::Vec3b data_tate_M =Normal_Patch_over_tate.at<cv::Vec3b>(j_tate, i_tate);
										cv::Vec3b out_tate_M =Normal_Patch_dst4_over_tate.at<cv::Vec3b>(j_tate, i_tate);

										int tate_b = (int)data_tate_M[0] - (int)out_tate_M[0];
										int tate_g = (int)data_tate_M[1] - (int)out_tate_M[1];
										int tate_r = (int)data_tate_M[2] - (int)out_tate_M[2];

										d_tate_M += tate_b*tate_b + tate_g*tate_g + tate_r*tate_r;

									}
								}

								//yoko
								for (int j_yoko = 0; j_yoko < Overlap; j_yoko++)
								{

									if (j == 0)continue;

									for (int i_yoko = 0; i_yoko < PatchSize; i_yoko++)
									{

										if (j == 0)continue;


										uchar data_yoko =RGBdata_Patch_over_yoko.at<uchar>(j_yoko, i_yoko);
										uchar out_yoko =RGBdata_Patch_dst2_over_yoko.at<uchar>(j_yoko, i_yoko);

										int yoko = (int)data_yoko - (int)out_yoko;
										d_yoko += yoko*yoko*3;

										cv::Vec3b data_yoko_M =Normal_Patch_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);
										cv::Vec3b out_yoko_M =Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);

										int yoko_b = (int)data_yoko_M[0] - (int)out_yoko_M[0];
										int yoko_g = (int)data_yoko_M[1] - (int)out_yoko_M[1];
										int yoko_r = (int)data_yoko_M[2] - (int)out_yoko_M[2];

										d_yoko_M += yoko_b*yoko_b + yoko_g*yoko_g + yoko_r*yoko_r;


									}
								}

								if (min_d > g*d_g + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M))
								//if (min_d > g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog))
									//if(min_d > g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog) )
								{
									//int number = n;

									min_d = g*d_g + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M);
									//min_d = g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
									//min_d = g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
									RGBdata_Patch.copyTo(RGBdata_Patch_dst);
									Normal_Patch.copyTo(Normal_Patch_dst3);
									//DOG_Patch.copyTo(DOG_Patch_dst5);

									//縦補完
									for (int k_tate = 0; k_tate < PatchSize; k_tate++)
									{

										if (i == 0)continue;

										for (int l_tate = 0; l_tate < Overlap; l_tate++)
										{

											if (i == 0)continue;

											int tate_blend = ((int)RGBdata_Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) + (int)RGBdata_Patch_over_tate.at<uchar>(k_tate, l_tate)) / 2;
											//std::cout << (int)Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) << (int)Patch_over_tate.at<uchar>(k_tate, l_tate) << tate_blend << std::endl;
											PatchNewtate.at<uchar>(k_tate, l_tate) = tate_blend;
											PatchNewtate.copyTo(RGBdata_Patch_dst_over_tate);

											cv::Vec3b tate_out_M = Normal_Patch_dst4_over_tate.at<cv::Vec3b>(k_tate, l_tate);
											cv::Vec3b tate_D_M = Normal_Patch_over_tate.at<cv::Vec3b>(k_tate, l_tate);
											cv::Vec3b tate_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
											tate_blend_M[0] = (((int)tate_D_M[0])+((int)tate_out_M[0]))/2;
											tate_blend_M[1] = (((int)tate_D_M[1])+((int)tate_out_M[1]))/2;
											tate_blend_M[2] = (((int)tate_D_M[2])+((int)tate_out_M[2]))/2;


										}
									}

									for (int k_yoko = 0; k_yoko < Overlap; k_yoko++)
									{

										if (j == 0)continue;

										for (int l_yoko = 0; l_yoko < PatchSize; l_yoko++)
										{

											if (j == 0)continue;

											int yoko_blend	= ((int)RGBdata_Patch_dst2_over_yoko.at<uchar>(k_yoko, l_yoko) + (int)RGBdata_Patch_over_yoko.at<uchar>(k_yoko, l_yoko)) / 2;
											PatchNewyoko.at<uchar>(k_yoko, l_yoko) = yoko_blend;                                    
											PatchNewyoko.copyTo(RGBdata_Patch_dst_over_yoko);

											cv::Vec3b yoko_out_M = Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
											cv::Vec3b yoko_D_M = Normal_Patch_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
											cv::Vec3b yoko_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
											yoko_blend_M[0] = (((int)yoko_D_M[0])+((int)yoko_out_M[0]))/2;
											yoko_blend_M[1] = (((int)yoko_D_M[1])+((int)yoko_out_M[1]))/2;
											yoko_blend_M[2] = (((int)yoko_D_M[2])+((int)yoko_out_M[2]))/2;

											PatchNewyoko2.at<cv::Vec3b>(k_yoko, l_yoko) = yoko_blend_M;
											PatchNewyoko2.copyTo(Normal_Patch_dst3_over_yoko);


										}
									}
								}	

							}//if up != 0 end
						}//u

					}//if left != 0 end

				}//l end

			}//n end

		}//i end
	}//j end



	//境界線でカット
	dst = cvt_data(dst, Mask_img);
	dst3 = cvt_data(dst3, Mask_img);

	Result.push_back(dst);
	Result.push_back(dst3);

	return Result;

}

std::vector<cv::Mat> PatchBasedNormalEstimation_iteration(cv::Mat Input, cv::Mat Mask_img, NormalMapTuple maps)
{

	std::vector<cv::Mat> Result;


	//dataを取り込み
	std::vector<cv::Mat> Data_L = DataOpen(type[0], Totalnumber, "");
	std::vector<cv::Mat> Data_N = DataOpen(type[1], Totalnumber, "");
	std::vector<cv::Mat> Data_Mask = DataOpen(type[3], Totalnumber, "");


	//Input
	//Mask_Image
	//cv::Mat Mask_img = cv::imread("Input/man_input_mask_1.png", 1);	

	//RGB_Image
	//cv::Mat Input_img_color = cv::imread("Input/man_input_1.bmp", 1);
	cv::Mat input_img = Input;
	cv::Mat input;
	cv::cvtColor(input_img, input, CV_RGB2GRAY);


	input = cvt_data(input, Mask_img);




	////出力
	cv::Mat dst(cv::Size(input.cols, input.rows), CV_8UC1, cv::Scalar(0, 0, 0));
	cv::Mat dst2(cv::Size(input.cols, input.rows), CV_8UC1, cv::Scalar(0, 0, 0));
	cv::Mat dst3(cv::Size(input.cols, input.rows), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat dst4(cv::Size(input.cols, input.rows), CV_8UC3, cv::Scalar(0, 0, 0));
	//cv::Mat dst5(cv::Size(input.cols, input.rows), CV_8UC3, cv::Scalar(0, 0, 0));
	//cv::Mat dst6(cv::Size(input.cols, input.rows), CV_8UC3, cv::Scalar(0, 0, 0));


	cv::Mat PatchNewtate(cv::Size(Overlap, PatchSize), CV_8UC1, cv::Scalar(0, 0, 0));
	cv::Mat PatchNewyoko(cv::Size(PatchSize, Overlap), CV_8UC1, cv::Scalar(0, 0, 0));

	cv::Mat PatchNewtate2(cv::Size(Overlap, PatchSize), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat PatchNewyoko2(cv::Size(PatchSize, Overlap), CV_8UC3, cv::Scalar(0, 0, 0));

	//cv::Mat PatchNewtate3(cv::Size(Overlap, PatchSize), CV_8UC3, cv::Scalar(0, 0, 0));
	//cv::Mat PatchNewyoko3(cv::Size(PatchSize, Overlap), CV_8UC3, cv::Scalar(0, 0, 0));


	//////パッチタイリング/////////////////////////////////////////////////////////////////////////////////////////


	////入力のパッチ
	for (int j = 0; j < input.rows / Difference; j++)
	{
		if (j*Difference + PatchSize > input.rows) continue;
		std::cout << j << std::endl;

		for (int i = 0; i < input.cols / Difference; i++)
		{
			if (i*Difference + PatchSize > input.cols) continue;

			int Black = 0;

			for (int jj = 0; jj < PatchSize; jj++)
			{
				for (int ii = 0; ii < PatchSize; ii++)
				{
					cv::Mat Patch_I(input, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));
					Black += (int)Patch_I.at<uchar>(jj, ii);
				}
			}

			//if(0 <= Black && Black < 892) continue;
			if (Black == 0) continue;

			dst.copyTo(dst2);
			dst3.copyTo(dst4);
			//dst5.copyTo(dst6);
			float min_d = 1000000000.0;

			//input
			cv::Mat Patch_I(input, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));
			//cv::Mat Patch_I_DOG(DOG_input, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));

			//output1
			cv::Mat RGBdata_Patch_dst(dst, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));
			cv::Mat RGBdata_Patch_dst_over_tate(dst, cv::Rect(i*Difference, j*Difference, Overlap, PatchSize));
			cv::Mat RGBdata_Patch_dst_over_yoko(dst, cv::Rect(i*Difference, j*Difference, PatchSize, Overlap));

			//output2
			cv::Mat RGBdata_Patch_dst2(dst2, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));
			cv::Mat RGBdata_Patch_dst2_over_tate(dst2, cv::Rect(i*Difference, j*Difference, Overlap, PatchSize));
			cv::Mat RGBdata_Patch_dst2_over_yoko(dst2, cv::Rect(i*Difference, j*Difference, PatchSize, Overlap));

			//output3
			cv::Mat Normal_Patch_dst3(dst3, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));
			cv::Mat Normal_Patch_dst3_over_tate(dst3, cv::Rect(i*Difference, j*Difference, Overlap, PatchSize));
			cv::Mat Normal_Patch_dst3_over_yoko(dst3, cv::Rect(i*Difference, j*Difference, PatchSize, Overlap));

			//output4
			cv::Mat Normal_Patch_dst4(dst4, cv::Rect(i*Difference, j*Difference, PatchSize, PatchSize));
			cv::Mat Normal_Patch_dst4_over_tate(dst4, cv::Rect(i*Difference, j*Difference, Overlap, PatchSize));
			cv::Mat Normal_Patch_dst4_over_yoko(dst4, cv::Rect(i*Difference, j*Difference, PatchSize, Overlap));


			for (int n = 0; n < Totalnumber; n++)
			{
				//if (n == N)
				//{
				//	continue;
				//}	



				for (int left = 0; left < range; left++)
				{
					if (i*Difference - left  < 0) continue;

					if (left == 0)
					{

						for (int right = 0; right < range; right++)
						{

							if (i*Difference + PatchSize + right > input.cols) continue;

							for (int up = 0; up < range; up++)
							{

								if (j*Difference - up < 0) continue;

								if (up == 0)
								{
									for (int down = 0; down < range; down++)
									{

										if (j*Difference + PatchSize + down > input.rows) continue;


										////データベース中のｎ番目の写真
										////大域
										cv::Mat RGBdata_Patch(Data_L[n], cv::Rect(i*Difference + right, j*Difference + down, PatchSize, PatchSize));
										////局所
										cv::Mat RGBdata_Patch_over_tate(Data_L[n], cv::Rect(i*Difference + right, j*Difference + down, Overlap, PatchSize));
										cv::Mat RGBdata_Patch_over_yoko(Data_L[n], cv::Rect(i*Difference + right, j*Difference + down, PatchSize, Overlap));

										////データベース中のｎ番目の法線マップ
										////大域
										cv::Mat Normal_Patch(Data_N[n], cv::Rect(i*Difference + right, j*Difference + down, PatchSize, PatchSize));
										////局所
										cv::Mat Normal_Patch_over_tate(Data_N[n], cv::Rect(i*Difference + right, j*Difference + down, Overlap, PatchSize));
										cv::Mat Normal_Patch_over_yoko(Data_N[n], cv::Rect(i*Difference + right, j*Difference + down, PatchSize, Overlap));


										float d_g = 0;
										float d_tate = 0;
										float d_yoko = 0;

										float d_tate_M = 0;
										float d_yoko_M = 0;

										//ここから先のGlobalとLocalの意味づけ
										//Global---入力画像の輝度とデータベース上の画像の輝度が似ている制約
										//Local---前に探索したパッチと現在見ているパッチの類似度評価、色の類似度と法線の類似度

										//global
										for (int j_g = 0; j_g < PatchSize; j_g++)
										{
											for (int i_g = 0; i_g < PatchSize; i_g++)
											{

												uchar data_g = RGBdata_Patch.at<uchar>(j_g, i_g);
												uchar input_g = Patch_I.at<uchar>(j_g, i_g);

												int global = (int)data_g - (int)input_g;
												d_g += global*global * 3;



											}
										}

										//tate
										for (int j_tate = 0; j_tate < PatchSize; j_tate++)
										{

											if (i == 0)continue;

											for (int i_tate = 0; i_tate < Overlap; i_tate++)
											{

												if (i == 0)continue;

												//すでに探索した結果のパッチと輝度が似ている
												uchar data_tate = RGBdata_Patch_over_tate.at<uchar>(j_tate, i_tate);
												uchar out_tate = RGBdata_Patch_dst2_over_tate.at<uchar>(j_tate, i_tate);

												int tate = (int)data_tate - (int)out_tate;
												d_tate += tate*tate * 3;

												//すでに探索した結果のパッチと法線方向が似ている
												cv::Vec3b data_tate_M = Normal_Patch_over_tate.at<cv::Vec3b>(j_tate, i_tate);
												cv::Vec3b out_tate_M = Normal_Patch_dst4_over_tate.at<cv::Vec3b>(j_tate, i_tate);

												int tate_b = (int)data_tate_M[0] - (int)out_tate_M[0];
												int tate_g = (int)data_tate_M[1] - (int)out_tate_M[1];
												int tate_r = (int)data_tate_M[2] - (int)out_tate_M[2];

												d_tate_M += tate_b*tate_b + tate_g*tate_g + tate_r*tate_r;


											}
										}

										//yoko
										for (int j_yoko = 0; j_yoko < Overlap; j_yoko++)
										{

											if (j == 0)continue;

											for (int i_yoko = 0; i_yoko < PatchSize; i_yoko++)
											{

												if (j == 0)continue;


												uchar data_yoko = RGBdata_Patch_over_yoko.at<uchar>(j_yoko, i_yoko);
												uchar out_yoko = RGBdata_Patch_dst2_over_yoko.at<uchar>(j_yoko, i_yoko);

												int yoko = (int)data_yoko - (int)out_yoko;
												d_yoko += yoko*yoko * 3;

												cv::Vec3b data_yoko_M = Normal_Patch_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);
												cv::Vec3b out_yoko_M = Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);

												int yoko_b = (int)data_yoko_M[0] - (int)out_yoko_M[0];
												int yoko_g = (int)data_yoko_M[1] - (int)out_yoko_M[1];
												int yoko_r = (int)data_yoko_M[2] - (int)out_yoko_M[2];

												d_yoko_M += yoko_b*yoko_b + yoko_g*yoko_g + yoko_r*yoko_r;


											}
										}
										if (min_d > g*d_g + l_RGB*(d_yoko + d_tate) + l_Normal*(d_yoko_M + d_tate_M))
											//if (min_d > g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog))
											//if(min_d > g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog) )
										{
											//int number = n;

											min_d = g*d_g + l_RGB*(d_yoko + d_tate) + l_Normal*(d_yoko_M + d_tate_M);
											//min_d = g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
											//min_d = g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
											RGBdata_Patch.copyTo(RGBdata_Patch_dst);
											Normal_Patch.copyTo(Normal_Patch_dst3);
											//DOG_Patch.copyTo(DOG_Patch_dst5);

											//縦補完
											for (int k_tate = 0; k_tate < PatchSize; k_tate++)
											{

												if (i == 0)continue;

												for (int l_tate = 0; l_tate < Overlap; l_tate++)
												{

													if (i == 0)continue;

													int tate_blend = ((int)RGBdata_Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) + (int)RGBdata_Patch_over_tate.at<uchar>(k_tate, l_tate)) / 2;
													//std::cout << (int)Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) << (int)Patch_over_tate.at<uchar>(k_tate, l_tate) << tate_blend << std::endl;
													PatchNewtate.at<uchar>(k_tate, l_tate) = tate_blend;
													PatchNewtate.copyTo(RGBdata_Patch_dst_over_tate);

													cv::Vec3b tate_out_M = Normal_Patch_dst4_over_tate.at<cv::Vec3b>(k_tate, l_tate);
													cv::Vec3b tate_D_M = Normal_Patch_over_tate.at<cv::Vec3b>(k_tate, l_tate);
													cv::Vec3b tate_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
													tate_blend_M[0] = (((int)tate_D_M[0]) + ((int)tate_out_M[0])) / 2;
													tate_blend_M[1] = (((int)tate_D_M[1]) + ((int)tate_out_M[1])) / 2;
													tate_blend_M[2] = (((int)tate_D_M[2]) + ((int)tate_out_M[2])) / 2;

													PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate) = tate_blend_M;
													PatchNewtate2.copyTo(Normal_Patch_dst3_over_tate);

													//cv::Vec3b tate_out_dog = DOG_Patch_dst6_over_tate.at<cv::Vec3b>(k_tate, l_tate);
													//cv::Vec3b tate_D_dog = DOG_Patch_over_tate.at<cv::Vec3b>(k_tate, l_tate);
													//cv::Vec3b tate_blend_dog;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
													//tate_blend_dog[0] = (((int)tate_D_dog[0])+((int)tate_out_dog[0]))/2;
													//tate_blend_dog[1] = (((int)tate_D_dog[1])+((int)tate_out_dog[1]))/2;
													//tate_blend_dog[2] = (((int)tate_D_dog[2])+((int)tate_out_dog[2]))/2;

													//PatchNewtate3.at<cv::Vec3b>(k_tate, l_tate) = tate_blend_dog;
													//PatchNewtate3.copyTo(DOG_Patch_dst5_over_tate);


												}
											}

											for (int k_yoko = 0; k_yoko < Overlap; k_yoko++)
											{

												if (j == 0)continue;

												for (int l_yoko = 0; l_yoko < PatchSize; l_yoko++)
												{

													if (j == 0)continue;

													int yoko_blend = ((int)RGBdata_Patch_dst2_over_yoko.at<uchar>(k_yoko, l_yoko) + (int)RGBdata_Patch_over_yoko.at<uchar>(k_yoko, l_yoko)) / 2;
													PatchNewyoko.at<uchar>(k_yoko, l_yoko) = yoko_blend;
													PatchNewyoko.copyTo(RGBdata_Patch_dst_over_yoko);

													cv::Vec3b yoko_out_M = Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
													cv::Vec3b yoko_D_M = Normal_Patch_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
													cv::Vec3b yoko_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
													yoko_blend_M[0] = (((int)yoko_D_M[0]) + ((int)yoko_out_M[0])) / 2;
													yoko_blend_M[1] = (((int)yoko_D_M[1]) + ((int)yoko_out_M[1])) / 2;
													yoko_blend_M[2] = (((int)yoko_D_M[2]) + ((int)yoko_out_M[2])) / 2;

													PatchNewyoko2.at<cv::Vec3b>(k_yoko, l_yoko) = yoko_blend_M;
													PatchNewyoko2.copyTo(Normal_Patch_dst3_over_yoko);

												}
											}
										}

									}//d


								}//if up= 0 end
								else
								{



									////データベース中のｎ番目の写真
									////大域
									cv::Mat RGBdata_Patch(Data_L[n], cv::Rect(i*Difference + right, j*Difference - up, PatchSize, PatchSize));
									////局所
									cv::Mat RGBdata_Patch_over_tate(Data_L[n], cv::Rect(i*Difference + right, j*Difference - up, Overlap, PatchSize));
									cv::Mat RGBdata_Patch_over_yoko(Data_L[n], cv::Rect(i*Difference + right, j*Difference - up, PatchSize, Overlap));

									////データベース中のｎ番目の法線マップ
									////大域
									cv::Mat Normal_Patch(Data_N[n], cv::Rect(i*Difference + right, j*Difference - up, PatchSize, PatchSize));
									////局所
									cv::Mat Normal_Patch_over_tate(Data_N[n], cv::Rect(i*Difference + right, j*Difference - up, Overlap, PatchSize));
									cv::Mat Normal_Patch_over_yoko(Data_N[n], cv::Rect(i*Difference + right, j*Difference - up, PatchSize, Overlap));

									//////データベース中のｎ番目のDOG
									//////大域
									//cv::Mat DOG_Patch(Data_HF[n], cv::Rect(i*Difference + right, j*Difference - up, PatchSize, PatchSize));
									//////局所
									//cv::Mat DOG_Patch_over_tate(Data_HF[n], cv::Rect(i*Difference + right, j*Difference - up, Overlap, PatchSize));
									//cv::Mat DOG_Patch_over_yoko(Data_HF[n], cv::Rect(i*Difference + right, j*Difference - up, PatchSize, Overlap));

									float d_g = 0;
									float d_tate = 0;
									float d_yoko = 0;

									float d_tate_M = 0;
									float d_yoko_M = 0;

									//float d_dog = 0;
									//float d_tate_dog = 0;
									//float d_yoko_dog = 0;

									//global
									for (int j_g = 0; j_g < PatchSize; j_g++)
									{
										for (int i_g = 0; i_g < PatchSize; i_g++)
										{

											uchar data_g = RGBdata_Patch.at<uchar>(j_g, i_g);
											uchar input_g = Patch_I.at<uchar>(j_g, i_g);

											int global = (int)data_g - (int)input_g;
											d_g += global*global * 3;


										}
									}

									//tate
									for (int j_tate = 0; j_tate < PatchSize; j_tate++)
									{

										if (i == 0)continue;

										for (int i_tate = 0; i_tate < Overlap; i_tate++)
										{

											if (i == 0)continue;


											uchar data_tate = RGBdata_Patch_over_tate.at<uchar>(j_tate, i_tate);
											uchar out_tate = RGBdata_Patch_dst2_over_tate.at<uchar>(j_tate, i_tate);

											int tate = (int)data_tate - (int)out_tate;
											d_tate += tate*tate * 3;

											cv::Vec3b data_tate_M = Normal_Patch_over_tate.at<cv::Vec3b>(j_tate, i_tate);
											cv::Vec3b out_tate_M = Normal_Patch_dst4_over_tate.at<cv::Vec3b>(j_tate, i_tate);

											int tate_b = (int)data_tate_M[0] - (int)out_tate_M[0];
											int tate_g = (int)data_tate_M[1] - (int)out_tate_M[1];
											int tate_r = (int)data_tate_M[2] - (int)out_tate_M[2];

											d_tate_M += tate_b*tate_b + tate_g*tate_g + tate_r*tate_r;

										}
									}

									//yoko
									for (int j_yoko = 0; j_yoko < Overlap; j_yoko++)
									{

										if (j == 0)continue;

										for (int i_yoko = 0; i_yoko < PatchSize; i_yoko++)
										{

											if (j == 0)continue;


											uchar data_yoko = RGBdata_Patch_over_yoko.at<uchar>(j_yoko, i_yoko);
											uchar out_yoko = RGBdata_Patch_dst2_over_yoko.at<uchar>(j_yoko, i_yoko);

											int yoko = (int)data_yoko - (int)out_yoko;
											d_yoko += yoko*yoko * 3;

											cv::Vec3b data_yoko_M = Normal_Patch_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);
											cv::Vec3b out_yoko_M = Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);

											int yoko_b = (int)data_yoko_M[0] - (int)out_yoko_M[0];
											int yoko_g = (int)data_yoko_M[1] - (int)out_yoko_M[1];
											int yoko_r = (int)data_yoko_M[2] - (int)out_yoko_M[2];

											d_yoko_M += yoko_b*yoko_b + yoko_g*yoko_g + yoko_r*yoko_r;


										}
									}

									if (min_d > g*d_g + l_RGB*(d_yoko + d_tate) + l_Normal*(d_yoko_M + d_tate_M))
										//if (min_d > g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog))
										//if(min_d > g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog) )
									{
										//int number = n;

										min_d = g*d_g + l_RGB*(d_yoko + d_tate) + l_Normal*(d_yoko_M + d_tate_M);
										//min_d = g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
										//min_d = g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
										RGBdata_Patch.copyTo(RGBdata_Patch_dst);
										Normal_Patch.copyTo(Normal_Patch_dst3);
										//DOG_Patch.copyTo(DOG_Patch_dst5);

										//縦補完
										for (int k_tate = 0; k_tate < PatchSize; k_tate++)
										{

											if (i == 0)continue;

											for (int l_tate = 0; l_tate < Overlap; l_tate++)
											{

												if (i == 0)continue;

												int tate_blend = ((int)RGBdata_Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) + (int)RGBdata_Patch_over_tate.at<uchar>(k_tate, l_tate)) / 2;
												//std::cout << (int)Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) << (int)Patch_over_tate.at<uchar>(k_tate, l_tate) << tate_blend << std::endl;
												PatchNewtate.at<uchar>(k_tate, l_tate) = tate_blend;
												PatchNewtate.copyTo(RGBdata_Patch_dst_over_tate);

												cv::Vec3b tate_out_M = Normal_Patch_dst4_over_tate.at<cv::Vec3b>(k_tate, l_tate);
												cv::Vec3b tate_D_M = Normal_Patch_over_tate.at<cv::Vec3b>(k_tate, l_tate);
												cv::Vec3b tate_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
												tate_blend_M[0] = (((int)tate_D_M[0]) + ((int)tate_out_M[0])) / 2;
												tate_blend_M[1] = (((int)tate_D_M[1]) + ((int)tate_out_M[1])) / 2;
												tate_blend_M[2] = (((int)tate_D_M[2]) + ((int)tate_out_M[2])) / 2;

												PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate) = tate_blend_M;
												PatchNewtate2.copyTo(Normal_Patch_dst3_over_tate);



											}
										}

										for (int k_yoko = 0; k_yoko < Overlap; k_yoko++)
										{

											if (j == 0)continue;

											for (int l_yoko = 0; l_yoko < PatchSize; l_yoko++)
											{

												if (j == 0)continue;

												int yoko_blend = ((int)RGBdata_Patch_dst2_over_yoko.at<uchar>(k_yoko, l_yoko) + (int)RGBdata_Patch_over_yoko.at<uchar>(k_yoko, l_yoko)) / 2;
												PatchNewyoko.at<uchar>(k_yoko, l_yoko) = yoko_blend;
												PatchNewyoko.copyTo(RGBdata_Patch_dst_over_yoko);

												cv::Vec3b yoko_out_M = Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
												cv::Vec3b yoko_D_M = Normal_Patch_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
												cv::Vec3b yoko_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
												yoko_blend_M[0] = (((int)yoko_D_M[0]) + ((int)yoko_out_M[0])) / 2;
												yoko_blend_M[1] = (((int)yoko_D_M[1]) + ((int)yoko_out_M[1])) / 2;
												yoko_blend_M[2] = (((int)yoko_D_M[2]) + ((int)yoko_out_M[2])) / 2;

												PatchNewyoko2.at<cv::Vec3b>(k_yoko, l_yoko) = yoko_blend_M;
												PatchNewyoko2.copyTo(Normal_Patch_dst3_over_yoko);


											}
										}
									}

								}//if up != 0 end
							}//u
						}//r

					}//if left = 0 end
					else
					{



						for (int up = 0; up < range; up++)
						{
							if (j*Difference - up < 0) continue;

							if (up == 0)
							{
								for (int down = 0; down < range; down++)
								{

									if (j*Difference + PatchSize + down > input.rows) continue;


									////データベース中のｎ番目の写真
									////大域
									cv::Mat RGBdata_Patch(Data_L[n], cv::Rect(i*Difference - left, j*Difference + down, PatchSize, PatchSize));
									////局所
									cv::Mat RGBdata_Patch_over_tate(Data_L[n], cv::Rect(i*Difference - left, j*Difference + down, Overlap, PatchSize));
									cv::Mat RGBdata_Patch_over_yoko(Data_L[n], cv::Rect(i*Difference - left, j*Difference + down, PatchSize, Overlap));

									////データベース中のｎ番目の法線マップ
									////大域
									cv::Mat Normal_Patch(Data_N[n], cv::Rect(i*Difference - left, j*Difference + down, PatchSize, PatchSize));
									////局所
									cv::Mat Normal_Patch_over_tate(Data_N[n], cv::Rect(i*Difference - left, j*Difference + down, Overlap, PatchSize));
									cv::Mat Normal_Patch_over_yoko(Data_N[n], cv::Rect(i*Difference - left, j*Difference + down, PatchSize, Overlap));


									float d_g = 0;
									float d_tate = 0;
									float d_yoko = 0;

									float d_tate_M = 0;
									float d_yoko_M = 0;

									//float d_dog = 0;
									//float d_tate_dog = 0;
									//float d_yoko_dog = 0;

									//global
									for (int j_g = 0; j_g < PatchSize; j_g++)
									{
										for (int i_g = 0; i_g < PatchSize; i_g++)
										{

											uchar data_g = RGBdata_Patch.at<uchar>(j_g, i_g);
											uchar input_g = Patch_I.at<uchar>(j_g, i_g);

											int global = (int)data_g - (int)input_g;
											d_g += global*global * 3;



										}
									}

									//tate
									for (int j_tate = 0; j_tate < PatchSize; j_tate++)
									{

										if (i == 0)continue;

										for (int i_tate = 0; i_tate < Overlap; i_tate++)
										{

											if (i == 0)continue;


											uchar data_tate = RGBdata_Patch_over_tate.at<uchar>(j_tate, i_tate);
											uchar out_tate = RGBdata_Patch_dst2_over_tate.at<uchar>(j_tate, i_tate);

											int tate = (int)data_tate - (int)out_tate;
											d_tate += tate*tate * 3;

											cv::Vec3b data_tate_M = Normal_Patch_over_tate.at<cv::Vec3b>(j_tate, i_tate);
											cv::Vec3b out_tate_M = Normal_Patch_dst4_over_tate.at<cv::Vec3b>(j_tate, i_tate);

											int tate_b = (int)data_tate_M[0] - (int)out_tate_M[0];
											int tate_g = (int)data_tate_M[1] - (int)out_tate_M[1];
											int tate_r = (int)data_tate_M[2] - (int)out_tate_M[2];

											d_tate_M += tate_b*tate_b + tate_g*tate_g + tate_r*tate_r;



										}
									}

									//yoko
									for (int j_yoko = 0; j_yoko < Overlap; j_yoko++)
									{

										if (j == 0)continue;

										for (int i_yoko = 0; i_yoko < PatchSize; i_yoko++)
										{

											if (j == 0)continue;


											uchar data_yoko = RGBdata_Patch_over_yoko.at<uchar>(j_yoko, i_yoko);
											uchar out_yoko = RGBdata_Patch_dst2_over_yoko.at<uchar>(j_yoko, i_yoko);

											int yoko = (int)data_yoko - (int)out_yoko;
											d_yoko += yoko*yoko * 3;

											cv::Vec3b data_yoko_M = Normal_Patch_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);
											cv::Vec3b out_yoko_M = Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);

											int yoko_b = (int)data_yoko_M[0] - (int)out_yoko_M[0];
											int yoko_g = (int)data_yoko_M[1] - (int)out_yoko_M[1];
											int yoko_r = (int)data_yoko_M[2] - (int)out_yoko_M[2];

											d_yoko_M += yoko_b*yoko_b + yoko_g*yoko_g + yoko_r*yoko_r;


										}
									}

									if (min_d > g*d_g + l_RGB*(d_yoko + d_tate) + l_Normal*(d_yoko_M + d_tate_M) )
										//if (min_d > g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog))
										//if(min_d > g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog) )
									{
										//int number = n;

										min_d = g*d_g + l_RGB*(d_yoko + d_tate) + l_Normal*(d_yoko_M + d_tate_M);
										//min_d = g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
										//min_d = g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
										RGBdata_Patch.copyTo(RGBdata_Patch_dst);
										Normal_Patch.copyTo(Normal_Patch_dst3);
										//DOG_Patch.copyTo(DOG_Patch_dst5);

										//縦補完
										for (int k_tate = 0; k_tate < PatchSize; k_tate++)
										{

											if (i == 0)continue;

											for (int l_tate = 0; l_tate < Overlap; l_tate++)
											{

												if (i == 0)continue;

												int tate_blend = ((int)RGBdata_Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) + (int)RGBdata_Patch_over_tate.at<uchar>(k_tate, l_tate)) / 2;
												//std::cout << (int)Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) << (int)Patch_over_tate.at<uchar>(k_tate, l_tate) << tate_blend << std::endl;
												PatchNewtate.at<uchar>(k_tate, l_tate) = tate_blend;
												PatchNewtate.copyTo(RGBdata_Patch_dst_over_tate);

												cv::Vec3b tate_out_M = Normal_Patch_dst4_over_tate.at<cv::Vec3b>(k_tate, l_tate);
												cv::Vec3b tate_D_M = Normal_Patch_over_tate.at<cv::Vec3b>(k_tate, l_tate);
												cv::Vec3b tate_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
												tate_blend_M[0] = (((int)tate_D_M[0]) + ((int)tate_out_M[0])) / 2;
												tate_blend_M[1] = (((int)tate_D_M[1]) + ((int)tate_out_M[1])) / 2;
												tate_blend_M[2] = (((int)tate_D_M[2]) + ((int)tate_out_M[2])) / 2;

												PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate) = tate_blend_M;
												PatchNewtate2.copyTo(Normal_Patch_dst3_over_tate);


											}
										}

										for (int k_yoko = 0; k_yoko < Overlap; k_yoko++)
										{

											if (j == 0)continue;

											for (int l_yoko = 0; l_yoko < PatchSize; l_yoko++)
											{

												if (j == 0)continue;

												int yoko_blend = ((int)RGBdata_Patch_dst2_over_yoko.at<uchar>(k_yoko, l_yoko) + (int)RGBdata_Patch_over_yoko.at<uchar>(k_yoko, l_yoko)) / 2;
												PatchNewyoko.at<uchar>(k_yoko, l_yoko) = yoko_blend;
												PatchNewyoko.copyTo(RGBdata_Patch_dst_over_yoko);

												cv::Vec3b yoko_out_M = Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
												cv::Vec3b yoko_D_M = Normal_Patch_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
												cv::Vec3b yoko_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
												yoko_blend_M[0] = (((int)yoko_D_M[0]) + ((int)yoko_out_M[0])) / 2;
												yoko_blend_M[1] = (((int)yoko_D_M[1]) + ((int)yoko_out_M[1])) / 2;
												yoko_blend_M[2] = (((int)yoko_D_M[2]) + ((int)yoko_out_M[2])) / 2;

												PatchNewyoko2.at<cv::Vec3b>(k_yoko, l_yoko) = yoko_blend_M;
												PatchNewyoko2.copyTo(Normal_Patch_dst3_over_yoko);


											}
										}
									}

								}//d


							}//if up= 0 end
							else
							{


								////データベース中のｎ番目の写真
								////大域
								cv::Mat RGBdata_Patch(Data_L[n], cv::Rect(i*Difference - left, j*Difference - up, PatchSize, PatchSize));
								////局所
								cv::Mat RGBdata_Patch_over_tate(Data_L[n], cv::Rect(i*Difference - left, j*Difference - up, Overlap, PatchSize));
								cv::Mat RGBdata_Patch_over_yoko(Data_L[n], cv::Rect(i*Difference - left, j*Difference - up, PatchSize, Overlap));

								////データベース中のｎ番目の法線マップ
								////大域
								cv::Mat Normal_Patch(Data_N[n], cv::Rect(i*Difference - left, j*Difference - up, PatchSize, PatchSize));
								////局所
								cv::Mat Normal_Patch_over_tate(Data_N[n], cv::Rect(i*Difference - left, j*Difference - up, Overlap, PatchSize));
								cv::Mat Normal_Patch_over_yoko(Data_N[n], cv::Rect(i*Difference - left, j*Difference - up, PatchSize, Overlap));

								float d_g = 0;
								float d_tate = 0;
								float d_yoko = 0;

								float d_tate_M = 0;
								float d_yoko_M = 0;

								//global
								for (int j_g = 0; j_g < PatchSize; j_g++)
								{
									for (int i_g = 0; i_g < PatchSize; i_g++)
									{

										uchar data_g = RGBdata_Patch.at<uchar>(j_g, i_g);
										uchar input_g = Patch_I.at<uchar>(j_g, i_g);

										int global = (int)data_g - (int)input_g;
										d_g += global*global * 3;


									}
								}

								//tate
								for (int j_tate = 0; j_tate < PatchSize; j_tate++)
								{

									if (i == 0)continue;

									for (int i_tate = 0; i_tate < Overlap; i_tate++)
									{

										if (i == 0)continue;


										uchar data_tate = RGBdata_Patch_over_tate.at<uchar>(j_tate, i_tate);
										uchar out_tate = RGBdata_Patch_dst2_over_tate.at<uchar>(j_tate, i_tate);

										int tate = (int)data_tate - (int)out_tate;
										d_tate += tate*tate * 3;

										cv::Vec3b data_tate_M = Normal_Patch_over_tate.at<cv::Vec3b>(j_tate, i_tate);
										cv::Vec3b out_tate_M = Normal_Patch_dst4_over_tate.at<cv::Vec3b>(j_tate, i_tate);

										int tate_b = (int)data_tate_M[0] - (int)out_tate_M[0];
										int tate_g = (int)data_tate_M[1] - (int)out_tate_M[1];
										int tate_r = (int)data_tate_M[2] - (int)out_tate_M[2];

										d_tate_M += tate_b*tate_b + tate_g*tate_g + tate_r*tate_r;

									}
								}

								//yoko
								for (int j_yoko = 0; j_yoko < Overlap; j_yoko++)
								{

									if (j == 0)continue;

									for (int i_yoko = 0; i_yoko < PatchSize; i_yoko++)
									{

										if (j == 0)continue;


										uchar data_yoko = RGBdata_Patch_over_yoko.at<uchar>(j_yoko, i_yoko);
										uchar out_yoko = RGBdata_Patch_dst2_over_yoko.at<uchar>(j_yoko, i_yoko);

										int yoko = (int)data_yoko - (int)out_yoko;
										d_yoko += yoko*yoko * 3;

										cv::Vec3b data_yoko_M = Normal_Patch_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);
										cv::Vec3b out_yoko_M = Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(j_yoko, i_yoko);

										int yoko_b = (int)data_yoko_M[0] - (int)out_yoko_M[0];
										int yoko_g = (int)data_yoko_M[1] - (int)out_yoko_M[1];
										int yoko_r = (int)data_yoko_M[2] - (int)out_yoko_M[2];

										d_yoko_M += yoko_b*yoko_b + yoko_g*yoko_g + yoko_r*yoko_r;


									}
								}

								if (min_d > g*d_g + l_RGB*(d_yoko + d_tate) + l_Normal*(d_yoko_M + d_tate_M))
									//if (min_d > g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog))
									//if(min_d > g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog) )
								{
									//int number = n;

									min_d = g*d_g + l_RGB*(d_yoko + d_tate) + l_Normal*(d_yoko_M + d_tate_M);
									//min_d = g*d_g + g*d_dog + l_RGB*(d_yoko + d_tate) +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
									//min_d = g*d_dog +l_Normal*(d_yoko_M + d_tate_M) + l_dog*(d_tate_dog+ d_yoko_dog);
									RGBdata_Patch.copyTo(RGBdata_Patch_dst);
									Normal_Patch.copyTo(Normal_Patch_dst3);
									//DOG_Patch.copyTo(DOG_Patch_dst5);

									//縦補完
									for (int k_tate = 0; k_tate < PatchSize; k_tate++)
									{

										if (i == 0)continue;

										for (int l_tate = 0; l_tate < Overlap; l_tate++)
										{

											if (i == 0)continue;

											int tate_blend = ((int)RGBdata_Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) + (int)RGBdata_Patch_over_tate.at<uchar>(k_tate, l_tate)) / 2;
											//std::cout << (int)Patch_dst2_over_tate.at<uchar>(k_tate, l_tate) << (int)Patch_over_tate.at<uchar>(k_tate, l_tate) << tate_blend << std::endl;
											PatchNewtate.at<uchar>(k_tate, l_tate) = tate_blend;
											PatchNewtate.copyTo(RGBdata_Patch_dst_over_tate);

											cv::Vec3b tate_out_M = Normal_Patch_dst4_over_tate.at<cv::Vec3b>(k_tate, l_tate);
											cv::Vec3b tate_D_M = Normal_Patch_over_tate.at<cv::Vec3b>(k_tate, l_tate);
											cv::Vec3b tate_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
											tate_blend_M[0] = (((int)tate_D_M[0]) + ((int)tate_out_M[0])) / 2;
											tate_blend_M[1] = (((int)tate_D_M[1]) + ((int)tate_out_M[1])) / 2;
											tate_blend_M[2] = (((int)tate_D_M[2]) + ((int)tate_out_M[2])) / 2;


										}
									}

									for (int k_yoko = 0; k_yoko < Overlap; k_yoko++)
									{

										if (j == 0)continue;

										for (int l_yoko = 0; l_yoko < PatchSize; l_yoko++)
										{

											if (j == 0)continue;

											int yoko_blend = ((int)RGBdata_Patch_dst2_over_yoko.at<uchar>(k_yoko, l_yoko) + (int)RGBdata_Patch_over_yoko.at<uchar>(k_yoko, l_yoko)) / 2;
											PatchNewyoko.at<uchar>(k_yoko, l_yoko) = yoko_blend;
											PatchNewyoko.copyTo(RGBdata_Patch_dst_over_yoko);

											cv::Vec3b yoko_out_M = Normal_Patch_dst4_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
											cv::Vec3b yoko_D_M = Normal_Patch_over_yoko.at<cv::Vec3b>(k_yoko, l_yoko);
											cv::Vec3b yoko_blend_M;   // = PatchNewtate2.at<cv::Vec3b>(k_tate, l_tate);
											yoko_blend_M[0] = (((int)yoko_D_M[0]) + ((int)yoko_out_M[0])) / 2;
											yoko_blend_M[1] = (((int)yoko_D_M[1]) + ((int)yoko_out_M[1])) / 2;
											yoko_blend_M[2] = (((int)yoko_D_M[2]) + ((int)yoko_out_M[2])) / 2;

											PatchNewyoko2.at<cv::Vec3b>(k_yoko, l_yoko) = yoko_blend_M;
											PatchNewyoko2.copyTo(Normal_Patch_dst3_over_yoko);


										}
									}
								}

							}//if up != 0 end
						}//u

					}//if left != 0 end

				}//l end

			}//n end

		}//i end
	}//j end



	 //境界線でカット
	dst = cvt_data(dst, Mask_img);
	dst3 = cvt_data(dst3, Mask_img);

	Result.push_back(dst);
	Result.push_back(dst3);

	return Result;

}