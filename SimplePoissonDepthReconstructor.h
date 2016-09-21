#pragma once
#include <Eigen/Eigen>
class SimplePoissonDepthReconstructor
{
public:
	struct DiffMapPair{
		Eigen::MatrixXd diffX;
		Eigen::MatrixXd diffY;
		DiffMapPair(){};
		DiffMapPair(Eigen::MatrixXd& diffX, Eigen::MatrixXd& diffY)
		:diffX(diffX), diffY(diffY){}
	};

public:
	SimplePoissonDepthReconstructor();
	~SimplePoissonDepthReconstructor();

	const Eigen::MatrixXd fromNormals(
		const Eigen::MatrixXd& normalX,
		const Eigen::MatrixXd& normalY,
		const Eigen::MatrixXd& normalZ,
		const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& mask);

	const DiffMapPair cvtNormals2DiffMapPair(
		const Eigen::MatrixXd& normalX,
		const Eigen::MatrixXd& normalY,
		const Eigen::MatrixXd& normalZ,
		const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& mask);
};

