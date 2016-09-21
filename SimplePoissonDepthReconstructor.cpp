#include "SimplePoissonDepthReconstructor.h"
#include "IndexTable.h"
#include <vector>
#include <Eigen/Sparse>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXBool;
using Eigen::SparseMatrix;

SimplePoissonDepthReconstructor::SimplePoissonDepthReconstructor()
{
}


SimplePoissonDepthReconstructor::~SimplePoissonDepthReconstructor()
{
}

const MatrixXd SimplePoissonDepthReconstructor::fromNormals(
	const MatrixXd& normalX, const MatrixXd& normalY, const MatrixXd& normalZ, const MatrixXBool& mask)
{
	const int rows = normalX.rows();
	const int cols = normalX.cols();
	const int len = mask.count();

	//差分マップ
	const DiffMapPair dp = cvtNormals2DiffMapPair(normalX, normalY, normalZ, mask);
	const MatrixXd dX = dp.diffX;
	const MatrixXd dY = dp.diffY;


	//ピクセル番号
	IndexTable<MatrixXBool> idxTable(mask, rows, cols);
	

	//指定ピクセルが有効か無効か調べるLambda式
	auto isValid = [rows, cols, mask](const int i, const int j) -> bool {
		if (i < 0 || i >= rows || j < 0 || j >= cols){ return false; }
		if (!mask(i, j)){ return false; }
		return true;
	};

	//----

	std::vector<Eigen::Triplet<double>> triplets;

	VectorXd rhs = VectorXd::Zero(len);
	for (int k = 0; k < len; k++){
		const int y = idxTable[k].y;
		const int x = idxTable[k].x;

		int nNeighbor = 0;


		//右
		if (isValid(y, x + 1)){
			nNeighbor++;
			triplets.emplace_back(idxTable(y, x), idxTable(y, x + 1), 1);
			rhs[k] += dX(y, x);
		}

		//左
		if (isValid(y, x - 1)){
			nNeighbor++;
			triplets.emplace_back(idxTable(y, x), idxTable(y, x - 1), 1);
			rhs[k] -= dX(y, x - 1);
		}

		//下
		if (isValid(y + 1, x)){
			nNeighbor++;
			triplets.emplace_back(idxTable(y, x), idxTable(y + 1, x), 1);
			rhs[k] += dY(y, x);
		}

		//上
		if (isValid(y - 1, x)){
			nNeighbor++;
			triplets.emplace_back(idxTable(y, x), idxTable(y - 1, x), 1);
			rhs[k] -= dY(y - 1, x);
		}

		//z0(0番目のピクセルの深度) = 0 の制約を加える
		if (k == 0) {
			nNeighbor++;
			//rhs[k] += 0.0;	//基準を変えたい時にはrhsも足す
		}

		triplets.emplace_back(idxTable(y, x), idxTable(y, x), -nNeighbor);
	}
	

	SparseMatrix<double> A(len, len);
	A.setFromTriplets(triplets.cbegin(), triplets.cend());

	VectorXd z_vec(len);	//求めた深度が格納される

	//Ax = b
	Eigen::SimplicialLLT<SparseMatrix<double>> solver(-A);
	z_vec = solver.solve(-rhs);


	MatrixXd depthMap = MatrixXd::Zero(rows, cols);
	idxTable.copyVec2Mat(z_vec, depthMap);

	return depthMap;
}


const SimplePoissonDepthReconstructor::DiffMapPair SimplePoissonDepthReconstructor::cvtNormals2DiffMapPair(
	const MatrixXd& normalX, const MatrixXd& normalY, const MatrixXd& normalZ, const MatrixXBool& mask)
{

	const int rows = normalX.rows();
	const int cols = normalX.cols();

	Eigen::MatrixXd diffX = Eigen::MatrixXd::Zero(rows, cols);
	Eigen::MatrixXd diffY = Eigen::MatrixXd::Zero(rows, cols);

	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			if (mask(i, j)){
				diffX(i, j) = -normalX(i, j) / normalZ(i, j);
				diffY(i, j) = -normalY(i, j) / normalZ(i, j);
			}
		}
	}

	return DiffMapPair(diffX, diffY);
}