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

	//�����}�b�v
	const DiffMapPair dp = cvtNormals2DiffMapPair(normalX, normalY, normalZ, mask);
	const MatrixXd dX = dp.diffX;
	const MatrixXd dY = dp.diffY;


	//�s�N�Z���ԍ�
	IndexTable<MatrixXBool> idxTable(mask, rows, cols);
	

	//�w��s�N�Z�����L�������������ׂ�Lambda��
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


		//�E
		if (isValid(y, x + 1)){
			nNeighbor++;
			triplets.emplace_back(idxTable(y, x), idxTable(y, x + 1), 1);
			rhs[k] += dX(y, x);
		}

		//��
		if (isValid(y, x - 1)){
			nNeighbor++;
			triplets.emplace_back(idxTable(y, x), idxTable(y, x - 1), 1);
			rhs[k] -= dX(y, x - 1);
		}

		//��
		if (isValid(y + 1, x)){
			nNeighbor++;
			triplets.emplace_back(idxTable(y, x), idxTable(y + 1, x), 1);
			rhs[k] += dY(y, x);
		}

		//��
		if (isValid(y - 1, x)){
			nNeighbor++;
			triplets.emplace_back(idxTable(y, x), idxTable(y - 1, x), 1);
			rhs[k] -= dY(y - 1, x);
		}

		//z0(0�Ԗڂ̃s�N�Z���̐[�x) = 0 �̐����������
		if (k == 0) {
			nNeighbor++;
			//rhs[k] += 0.0;	//���ς��������ɂ�rhs������
		}

		triplets.emplace_back(idxTable(y, x), idxTable(y, x), -nNeighbor);
	}
	

	SparseMatrix<double> A(len, len);
	A.setFromTriplets(triplets.cbegin(), triplets.cend());

	VectorXd z_vec(len);	//���߂��[�x���i�[�����

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