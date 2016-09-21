#pragma once
#include <vector>

template<typename _MaskType>
class IndexTable
{
public:
	struct Point{
		int y;
		int x;
		Point(const int y_, const int x_){
			y = y_;
			x = x_;
		}
	};

	static const int npos = -1;

private:
	const int _rows;
	const int _cols;
	int _len;

	
	std::vector<int> _yx2idx;	//size:rows*cols, 	-1:npos, rowMajor
	std::vector<Point> _idx2yx;	//len

private:
	IndexTable();
public:
	IndexTable(const _MaskType mask, const int rows, const int cols)
		:_rows(rows), _cols(cols)
	{
		_yx2idx.resize(_rows * _cols, -1);

		_idx2yx.reserve(_rows * _cols);

		int count = 0;
		for (int i = 0; i < _rows; i++){
			for (int j = 0; j < _cols; j++){
				if (mask(i, j)){
					yx2idx(i, j) = count;
					++count;
					_idx2yx.emplace_back(i, j);
				}
			}
		}

		_len = count;
		_idx2yx.shrink_to_fit();
	}

	~IndexTable()
	{
	};


	const int yx2idx(const int y, const int x) const
	{
		const auto pos = _yx2idx[y * _cols + x];
		if (pos == npos){
			throw "npos";
		}
		return pos;
	}

	const Point idx2yx(const int idx) const
	{
		return _idx2yx[idx];
	}

	const int operator()(const int y, const int x) const
	{
		return yx2idx(y, x);
	}

	const Point operator[](const int idx) const
	{
		return idx2yx(idx);
	}

	int& yx2idx(const int y, const int x)
	{
		return _yx2idx[y * _cols + x];
	}
	Point& idx2yx(const int idx)
	{
		return _idx2yx[idx];
	}

	//‰Šú‰»‚Í—\‚ß‚µ‚Ä‚¨‚­•K—v‚ª‚ ‚é
	template<class Vec, class Mat>
	void copyVec2Mat(const Vec& vec, Mat& mat) const
	{
		for (int k = 0; k < _len; k++)
		{
			mat(idx2yx(k).y, idx2yx(k).x) = vec[k];
		}
	}

	//‰Šú‰»‚Í—\‚ß‚µ‚Ä‚¨‚­•K—v‚ª‚ ‚é
	template<class Mat, class Vec>
	void copyMat2Vec(const Mat& mat, Vec& vec) const
	{
		for (int k = 0; k < _len; k++)
		{
			vec[k] = mat(idx2yx(k).y, idx2yx(k).x);
		}
	}

};

