#ifndef TENSOR_TRAIN
#define TENSOR_TRAIN

#include "tensor.h"

namespace yph
{
template <class T>
class TensorTrain
{
  private:
    Cube<T> tensor;
    Cube<T> cores[3];
    int ttRank[4];
    double delta;
    void calculateTTcores();

  public:
    TensorTrain() {}
    TensorTrain(const Cube<T> &t, const double &epsilon);
    TensorTrain(const TensorTrain<T> &t);
    TensorTrain(const TensorTrain<T> &t, double newDelta);
    TensorTrain(const cp_mats<T> &t);
    ~TensorTrain();

    const Cube<T> &getTensor();
    const int &getTTRank(int index);
    const double &getDelta();
    const Cube<T> &getCores(int index);

    void updateDelta(int newDelta);

    const int &numel();
    const int &size(int mode);

    friend TensorTrain<T> operator+(const TensorTrain<T> &t1, const TensorTrain<T> &t2);
    friend void multiContraction();
    friend void dotProduct();
};

template <class T>
TensorTrain<T>::TensorTrain(const Cube<T> &t, const double &epsilon)
{
    tensor = t;
    delta = epsilon * norm(t, "fro") / sqrt(modeNum - 1);
    calculateTTcores();
}

template <class T>
TensorTrain<T>::TensorTrain(const TensorTrain<T> &t)
{
    tensor = t.getTensor();
    for (int i = 0; i < 3; ++)
    {
        cores[i] = t.getCore(i);
    }
    for (int i = 0; i < 4; ++i)
    {
        ttRank[i] = t.getTTRank(i);
    }
    delta = t.getTTRank();
}

template <class T>
TensorTrain<T>::TensorTrain(const TensorTrain<T> &t, double newDelta)
{
    Mat<T> temp;
    for (int k = 2; k > 0; --1)
    {
    }
    qr()
}

template <class T>
TensorTrain<T>::TensorTrain(const cp_mats<T> &t)
{
}

template <class T>
TensorTrain<T>::~TensorTrain()
{
}

template <class T>
inline const Cube<T> &TensorTrain<T>::getTensor()
{
    return tensor;
}

template <class T>
inline const double &TensorTrain<T>::getDelta()
{
    return delta;
}

template <class T>
inline const Cube<T> &TensorTrain<T>::getCore(int index)
{
    return cores[index];
}
template <class T>
inline const int &TensorTrain<T>::getTTRank(int index)
{
    return ttRank[index];
}

template <class T>
inline const int &TensorTrain<T>::numel()
{
    return tensor.n_elem;
}

template <class T>
inline const int TensorTrain<T>::size(int mode)
{
    switch (mode)
    {
    case 1:
        return tensor.n_rows;
    case 2:
        return tensor.n_cols;
    case 3:
        return tensor.n_slices;
    }
}

template <class T>
void TensorTrain<T>::calculateTTcores()
{
    ttRank[0] = 1;
    ttRank[3] = 1;
    Cube<T> tempTensor = tensor;
    Mat<T> tempMat;
    for (int i = 1; i < 3; ++i)
    {
        int rowSize = ttRank[i - 1] * modeSize[i - 1];
        tempMat = ten2mat(tempTensor, rowSize, numel(tensor));
        Mat<T> U, S, V;
        int r;
        deltaSVD(tempMat, U, S, V, r, delta);
        ttRank[i] = r;
        mat2ten(U, cores[i - 1], ttRank[i - 1], modeSize[i - 1], ttRank[i]); // need to implement
        tempTensor = S * V.t();
    }
    cores[modeNum - 1] = tempTensor;
}

template <class T>
void TensorTrain<T>::updateDelta(int newDelta)
{
    *this = TensorTrain(*this, newDelta);
}

// need to testify
template <class T>
Mat<T> ten2mat(const Cube<T> &t, const int &row, const int &col)
{
    if (t.numel() != row * col)
    {
        cout << "参数设置错误与张量大小不匹配" << endl;
    }
    Mat<T> re(row, col);
    int pos;
    int m = t.n_rows, n = t.n_cols, s = t.n_slices;
    // addressing from lower dimension to higher
    for (int i = 0; i < t.n_rows; ++i)
    {
        for (int j = 0; j < t.n_cols; ++j)
        {
            for (int k = 0; k < t.n_slices; ++k)
            {
                pos = i + j * m + k * m * n;
                re[pos % row][floor(pos / row)] = t(i, j, k);
            }
        }
    }
    return re;
}

/*
template <class T>
void ten2mat(const Cube<T> &t, Mat<T> &re, const int &row, const int &col)
{
    int pos;
    int m = t.n_rows, n = t.n_cols, s = t.n_slices;
    re.resize(row,col);
    // addressing from lower dimension to higher
    for (int i = 0; i < t.n_rows; ++i)
    {
        for (int j = 0; j < t.n_cols; ++j)
        {
            for (int k = 0; k < t.n_slices; ++k)
            {
                pos = i + j * m + k * m * n;
                re[pos % row][floor(pos / col)] = t(i, j, k);
            }
        }
    }
}
*/

// need to testify
template <class T>
void mat2ten(const Mat<T> &mat, Cube<T> &re, const int &row, const int &col, const int &slice)
{
    if (mat.n_elem != row * col * slice)
    {
        cout << "参数设置错误与矩阵大小不匹配" << endl;
    }
    int pos;
    int m = mat.n_rows, n = mat.n_cols;
    re.resize(row, col, slice);
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            pos = i + j * m;
            re[pos % row][floor(pos / row) % col][floor(pos / (row * col))] = mat(i, j);
        }
    }
}
// -------------------------------------------------------------
template <class T>
void deltaSVD(Mat<T> mat, Mat<T> &U, Mat<T> &S, Mat<T> &V, int &r, const double &delta)
{
    Col<T> s = zeros<Col<T>>(min(mat.n_rows, mat.n_cols));
    svd(U, s, V, mat); // use V.t() to get trans
    truncation(s, r, delta);
    S = diagmat(s);
}

template <class T>
void truncation(Col<T> &s, int &r, const double &delta)
{
    T temp = 0;
    for (uword i = s.n_rows - 1; i >= 0; --i)
    {
        temp += s(i); // 利用的是norm(A) == norm(S)
        if (temp > delta)
        {
            for (uword j = s.n_rows - 1; j > i; --j)
            {
                s(j) = 0;
            }
            r = i + 1;
            break;
        }
    }
}
template <class T>
TensorTrain<T> operator+(const TensorTrain<T> &t1, const TensorTrain<T> &t2)
{
    TensorTrain<T> answer;
    answer.tensor.set_size(size(t1));
    answer.tensor = t1.tensor + t2.tensor;

    answer.cores[0].set_size(1, t1.size(1), t1.cores[0].n_slices + t2.cores[0].n_slices);
    answer.cores[1].set_size(t1.cores[1].n_rows + t2.cores[1].n_rows,
                             t1.cores[1].n_cols + t2.cores[1].n_cols,
                             t1.cores[1].n_slices + t2.cores[1].n_slices);
    answer.cores[2].set_size(t1.cores[2].n1 + t2.cores[2].n1, t1.size(3), 1);

    ////////////////////////////submat///////////////////////////

    answer.ttRank[0] = max(t1.ttRank[0], t2.ttRank[0]);
    answer.ttRank[3] = max(t1.ttRank[3], t2.ttRank[3]);
    answer.ttRank[1] = t1.ttRank[1] + t2.ttRank[1];
    answer.ttRank[2] = t1.ttRank[2] + t2.ttRank[2];
    answer.delta = min(t1.delta, t2.delta);
    return answer;
}

template <class T>
void dotProduct()
{
}

template <class T>
void multiContraction()
{
}

} // namespace yph

#endif
