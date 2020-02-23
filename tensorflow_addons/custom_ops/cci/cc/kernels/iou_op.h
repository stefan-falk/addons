#ifndef TENSORFLOW_ADDONS_CCI_KERNELS_POLYGON_IOU_OP_H_
#define TENSORFLOW_ADDONS_CCI_KERNELS_POLYGON_IOU_OP_H_

#define EIGEN_USE_THREADS

#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

template <typename T>
inline int sig(T d) {
  const T eps = 1E-8;
  return (d > eps) - (d < -eps);
}

template <typename T>
struct Point {
  T x, y;
  Point() {}
  Point(T x, T y) : x(x), y(y) {}
  bool operator==(const Point<T>& p) const {
    return sig(x - p.x) == 0 && sig(y - p.y) == 0;
  }
};

template <typename T>
T cross(Point<T> o, Point<T> a, Point<T> b) {  //叉积
  return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}
template <typename T>
T area(Point<T>* ps, int n) {
  ps[n] = ps[0];
  T res = 0;
  for (int i = 0; i < n; i++) {
    res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
  }
  return res / static_cast<T>(2.0);
}
template <typename T>
void lineCross(Point<T> a, Point<T> b, Point<T> c, Point<T> d, Point<T>& p) {
  T s1, s2;
  s1 = cross(a, b, c);
  s2 = cross(a, b, d);
  if (sig(s1) == 0 && sig(s2) == 0) return;
  if (sig(s2 - s1) == 0) return;
  p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
  p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
}
//多边形切割
//用直线ab切割多边形p，切割后的在向量(a,b)的左侧，并原地保存切割结果
//如果退化为一个点，也会返回去,此时n为1
template <typename T>
void polygon_cut(Point<T>* p, int& n, Point<T> a, Point<T> b, Point<T>* pp) {
  int m = 0;
  p[n] = p[0];
  for (int i = 0; i < n; i++) {
    if (sig(cross(a, b, p[i])) > 0) pp[m++] = p[i];
    if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1])))
      lineCross(a, b, p[i], p[i + 1], pp[m++]);
  }
  n = 0;
  for (int i = 0; i < m; i++)
    if (!i || !(pp[i] == pp[i - 1])) p[n++] = pp[i];
  while (n > 1 && p[n - 1] == p[0]) n--;
}
//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
template <typename T>
T intersectArea(Point<T> a, Point<T> b, Point<T> c, Point<T> d) {
  Point<T> o(0, 0);
  int s1 = sig(cross(o, a, b));
  int s2 = sig(cross(o, c, d));
  if (s1 == 0 || s2 == 0) return 0.0;  //退化，面积为0
  if (s1 == -1) swap(a, b);
  if (s2 == -1) swap(c, d);
  int n = 3;
  Point<T> p[n] = {o, a, b};
  Point<T> pp[2 * n];
  polygon_cut(p, n, o, c, pp);
  polygon_cut(p, n, c, d, pp);
  polygon_cut(p, n, d, o, pp);
  T res = fabs(area(p, n));
  if (s1 * s2 == -1) res = -res;
  return res;
}

//求两多边形的交面积
template <typename T>
T intersectArea(Point<T>* ps1, int n1, Point<T>* ps2, int n2) {
  if (area(ps1, n1) < 0) reverse(ps1, ps1 + n1);
  if (area(ps2, n2) < 0) reverse(ps2, ps2 + n2);
  ps1[n1] = ps1[0];
  ps2[n2] = ps2[0];
  T res = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
    }
  }
  return res;
}
#endif