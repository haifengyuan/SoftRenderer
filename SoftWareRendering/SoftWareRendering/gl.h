#ifndef __GL_H__
#define __GL_H__
#include "tgaimage.h"
#include"geometry.h"
#include<cmath>

//世界空间视口空间转换矩阵
extern Matrix ModelView;
//屏幕裁剪矩阵
extern Matrix ViewPort;
//投影矩阵
extern Matrix Projection;
//clip（决定该像素点是否在屏幕上绘制）
void viewport(int x, int y, int w, int h);
//投影变换（场景缩减至视锥内）
void projection(float coeff = 0.0f);
//将模型转化为世界坐标再转化为相机坐标
void lookat(Vec3f eye, Vec3f center, Vec3f up);

struct IShader
{
	virtual ~IShader();
	//顶点着色器
	virtual Vec4f vertex(int iface, int nthvert) = 0;//纯虚函数
	/// <summary>
	/// 片段着色器
	/// </summary>
	/// <param name="bar">重心坐标差值参数</param>
	/// <param name="color">颜色</param>
	/// <returns></returns>
	virtual bool fragment(Vec3f bar, TGAColor& color) = 0;

	//virtual bool fragment(Vec3f gl_FragCoord, Vec3f bar, TGAColor& color) = 0;
};

/// <summary>
/// 三角形绘制函数
/// </summary>
/// <param name="pts">三个顶点数据</param>
/// <param name="shader">使用的着色器</param>
/// <param name="image">绘制的图片</param>
/// <param name="zbuffer">深度图</param>
void DrawTriangle(Vec4f* pts, IShader& shader, TGAImage& image, float* zbuffer);

void triangle(mat<4, 3, float>& clipc, IShader& shader, TGAImage& image, float* zbuffer);
#endif  __GL_H__