#ifndef __GL_H__
#define __GL_H__
#include "tgaimage.h"
#include"geometry.h"
#include<cmath>

//����ռ��ӿڿռ�ת������
extern Matrix ModelView;
//��Ļ�ü�����
extern Matrix ViewPort;
//ͶӰ����
extern Matrix Projection;
//clip�����������ص��Ƿ�����Ļ�ϻ��ƣ�
void viewport(int x, int y, int w, int h);
//ͶӰ�任��������������׶�ڣ�
void projection(float coeff = 0.0f);
//��ģ��ת��Ϊ����������ת��Ϊ�������
void lookat(Vec3f eye, Vec3f center, Vec3f up);

struct IShader
{
	virtual ~IShader();
	//������ɫ��
	virtual Vec4f vertex(int iface, int nthvert) = 0;//���麯��
	/// <summary>
	/// Ƭ����ɫ��
	/// </summary>
	/// <param name="bar">���������ֵ����</param>
	/// <param name="color">��ɫ</param>
	/// <returns></returns>
	virtual bool fragment(Vec3f bar, TGAColor& color) = 0;

	//virtual bool fragment(Vec3f gl_FragCoord, Vec3f bar, TGAColor& color) = 0;
};

/// <summary>
/// �����λ��ƺ���
/// </summary>
/// <param name="pts">������������</param>
/// <param name="shader">ʹ�õ���ɫ��</param>
/// <param name="image">���Ƶ�ͼƬ</param>
/// <param name="zbuffer">���ͼ</param>
void DrawTriangle(Vec4f* pts, IShader& shader, TGAImage& image, float* zbuffer);

void triangle(mat<4, 3, float>& clipc, IShader& shader, TGAImage& image, float* zbuffer);
#endif  __GL_H__