#include "gl.h"
#include <limits>
#include<cstdlib>
#include<cmath>
#include"gl.h"

Matrix ModelView;
Matrix Projection;
Matrix ViewPort;


IShader::~IShader() {}

void lookat(Vec3f eye, Vec3f center, Vec3f up)
{
	Vec3f z = (eye - center).normalize();
	Vec3f x = cross(up, z).normalize();
	Vec3f y = cross(z, x).normalize();
	ModelView = Matrix::identity();
	for (int i = 0; i < 3; i++)
	{
		ModelView[0][i] = x[i];
		ModelView[1][i] = y[i];
		ModelView[2][i] = z[i];
		ModelView[i][3] = -center[i];
	}
}

void projection(float coeff)
{
	Projection = Matrix::identity();
	Projection[3][2] = coeff;
}

void viewport(int x, int y, int w, int h)
{
	ViewPort = Matrix::identity();
	ViewPort[0][3] = x + w / 2.f;
	ViewPort[1][3] = y + h / 2.f;
	ViewPort[2][3] = 255.f / 2.f;
	ViewPort[0][0] = w / 2.f;
	ViewPort[1][1] = h / 2.f;
	ViewPort[2][2] = 255.f / 2.f;
} 

Vec3f BaryCenteric(Vec3f* TrianglePoints, Vec3f DecetePoint)
{
	/*Vec3f A = TrianglePoints[0];
	Vec3f B = TrianglePoints[1];
	Vec3f C = TrianglePoints[2];
	Vec3f P = DecetePoint;

	Vec3f AB = TrianglePoints[0] - TrianglePoints[1];
	Vec3f BC = TrianglePoints[1] - TrianglePoints[2];
	Vec3f CA = TrianglePoints[2] - TrianglePoints[0];

	Vec3f AP = TrianglePoints[0] - DecetePoint;
	Vec3f BP = TrianglePoints[1] - DecetePoint;
	Vec3f CP = TrianglePoints[2] - DecetePoint;

	Vec3f PA = DecetePoint - TrianglePoints[0];
	Vec3f AC = TrianglePoints[0] - TrianglePoints[2];

	Vec3i x = Vec3i(AB.x, AC.x, PA.x);
	Vec3i y = Vec3i(AB.y, AC.y, PA.y);*/

	Vec3i A = (Vec3i)TrianglePoints[0];
	Vec3i B = (Vec3i)TrianglePoints[1];
	Vec3i C = (Vec3i)TrianglePoints[2];
	Vec3i P = (Vec3i)DecetePoint;

	Vec3i AB = TrianglePoints[0] - TrianglePoints[1];
	Vec3i BC = TrianglePoints[1] - TrianglePoints[2];
	Vec3i CA = TrianglePoints[2] - TrianglePoints[0];

	Vec3i AP = TrianglePoints[0] - DecetePoint;
	Vec3i BP = TrianglePoints[1] - DecetePoint;
	Vec3i CP = TrianglePoints[2] - DecetePoint;

	Vec3i PA = DecetePoint - TrianglePoints[0];
	Vec3i AC = TrianglePoints[0] - TrianglePoints[2];

	Vec3i x = Vec3i(AB.x, AC.x, PA.x);
	Vec3i y = Vec3i(AB.y, AC.y, PA.y);

	//Vec3i BaryCentericParas = Vec3i(x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]);
	Vec3i BaryCentericParas = cross(x, y);
	if (std::abs(BaryCentericParas[2])>1e-2)
	{
		return Vec3f(1.f - (BaryCentericParas.x + BaryCentericParas.y) / (float)BaryCentericParas.z, BaryCentericParas.x / (float)BaryCentericParas.z, BaryCentericParas.y / (float)BaryCentericParas.z);
	}
	else 
	{
		return Vec3f(-1, 1, 1);
	}

	


	//Vec3f s[2];
	//for (int i = 2; i--; ) {
	//	s[i][0] = C[i] - A[i];
	//	s[i][1] = B[i] - A[i];
	//	s[i][2] = A[i] - P[i];
	//}
	//Vec3f u = cross(s[0], s[1]);
	//if (std::abs(u[2]) > 1e-2) // dont forget that u[2] is integer. If it is zero then triangle ABC is degenerate
	//	return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
	//return Vec3f(-1, 1, 1); // in this case generate negative coordinates, it will be thrown away by the rasterizator
}

Vec3f barycentric(Vec2f A, Vec2f B, Vec2f C, Vec2f P) {
	Vec3f s[2];
	for (int i = 2; i--; ) {
		s[i][0] = C[i] - A[i];
		s[i][1] = B[i] - A[i];
		s[i][2] = A[i] - P[i];
	}
	Vec3f u = cross(s[0], s[1]);
	if (std::abs(u[2]) > 1e-2) // dont forget that u[2] is integer. If it is zero then triangle ABC is degenerate
		return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
	return Vec3f(-1, 1, 1); // in this case generate negative coordinates, it will be thrown away by the rasterizator
}

void DrawTriangle(Vec4f* TrianglePoints,IShader & shader,TGAImage& image, float* zbuffer)
{
	//1.做方形包围盒缩小判断面积
	/*Vec2f bboxmin = Vec2f(image.get_width() - 1, image.get_height() - 1);
	Vec2f bboxmax = Vec2f(0, 0);
	Vec2f imagemin= Vec2f(0, 0);
	Vec2f imagemax = Vec2f(image.get_width() - 1, image.get_height() - 1);*/

	Vec2f bboxmin=Vec2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

	Vec2f bboxmax = Vec2f(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());;

	for (int i = 0; i < 3; i++)
	{
		bboxmin.x = std::min(bboxmin.x, TrianglePoints[i][0]/ TrianglePoints[i][3]) ;
		bboxmin.y =  std::min(bboxmin.y, TrianglePoints[i][1]/ TrianglePoints[i][3]);

		bboxmax.x =std::max(bboxmax.x, TrianglePoints[i][0]/ TrianglePoints[i][3]) ;
		bboxmax.y =std::max(bboxmax.y, TrianglePoints[i][1]/ TrianglePoints[i][3]);
	}



	//2.判断该像素是否在三角形内部绘该像素并且更新深度
	Vec2i PointTemp;
	TGAColor ColorTemp;
	for (PointTemp.x  = bboxmin.x; PointTemp.x <= bboxmax.x; PointTemp.x++)
	{
		for (PointTemp.y = bboxmin.y; PointTemp.y <= bboxmax.y; PointTemp.y++)
		{
			//四维提取前三维
			/*Vec3f TrianglePointsTemp[3];
			TrianglePointsTemp[0] = proj<3>(TrianglePoints[0]/ TrianglePoints[0][3]);
			TrianglePointsTemp[1] = proj<3>(TrianglePoints[1]/ TrianglePoints[1][3]);
			TrianglePointsTemp[2] = proj<3>(TrianglePoints[2]/ TrianglePoints[2][3]);*/

			Vec2f TrianglePointsTemp[3];
			TrianglePointsTemp[0] = proj<2>(TrianglePoints[0] / TrianglePoints[0][3]);
			TrianglePointsTemp[1] = proj<2>(TrianglePoints[1] / TrianglePoints[1][3]);
			TrianglePointsTemp[2] = proj<2>(TrianglePoints[2] / TrianglePoints[2][3]);

			//三角形重心坐标的三个权重
			//Vec3f  InterpolationParam= BaryCenteric(TrianglePointsTemp, (PointTemp));
			Vec3f InterpolationParam = barycentric(TrianglePointsTemp[0], TrianglePointsTemp[1], TrianglePointsTemp[2], PointTemp);

			//通过权重插值得到z与w值
			float z = TrianglePoints[0][2] * InterpolationParam[0] + TrianglePoints[1][2] * InterpolationParam[1] + TrianglePoints[2][2] * InterpolationParam[2];
			float w = TrianglePoints[0][3] * InterpolationParam[0] + TrianglePoints[1][3] * InterpolationParam[1] + TrianglePoints[2][3] * InterpolationParam[2];
			int frag_depth = std::max(0,std::min(255,int(z/w+0.5)));
			//if (InterpolationParam.x < 0 || InterpolationParam.y < 0 || InterpolationParam.z < 0|| zbuffer.get(PointTemp.x, PointTemp.y)[0]>frag_depth)continue;
			if (InterpolationParam.x < 0 || InterpolationParam.y < 0 || InterpolationParam.z < 0 || zbuffer[PointTemp.x+ PointTemp.y*image.get_width()]>frag_depth)continue;
			////PointTemp.z = TrianglePoints[0].z * bc_screen.x+ TrianglePoints[1].z * bc_screen.y+ TrianglePoints[2].z * bc_screen.z;
			//if (PointTemp.z > ZBuffer[(int)(PointTemp.x + width * PointTemp.y)])
			//{
			//	ZBuffer[(int)(PointTemp.x + width * PointTemp.y)] = PointTemp.z;
			//	image.set(PointTemp.x, PointTemp.y, color);
			//}

			bool discard = shader.fragment(InterpolationParam, ColorTemp);
			if (!discard)
			{
				zbuffer[PointTemp.x + PointTemp.y * image.get_width()] = frag_depth;
				image.set(PointTemp.x, PointTemp.y, ColorTemp);
			}
		}
	}

}


//void triangle(mat<4, 3, float>& clipc, IShader& shader, TGAImage& image, float* zbuffer) {
//	mat<3, 4, float> pts = (ViewPort * clipc).transpose(); // transposed to ease access to each of the points
//	mat<3, 2, float> pts2;
//	for (int i = 0; i < 3; i++) pts2[i] = proj<2>(pts[i] / pts[i][3]);
//
//	Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
//	Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
//	Vec2f clamp(image.get_width() - 1, image.get_height() - 1);
//	for (int i = 0; i < 3; i++) {
//		for (int j = 0; j < 2; j++) {
//			bboxmin[j] = std::max(0.f, std::min(bboxmin[j], pts2[i][j]));
//			bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], pts2[i][j]));
//		}
//	}
//	Vec2i P;
//	TGAColor color;
//	for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++) {
//		for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++) {
//			Vec3f bc_screen = barycentric(pts2[0], pts2[1], pts2[2], P);
//			Vec3f bc_clip = Vec3f(bc_screen.x / pts[0][3], bc_screen.y / pts[1][3], bc_screen.z / pts[2][3]);
//			bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);
//			float frag_depth = clipc[2] * bc_clip;
//			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z<0 || zbuffer[P.x + P.y * image.get_width()]>frag_depth) continue;
//			bool discard = shader.fragment(Vec3f(P.x, P.y, frag_depth), bc_clip, color);
//			if (!discard) {
//				zbuffer[P.x + P.y * image.get_width()] = frag_depth;
//				image.set(P.x, P.y, color);
//			}
//		}
//	}
//}