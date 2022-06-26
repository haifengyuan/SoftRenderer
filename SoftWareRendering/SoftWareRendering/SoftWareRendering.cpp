#include "tgaimage.h"
#include<cmath>
#include"model.h"
#include"geometry.h"
#include<vector>
#include <iostream>
#include"gl.h"
using namespace std;

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor blue = TGAColor(0, 0, 255, 255);
const TGAColor red = TGAColor(255,0,0,255);

Model* model = NULL;
float* shadowbuffer = NULL;

const int width = 800;
const int height = 800;
bool IsUsedBaryCenteric = false;//重心坐标插值
//float* ZBuffer = new float[width* height];

Vec3f light_dir(1, 1, 1);
//Vec3f eye(1, 1, 3);
Vec3f eye(1.2, -0.8, 3);
Vec3f center(0, 0, 0);
Vec3f up(0, 1, 0);

//阴影先渲染相机处的深度图
struct DepthShader:public IShader
{
	mat<3, 3, float> varying_tri;

	DepthShader() :varying_tri() {}

	virtual Vec4f vertex(int iface, int nthvert)
	{
		Vec4f gl_Vertex =embed<4>(model->vert(iface, nthvert));
		gl_Vertex = ViewPort*Projection*ModelView * gl_Vertex;
		varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
		return gl_Vertex;
	}

	virtual bool fragment(Vec3f bar, TGAColor& color)
	{
		Vec3f p = varying_tri * bar;
		color = TGAColor(255, 255, 255) * (p.z / 255.f);
		return false;
	}

};

struct Shader:public IShader
{
	mat<4, 4, float> uniform_MVP;//Projection*ModelView
	mat<4, 4, float> uniform_MVPIT;//(Projection*ModelView)^invert_transpose
	mat<4, 4, float> uniform_Mshadow;//transform framebuffer screen coordinates to shadowbuffer screen coordinates（面元屏幕空间到阴影屏幕空间变换矩阵）

	mat<3, 3, float> varying_tri;//裁剪空三角形坐标，顶点shader写入，fragementshader读取
	mat<2, 3, float> varying_uv;//uv 贴图，两行(uv坐标)三列(对应三个点)

	Shader(Matrix M, Matrix MVPIT, Matrix MS) :uniform_MVP(M), uniform_MVPIT(MVPIT), uniform_Mshadow(MS), varying_uv(), varying_tri() {}

	virtual Vec4f vertex(int iface, int nthvert)
	{
		varying_uv.set_col(nthvert, model->uv(iface, nthvert));//记录.obj文件中的三个点的uv坐标

		Vec4f gl_Vertex =embed<4>(model->vert(iface, nthvert)) ;
		gl_Vertex = ViewPort*Projection*ModelView * gl_Vertex;//模型空间转屏幕空间
		varying_tri.set_col(nthvert, proj<3>(gl_Vertex/gl_Vertex[3]));
		return gl_Vertex;
	};

	virtual bool fragment(Vec3f bar,TGAColor &color)
	{
		Vec4f sb_p = uniform_Mshadow * embed<4>(varying_tri*bar);//得到framebuffer中插值的像素点坐标，然后将其转换到shadowbuffer空间中
		sb_p = sb_p / sb_p[3];
		int idx = int(sb_p[0]) + int(sb_p[1]) * width;//找到该像素点再shadowbuffer中的index
		//float shadow = 0.3f + 0.7 * (shadowbuffer[idx] < sb_p[2]);//将当前的点的深度与shadowbuffer中的记录的深度最比较以决定该点在不在阴影中
		float shadow = 0.3f + 0.7 * (shadowbuffer[idx] < sb_p[2]+43.34);//magic coff to avoid z-fighting
		Vec2f uv = varying_uv * bar;//插值得到要绘制点的UV坐标
		Vec3f n = proj<3>(uniform_MVPIT*embed<4>(model->normal(uv))).normalize();//通过UV坐标找到法线贴图中记录的法线值，并且将该法线变换到投影空间中，并且归一化
		Vec3f l = proj<3>(uniform_MVP * embed<4>(light_dir)).normalize();//光线转化到投影空间中并且归一化
		Vec3f r = (n * (n * l * 2) - l).normalize();//反射光线的方向
		float spec = pow(std::max(r.z, 0.f), model->specular(uv));//寻找高光值
		float diff = std::max(0.f,n*l);//漫反射值
		TGAColor c = model->diffuse(uv);//主贴图颜色
		for (int i = 0; i < 3; i++)
		{
			color[i] = std::min<float>(20+c[i]*shadow*(1.2*diff+0.6*spec), 255);
			//color[i] += 20;//只有环境光
		}
		return false;
	}
};

struct GouraudShader:public IShader
{
	Vec3f varying_intensity;//diffuse lighting intensity for fragement shader to read；
	mat<2, 3, float> varying_uv;//uv 贴图，两行(uv坐标)三列(对应三个点)
	mat<4, 4, float> uniform_MVP;//Projection*ModelView
	mat<4, 4, float> uniform_MVPT;//(Projection*ModelView)^transpose

	mat<4, 3, float> varying_tri;//裁剪空三角形坐标，顶点shader写入，fragementshader读取
	mat<3, 3, float> varying_nrm;//每个顶点的法线
	mat<3, 3, float> ndc_tri;//三角形标准坐标（normalized device coordinates）

	virtual Vec4f vertex(int iface,int nthvert)
	{
		//1.顶点
		Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));//第iface个三角形的第nthvert个顶点坐标
		gl_Vertex =  Projection * ModelView * gl_Vertex;//MVP变换，将自身坐标下转为裁剪空间
		varying_tri.set_col(nthvert, gl_Vertex);
		ndc_tri.set_col(nthvert, proj<3>(gl_Vertex/gl_Vertex[3]));

		gl_Vertex = ViewPort* gl_Vertex;
		//2.法线
		//float difffuse=model->normal(iface, nthvert)* light_dir;//iface个三角形的第nthvert个顶点法线
		//varying_intensity[nthvert] = std::max(0.f, difffuse);
		varying_nrm.set_col(nthvert,proj<3>(( Projection * ModelView).invert_transpose()*embed<4>(model->normal(iface,nthvert),0.f)));

		//3.UV
		varying_uv.set_col(nthvert,model->uv(iface, nthvert));//


		
		return gl_Vertex;
	}

	virtual bool fragment(Vec3f bar, TGAColor& color)
	{
		//float intensity = varying_intensity * bar;//差值得到改点颜色强度大小
		//Vec2f uv = varying_uv * bar;//差值得到UV坐标
		//Vec3f n=proj<3>(uniform_MVPT*embed<4>(model->normal(uv))).normalize();//法线的MVP变换，自身空间转化到投影空间
		//Vec3f l = proj<3>(uniform_MVP * embed<4>(light_dir)).normalize();//入射光的MVP变换，自身空间转化到投影空间
		//Vec3f r = (n * (n * l) * 2.f - l).normalize();//根据法线方向光线防线计算反射方向
		//float spec = pow(std::max(r.z, 0.0f), model->specular(uv));//高光贴图  ??为什么只考虑反射光线的z方向？？？
		//float diff = std::max(0.f, n * l);
		//TGAColor c = model->diffuse(uv);
		//color = c;
		//color = model->diffuse(uv) * intensity*10;
		//float ambient = 5;
		//for (int i = 0; i < 3; i++)
		//{
		//	color[i] = std::min<float>(ambient+color[i]*(diff+spec), 255);
		//}

		Vec3f bn = (varying_nrm * bar).normalize();//法线插值得到该像素点的法线
		Vec2f uv = varying_uv * bar;//差值得到UV坐标


		//求切向空间变换矩阵
		mat<3, 3, float> A;
		A[0] = ndc_tri.col(1) - ndc_tri.col(0);
		A[1] = ndc_tri.col(2) - ndc_tri.col(0);
		A[2] = bn;
		A = A.invert();

		Vec3f i = A * Vec3f(varying_uv[0][1]- varying_uv[0][0], varying_uv[0][2]- varying_uv[0][0],0);
		Vec3f j = A * Vec3f(varying_uv[1][1] - varying_uv[1][0], varying_uv[1][2] - varying_uv[1][0], 0);

		mat<3, 3, float> B;
		B.set_col(0,i.normalize());
		B.set_col(1, j.normalize());
		B.set_col(2, bn);

		Vec3f n = (B * model->normal(uv)).normalize();//通过uv找到法线再变换再从切向空间转化到世界空间中
		light_dir = light_dir.normalize();
		//float diff= std::max(n*light_dir, 0.f);
		float diff = std::max(bn * light_dir, 0.f);
		color = model->diffuse(uv)* diff;




		return false;
	}

	//sample NPR
	//virtual bool fragment(Vec3f bar, TGAColor& color)
	//{
	//	float intensity = varying_intensity * bar;
	//	if (intensity > .85) intensity = 1;
	//	else if (intensity > .60) intensity = .80;
	//	else if (intensity > .45) intensity = .60;
	//	else if (intensity > .30) intensity = .45;
	//	else if (intensity > .15) intensity = .30;
	//	else intensity = 0;
	//	color = TGAColor(255, 0, 0) * intensity;
	//	return false;
	//}
};

struct ZShader:public IShader
{
	mat<4, 3, float> varying_tri;

	virtual Vec4f vertex(int iface, int nthvert)
	{
		Vec4f gl_Vertex =  ViewPort*Projection * ModelView * embed<4>(model->vert(iface, nthvert));
		varying_tri.set_col(nthvert, gl_Vertex);
		return gl_Vertex;
	}

	//virtual bool fragment(Vec3f gl_FragCoord,Vec3f bar, TGAColor& color)
	//{
	//	color = TGAColor(0, 0, 0);
	//	return false;
	//}

	virtual bool fragment(Vec3f bar, TGAColor& color)
	{
		color = TGAColor(0, 0, 0);
		return false;
	}
};

//
//void line(int x0, int y0, int x1, int y1, TGAImage& image, TGAColor color) {
//	/*for (float t = 0.; t < 1.; t += .1) {
//		int x = x0 + (x1 - x0) * t;
//		int y = y0 + (y1 - y0) * t;
//		image.set(x, y, color);
//	}*/
//
//	/*for (int x = x0; x < x1; x++)
//	{
//		float t = (x - x0) /(float) (x1 - x0);
//		int y = (1 - t) * y0 + t * y1;
//		image.set(x, y, color);
//	}*/
//
//	bool sweep = false;
//	if (std::abs(x1-x0)<std::abs(y1-y0))
//	{
//		std::swap(x0, y0);
//		std::swap(x1,y1);
//		sweep = true;
//	}
//	if (x0 > x1)
//	{
//		std::swap(x0, x1);
//		std::swap(y0, y1);
//	}
//	int dx = x1 - x0;
//	int dy = y1 - y0;
//	int derror2 = std::abs(dy) * 2;
//	int error2 = 0;
//	int y = y0;
//	for (int x = x0; x < x1; x++)
//	{
//		if (sweep)
//		{
//			image.set(y, x, color);
//		}
//		else
//		{
//			image.set(x,y, color);
//		}
//		error2 += derror2;
//		if (error2 > dx)
//		{
//			y += (y1>y0?1:-1);
//			error2 -= dx * 2;
//		}
//	}
//
//	/*bool steep = false;
//	if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
//		std::swap(x0, y0);
//		std::swap(x1, y1);
//		steep = true;
//	}
//	if (x0 > x1) {
//		std::swap(x0, x1);
//		std::swap(y0, y1);
//	}
//
//	for (int x = x0; x <= x1; x++) {
//		float t = (x - x0) / (float)(x1 - x0);
//		int y = y0 * (1. - t) + y1 * t;
//		if (steep) {
//			image.set(y, x, color);
//		}
//		else {
//			image.set(x, y, color);
//		}
//	}*/
//
//}
//
//int TwoDimensionCrossProduct(Vec2i a, Vec2i b)
//{
//	return (a.x * b.y - b.x * a.y);
//}
//
//Vec3f BaryCenteric(Vec3f* TrianglePoints, Vec3f DecetePoint)
//{
//	Vec3f AB = TrianglePoints[0] - TrianglePoints[1];
//	Vec3f BC = TrianglePoints[1] - TrianglePoints[2];
//	Vec3f CA = TrianglePoints[2] - TrianglePoints[0];
//
//	Vec3f AP = TrianglePoints[0] - DecetePoint;
//	Vec3f BP = TrianglePoints[1] - DecetePoint;
//	Vec3f CP = TrianglePoints[2] - DecetePoint;
//
//	Vec3f PA = DecetePoint - TrianglePoints[0];
//	Vec3f AC = TrianglePoints[0] - TrianglePoints[2];
//
//	Vec3i x = Vec3i(AB.x, AC.x, PA.x);
//	Vec3i y = Vec3i(AB.y, AC.y, PA.y);
//
//	Vec3i BaryCentericParas = x ^ y;
//	if (std::abs(BaryCentericParas.z) < 1)return Vec3f(-1,1,1);
//
//	
//	return Vec3f(1.f- (BaryCentericParas.x+ BaryCentericParas.y)/(float) BaryCentericParas.z, BaryCentericParas.x/ (float)BaryCentericParas.z, BaryCentericParas.y/ (float) BaryCentericParas.z);
//
//}
//
//bool PonitIsInTriangle(Vec2i* TrianglePoints, Vec2i DecetePoint)
//{
//	Vec2i AB = TrianglePoints[0] - TrianglePoints[1];
//	Vec2i BC = TrianglePoints[1] - TrianglePoints[2];
//	Vec2i CA = TrianglePoints[2] - TrianglePoints[0];
//
//	Vec2i AP = TrianglePoints[0] - DecetePoint;
//	Vec2i BP = TrianglePoints[1] - DecetePoint;
//	Vec2i CP = TrianglePoints[2] - DecetePoint;
//
//	Vec2i PA = DecetePoint - TrianglePoints[0];
//	Vec2i AC = TrianglePoints[0] - TrianglePoints[2];
//
//
//
//	if (IsUsedBaryCenteric)
//	{
//		//1.BaryCenteric Method 
//		//BaryCenteric formulation: P=(1-u-v)A+uB+vC to calculate out vector[u,v,1]
//		//[AB, AC,PA]x  [AB,AC,PA]y cross product [u,v,1] equal zero thus 
//		//[u,v,1]= Cross() [AB, AC,PA]x  [AB,AC,PA]y
//		Vec3i x = Vec3i(AB.x, AC.x, PA.x);
//		Vec3i y = Vec3i(AB.y, AC.y, PA.y);
//
//		Vec3i BaryCentericParas = x ^ y;
//		if (std::abs(BaryCentericParas.z) < 1)return false;
//
//		BaryCentericParas = Vec3i(1.f-(BaryCentericParas .y+BaryCentericParas.x) /(float) BaryCentericParas.z, BaryCentericParas.x / (float)BaryCentericParas.z, BaryCentericParas.y / (float)BaryCentericParas.z);
//		
//		if (BaryCentericParas.x < 0 || BaryCentericParas.y < 0 || BaryCentericParas.z<0)
//		{
//			return false;
//		}
//		else
//		{
//			return true;
//		}
//	}
//	else 
//	{
//		//2.CrossProduct Method
//		Vec3i result = Vec3i(TwoDimensionCrossProduct(AB, AP), TwoDimensionCrossProduct(BC, BP), TwoDimensionCrossProduct(CA, CP));
//		if (result.x < 0 && result.y < 0 && result.z < 0)
//		{
//			return true;
//		}
//		if (result.x > 0 && result.y > 0 && result.z > 0)
//		{
//			return true;
//		}
//		return false;
//	}
//	
//}
//
//void DrawTriangle(Vec3f* TrianglePoints,float* ZBuffer,TGAImage& image, TGAColor color)
//{
//	//1.做方形包围盒缩小判断面积
//	Vec2f bboxmin = Vec2f(image.get_width() - 1, image.get_height() - 1);
//	Vec2f bboxmax = Vec2f(0, 0);
//	Vec2f imagemin= Vec2f(0, 0);
//	Vec2f imagemax = Vec2f(image.get_width() - 1, image.get_height() - 1);
//
//	for (int i = 0; i < 3; i++)
//	{
//		bboxmin.x =std::max(imagemin.x, std::min(bboxmin.x, TrianglePoints[i].x)) ;
//		bboxmin.y = std::max(imagemin.y, std::min(bboxmin.y, TrianglePoints[i].y));
//
//		bboxmax.x =std::min(std::max(bboxmax.x, TrianglePoints[i].x),imagemax.x) ;
//		bboxmax.y =std::min(std::max(bboxmax.y, TrianglePoints[i].y),imagemax.y);
//	}
//
//	//2.判断该像素是否在三角形内部绘该像素并且更新深度
//	Vec3f PointTemp=Vec3f(0,0,0);
//	for (PointTemp.x  = bboxmin.x; PointTemp.x < bboxmax.x; PointTemp.x++)
//	{
//		for (PointTemp.y = bboxmin.y; PointTemp.y < bboxmax.y; PointTemp.y++)
//		{
//			//if (PonitIsInTriangle(TrianglePoints, PointTemp))image.set(PointTemp.x, PointTemp.y, color);
//			Vec3f bc_screen = BaryCenteric(TrianglePoints, PointTemp);
//			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0)continue;
//
//			PointTemp.z = TrianglePoints[0].z * bc_screen.x+ TrianglePoints[1].z * bc_screen.y+ TrianglePoints[2].z * bc_screen.z;
//			if (PointTemp.z > ZBuffer[(int)(PointTemp.x + width * PointTemp.y)])
//			{
//				ZBuffer[(int)(PointTemp.x + width * PointTemp.y)] = PointTemp.z;
//				image.set(PointTemp.x, PointTemp.y, color);
//			}
//		}
//	}
//
//}
//
//Vec3f world2screen(Vec3f v)
//{
//	return Vec3f(int((v.x + 1.) * width / 2. + .5), int((v.y + 1.) * height / 2. + .5), v.z);
//}

float max_elevation_angle(float* zbuffer,Vec2f p,Vec2f dir)
{
	float	maxangle = 0;
	for (float t = 0.; t < 10000.; t+=1.)
	{
		
		Vec2f cur = p + dir * t;//屏幕空间固定方向采样点
		if (cur.x >= width || cur.y >= height || cur.x < 0 || cur.y < 0)return maxangle;//超出屏幕外则不在继续在该方向采样直接放回值

		float distance = (p - cur).norm();//屏幕空间像素点与采样点之间距离
		if (distance < 1.f)continue;//
		float elevation = zbuffer[int(cur.x)+int(cur.y)*width] - zbuffer[int(p.x) + int(p.y) * width];//像素点与采样点之间的深度差
		maxangle = std::max<float>(maxangle, atanf(elevation / distance));
	}
	return maxangle;
}

int main(int argc, char** argv) {
	//TGAImage image(100, 100, TGAImage::RGB);
	//line(13, 20, 80, 40, image, white);
	//line(20, 13, 40, 80, image, red);
	//line(80, 40, 13, 20, image, red);
	//// image.set(52, 41, red);
	//image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
	//image.write_tga_file("output.tga");

	//init ZBuffer
	//for (int i = 0; i < width*height; i++)
	//{
	//	ZBuffer[i] = -std::numeric_limits<float>::max();
	//}

	//if (argc == 2)
	//{
	//	model = new Model(argv[1]);
	//}
	//else
	//{
	//	model = new Model("obj/african_head.obj");
	//}

	//Vec3f LightDirection = Vec3f(0,0,-1);

	//TGAImage image(width, height, TGAImage::RGB);
	//for (int i = 0; i < model->nfaces(); i++) {
	//	std::vector<int> face = model->face(i);
	//	Vec2i screen_coords[3];
	//	Vec3f world_coords[3];
	//	Vec3f TrianglePoints[3];

	//	for (int j = 0; j < 3; j++) {
	//		TrianglePoints[j] = world2screen(model->vert(face[j]));
	//		Vec3f v = model->vert(face[j]);
	//		screen_coords[j] = Vec2i((v.x + 1.) * width / 2., (v.y + 1.) * height / 2.);
	//		world_coords[j] = v;
	//	}

	//	Vec3f TriangleNormal = (world_coords[2] - world_coords[0]) ^ (world_coords[1] - world_coords[0]);
	//	TriangleNormal.normalize();
	//	float Intensity = LightDirection * TriangleNormal;
	//	if(Intensity>0)DrawTriangle(TrianglePoints, ZBuffer, image, TGAColor(Intensity * 255, Intensity * 255, Intensity * 255, 255));
	//	//DrawTriangle (TrianglePoints, ZBuffer, image, TGAColor(rand() % 255, rand() % 255, rand() % 255, 255));
	//	}

	//image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
	//image.write_tga_file("manface1.tga");


	//TGAImage Triangle1(200,200,TGAImage::RGB);
	//Vec2i TrianglePoints1[3] = {Vec2i(10,10),Vec2i(100,30),Vec2i(190,160)};
	//DrawTriangle(TrianglePoints1, Triangle1, blue);
	//Triangle1.flip_vertically(); // i want to have the origin at the left bottom corner of the image
	//Triangle1.write_tga_file("TriangleBaryCenteric.tga");



	//TGAImage Triangle2(200, 200, TGAImage::RGB);
	//Vec2i TrianglePoints2[3] = { Vec2i(10,10),Vec2i(100,30),Vec2i(190,160) };
	//DrawTriangle(TrianglePoints2, Triangle2, blue);
	//Triangle2.flip_vertically(); // i want to have the origin at the left bottom corner of the image
	//Triangle2.write_tga_file("TriangleCrossProduct.tga");



	if (argc == 2)
	{
		model = new Model(argv[1]);
	}
	else
	{
		//model = new Model("obj/african_head.obj");
		model = new Model("obj/diablo3_pose.obj");
	}

	long float PI = 4*atanf(1.0f);

	//有阴影的渲染
	
	float* zbuffer = new float[width*height];
	shadowbuffer = new float[width * height];
	light_dir.normalize();

	for (int i = width * height; --i;)
	{
		zbuffer[i] = shadowbuffer[i] = -std::numeric_limits<float>::max();
	}

#pragma region 有阴影的渲染过程1.rendering the shadow buffer  2.rendering the frame buffer

	//{//rendering the shadow buffer
	//	TGAImage depth(width, height, TGAImage::RGB);
	//	lookat(light_dir, center, up);
	//	viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
	//	projection(0);




	//	DepthShader depthshader;
	//	Vec4f screen_coords[3];
	//	for (int i = 0; i < model->nfaces(); i++)
	//	{
	//		for (int j = 0; j < 3; j++)
	//		{
	//			screen_coords[j] = depthshader.vertex(i,j);
	//		}
	//		DrawTriangle(screen_coords, depthshader, depth, shadowbuffer);
	//	}
	//	depth.flip_vertically();
	//	depth.write_tga_file("depth0508.tga");
	//}


	//Matrix M = ViewPort * Projection * ModelView;//模型空间到屏幕空间变换矩阵
	//{//rendering the frame buffer

	//	lookat(eye, center, up);
	//	viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
	//	projection(-1.f / (eye - center).norm());//投影矩阵的缩放程度与视角位置有关
	//	light_dir.normalize();

	//	TGAImage frame(width, height, TGAImage::RGB);
	//	

	//	Shader shader(Projection*ModelView,(Projection * ModelView).invert_transpose(),M*(ViewPort * Projection * ModelView).invert());

	//	shader.uniform_MVP = Projection * ModelView;
	//	shader.uniform_MVPIT = (Projection * ModelView).invert_transpose();

	//	
	//	//遍历每个三角形
	//	for (int i = 0; i < model->nfaces(); i++) {
	//		Vec4f screen_coords[3];//三角形的三个屏幕坐标
	//		//遍历三个顶点
	//		for (int j = 0; j < 3; j++)
	//		{
	//			screen_coords[j] = shader.vertex(i, j);
	//		}
	//		DrawTriangle(screen_coords, shader, frame, zbuffer);
	//	}
	//	frame.flip_vertically();
	//	frame.write_tga_file("frame0508.tga");

	//	/*Vec4f screen_coords[3];
	//	screen_coords[0] = embed<4>(Vec3f(10.f, 10.f, 1.f));
	//	screen_coords[1] = embed<4>(Vec3f(100.f, 30.f, 1.f));
	//	screen_coords[2] = embed<4>(Vec3f(190.f, 160.f, 1.f));

	//	DrawTriangle(screen_coords, shader, image, zbuffer);*/

	//
	//}



#pragma endregion

#pragma region SSAO
	TGAImage frame(width, height, TGAImage::RGB);
	lookat(eye,center,up);
	viewport(width/8,height/8,width*3/4, height * 3 / 4);
	projection(-1.f / (eye - center).norm());

	ZShader zshader;
	for (int i = 0; i < model->nfaces(); i++)
	{
		Vec4f screen_coords[3];
		for (int j = 0; j < 3; j++)
		{
			screen_coords[j]=zshader.vertex(i, j);
		}
			DrawTriangle(screen_coords,zshader,frame,zbuffer);
			//triangle(zshader.varying_tri,zshader,frame,zbuffer);
	}

	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			if (zbuffer[x + y * width] < -1e5)continue;
			float total = 0;
			for (float a = 0; a < PI*2-1e-4; a+=PI/4)//球面采样
			{
				total += PI / 2 - max_elevation_angle(zbuffer,Vec2f(x,y),Vec2f(cos(a),sin(a)));
			}
			total /= (PI / 2) * 8;
			total = pow(total,100.f);
			frame.set(x, y, TGAColor(255 * total, 255 * total, 255 * total));
		}
	}
	frame.flip_vertically();
	frame.write_tga_file("framebuffer0508.tga");


#pragma endregion

	delete model;
	delete[] zbuffer;
	delete[] shadowbuffer;
	return 0;
}