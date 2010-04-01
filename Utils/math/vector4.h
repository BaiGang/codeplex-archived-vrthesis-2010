/********************************************************************
	created:	2005/06/02
	created:	2:6:2005   17:47
	filename: 	f:\code\VREngine\VREngine\Math\vector4.h
	file path:	f:\code\VREngine\VREngine\Math
	file base:	vector4
	file ext:	h
	author:		HuYong
	
	purpose:	
*********************************************************************/

#ifndef _VECTOR4_H
#define _VECTOR4_H

#include "geomath.h"
#include <assert.h>
class Matrix4;

class Vector4
{
public:
	float x,y,z,w;

	Vector4();     //空构造函数

	Vector4(float x0, float y0, float z0, float w0 = 1.0f); //构造函数，初始化可接受3个或者4个参数

	void setVec(float x0, float y0, float z0, float w0 = 1.0f);//设置向量值

	void nullVec(void); //设置向量值为(0,0,0,1)

	

	float lengthVec(void); // 返回向量大小

	void negateVec(void); //向量取反

	void normaVec(void); //单位化向量

	void crossVec(const Vector4& v1, const Vector4& v2); //计算两个向量叉积

	float dotVec(const Vector4& v); //点积

	//重载常用运算符
	float& operator[](int i); //返回第i个值

	Vector4& operator = (const Vector4& v);

	bool operator == (const Vector4& v);

	bool operator != (const Vector4& v);

	void operator += (const Vector4& v); 

	void operator -= (const Vector4& v);

	//	void operator /= (Vector4& v);

	void operator /= (float scal);

	//	void operator *= (Vector4& v); 

	void operator *= (float scal);

	Vector4 operator + (const Vector4& v);

	Vector4 operator - (const Vector4& v);

	//	Vector4 operator * (Vector4& v);  

	Vector4 operator * (float scal);

	//	Vector4 operator / (Vector4& v);

	Vector4 operator / (float scal);

	Vector4 operator * (const Matrix4& m); //mat4×4 * vector4

};

#endif