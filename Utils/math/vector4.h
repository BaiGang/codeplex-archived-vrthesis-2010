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

	Vector4();     //�չ��캯��

	Vector4(float x0, float y0, float z0, float w0 = 1.0f); //���캯������ʼ���ɽ���3������4������

	void setVec(float x0, float y0, float z0, float w0 = 1.0f);//��������ֵ

	void nullVec(void); //��������ֵΪ(0,0,0,1)

	

	float lengthVec(void); // ����������С

	void negateVec(void); //����ȡ��

	void normaVec(void); //��λ������

	void crossVec(const Vector4& v1, const Vector4& v2); //���������������

	float dotVec(const Vector4& v); //���

	//���س��������
	float& operator[](int i); //���ص�i��ֵ

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

	Vector4 operator * (const Matrix4& m); //mat4��4 * vector4

};

#endif