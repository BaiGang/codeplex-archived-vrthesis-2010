#ifndef _MATRIX4_H
#define _MATRIX4_H
#include "geomath.h"
#include <assert.h>


class Matrix4
{
private:
	float m_data[16];
public:
	Matrix4();
	Matrix4(float mat[16]);

	Matrix4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33);
	
	Matrix4(const Matrix4 &other);//�������캯��
	Matrix4& operator =(const Matrix4 &other);//��ֵ����


	float& operator() (int x, int y) { return m_data[ x + y * 4 ]; }

	const float operator() (int x, int y) const 
	{ return m_data[ x + y * 4 ]; }

	void GetData(float mat[16]);

	void Inverse();

	void SetMatrix(float mat[16]);

	void nullMat(void); //���þ���Ϊ0

	void identityMat(void); //����Ϊ��λ����

	void rotationMat(Vector4& dir, float rad); //��dir��תrad

	void transposeMat(void);//����ת��

	//���س��������
//	void operator = (const Matrix4& m1);

	bool operator == (const Matrix4& m1);

	bool operator != (const Matrix4& m1);

	void operator += (const Matrix4& m1); 

	void operator -= (const Matrix4& m1);

	void operator /= (float scal);

	void operator *= (float scal);

	Matrix4 operator * (const Matrix4& m1); //mat4��4 * mat4��4

	Vector4 operator * (const Vector4& v); //mat4��4 * vector4

};
#endif