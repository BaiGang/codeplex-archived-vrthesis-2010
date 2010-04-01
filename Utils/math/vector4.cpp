#include "geomath.h"
using namespace std;



#pragma warning(disable: 4405)

Vector4::Vector4()
{
	x = y = z = 0;
	w = 0.0f;
}

//定义Vector4类的各个接口
Vector4::Vector4(float x0, float y0, float z0, float w0 /*= 1.0f*/)
{
	x = x0;
	y = y0;
	z = z0;
	w = w0;
}

void Vector4::setVec(float x0, float y0, float z0, float w0 /*= 1.0f*/)
{
	x = x0;
	y = y0;
	z = z0;
	w = w0;
}

void Vector4::nullVec()
{
	x = y = z = 0;
	w = 0.0f;
}

float& Vector4::operator [](int i)
{
	assert(i < 4);
	return (&x)[i];
}

float Vector4::lengthVec()
{
	return (float)sqrt(x*x + y*y + z*z);
}

void Vector4::negateVec()
{
	x = -x;
	y = -y;
	z = -z;
}

void Vector4::normaVec()
{
	float len = (float)sqrt(x*x + y*y + z*z);

	if(0.0f == len)
		return;

	x /= len;
	y /= len;
	z /= len;

	//当Vector4中四个变量作为平面方程系数，进行单位化处理要处理w
	w /= len;
}

void Vector4::crossVec(const Vector4& v1, const Vector4& v2)
{
#ifdef SIMD
	_asm
	{
		mov esi, v1
		mov edi, v2
		movups xmm0, [esi]
		movups xmm1, [edi]
		movaps xmm2, xmm0
		movaps xmm3, xmm1

		//xmm0中存的是v1，经过shufps打乱后xmm0中的顺序为y,z,x,w
		shufps xmm0, xmm0, 0xc9
		//xmm1中存的是v2，经过shufps打乱后xmm0中的顺序为z,x,y,w
		shufps xmm1, xmm1, 0xd2
		mulps xmm0, xmm1
		
		//xmm2中存的是v1，经过shufps打乱后xmm0中的顺序为z,x,y,w
		shufps xmm2, xmm2, 0xd2
		//xmm3中存的是v2，经过shufps打乱后xmm0中的顺序为y,z,x,w
		shufps xmm3, xmm3, 0xc9
		mulps xmm2, xmm3

		subps xmm0, xmm2
		mov esi, this
		movups [esi], xmm0
	}
#else
	x = v1.y * v2.z - v1.z * v2.y;
	y = v1.z * v2.x - v1.x * v2.z;
	z = v1.x * v2.y - v1.y * v2.x;	
#endif

}

float Vector4::dotVec(const Vector4& v)
{
	return (x * v.x + y * v.y + z * v.z);
/*
#ifdef SIMD
	_asm
	{
		mov esi, v1
		mov edi, v2
		movaps xmm0, [esi]
		movaps xmm1, [edi]
		mulps xmm0, xmm1

	}
#endif*/

}

Vector4& Vector4::operator = (const Vector4& v)
{
	this->x = v.x;
	this->y = v.y;
	this->z = v.z;
	this->w = v.w;

	return *this;
}

bool Vector4::operator == (const Vector4& v)
{
	return(x == v.x && 
		y == v.y &&
		z == v.z &&
		w == v.w);
}

bool Vector4::operator != (const Vector4& v)
{
	return(x != v.x ||
		y != v.y ||
		z != v.z ||
		w != v.w);
}

void Vector4::operator += (const Vector4& v)
{
#ifdef SIMD
	_asm
	{
		mov esi, this
		mov edi, v
		movups xmm0, [esi]
		movups xmm1, [edi]
		addps xmm0, xmm1
		movups [esi], xmm0
	}
#else
	x += v.x;
	y += v.y;
	z += v.z;
	w += v.w;
#endif
}

void Vector4::operator -= (const Vector4& v)
{
#ifdef  SIMD
	_asm
	{
		mov esi, this
		mov edi, v
		movups xmm0, [esi]
		movups xmm1, [edi]
		subps xmm0, xmm1
		movups [esi],xmm0
	}
#else
	x -= v.x;
	y -= v.y;
	z -= v.z;
	w -= v.w;
#endif
}

void Vector4::operator *= (float scal)
{
#ifdef SIMD
	_asm
	{
		mov esi, this
		movups xmm0, [esi]
		movss xmm1, scal  //xmm1[0] = scal
		shufps xmm1, xmm1, 0  //xmm1[1,2,3] = xmm1[0]
		mulps xmm0, xmm1
		movups [esi], xmm0
	}
#else
	x *= scal;
	y *= scal;
	z *= scal;
	w *= scal;
#endif

}

void Vector4::operator /= (float scal)
{
	assert(scal != 0);
#ifdef SIMD
	_asm
	{
		mov esi, this
		movups xmm0, [esi]
		movss xmm1, scal
		shufps xmm1, xmm1, 0
		divps xmm0, xmm1
		movups [esi], xmm0
	}
#else
	x /= scal;
	y /= scal;
	z /= scal;
	w /= scal;
#endif

}

Vector4 Vector4::operator + (const Vector4& v)
{
	_declspec(align(16)) Vector4 ret;
#ifdef SIMD
	_asm
	{
		mov esi, this
		mov edi, v
		movups xmm0, [esi]
		movups xmm1, [edi]
		addps xmm0, xmm1
		movups ret, xmm0
	}
#else
	ret.x = x + v.x;
	ret.y = y + v.y;
	ret.z = z + v.z;
	ret.w = w + v.w;
#endif
	return ret;

}

Vector4 Vector4::operator - (const Vector4& v)
{
	_declspec(align(16)) Vector4 ret;
#ifdef SIMD
	_asm
	{
		mov esi, this
		mov edi, v
		movups xmm0, [esi]
		movups xmm1, [edi]
		subps xmm0, xmm1
		movups ret, xmm0
	}
#else
	ret.x = x - v.x;
	ret.y = y - v.y;
	ret.z = z - v.z;
	ret.w = w - v.w;
#endif
	return ret;
}

Vector4 Vector4::operator * (float scal)
{
	_declspec(align(16)) Vector4 ret;
#ifdef SIMD
	_asm
	{
		mov esi, this
		movss xmm1, scal
		shufps xmm1, xmm1, 0
		movups xmm0, [esi]
		mulps xmm0, xmm1
		movups ret, xmm0
	}
#else
	ret.x = x * scal;
	ret.y = y * scal;
	ret.z = z * scal;
	ret.w = w * scal;
#endif
	return ret;
}

Vector4 Vector4::operator / (float scal)
{
	assert(scal != 0);
	_declspec(align(16)) Vector4 ret;
#ifdef SIMD
	_asm
	{
		mov esi, this
		movss xmm1, scal
		shufps xmm1, xmm1, 0
		movups xmm0, [esi]
		divps xmm0, xmm1
		movups ret, xmm0
	}
#else
	ret.x = x / scal;
	ret.y = y / scal;
	ret.z = z / scal;
	ret.w = w / scal;
#endif
	return ret;
}

Vector4 Vector4::operator *(const Matrix4& mat)
{
	_declspec(align(16)) Vector4 ret;
#ifdef SIMD
	_asm
	{
		mov esi, this
		mov edi, mat

		movups xmm0, [esi]
		movaps xmm1, xmm0
		movaps xmm2, xmm0
		movaps xmm3, xmm0

		shufps xmm0, xmm2, 0x00
		shufps xmm1, xmm2, 0x55
		shufps xmm2, xmm2, 0xaa
		shufps xmm3, xmm3, 0xff

		movups xmm4, [edi]
		movups xmm5, [edi + 16]
		movups xmm6, [edi + 32]
		movups xmm7, [edi + 48]

		mulps xmm0, xmm4
		mulps xmm1, xmm5
		mulps xmm2, xmm6
		mulps xmm3, xmm7

		addps xmm0, xmm1
		addps xmm0, xmm2
		addps xmm0, xmm3

		movups ret, xmm0
	}
#else

	ret.x = x * mat.m[0][0] + y * mat.m[0][1] + z * mat.m[0][2] + w * mat.m[0][3];
	ret.y = x * mat.m[1][0] + y * mat.m[1][1] + z * mat.m[1][2] + w * mat.m[1][3];
	ret.z = x * mat.m[2][0] + y * mat.m[2][1] + z * mat.m[2][2] + w * mat.m[2][3];
	ret.w = x * mat.m[3][0] + y * mat.m[3][1] + z * mat.m[3][2] + w * mat.m[3][3];
#endif
	return ret;
}



























