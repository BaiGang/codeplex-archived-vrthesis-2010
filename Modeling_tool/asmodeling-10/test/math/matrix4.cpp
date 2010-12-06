//Copyright (c) 2005 HU Yong (huyong@vrlab.buaa.edu.cn)
//  All rights reserved.
#include "StdAfx.h"


#include "geomath.h"
#include <iomanip>
using namespace std;
const float EPSILON = 1.0e-8f;

#pragma warning(disable: 4405)

Matrix4::Matrix4(float m00, float m01, float m02, float m03,
				 float m10, float m11, float m12, float m13,
				 float m20, float m21, float m22, float m23,
				 float m30, float m31, float m32, float m33)
{
	m_data[0]=m00;
	m_data[1]=m10;
	m_data[2]=m20;
	m_data[3]=m30;

	m_data[4]=m01;
	m_data[5]=m11;
	m_data[6]=m21;
	m_data[7]=m31;

	m_data[8]=m02;
	m_data[9]=m12;
	m_data[10]=m22;
	m_data[11]=m32;

	m_data[12]=m03;
	m_data[13]=m13;
	m_data[14]=m23;
	m_data[15]=m33;

	
}

Matrix4::Matrix4()
{
	memset(m_data, 0, sizeof(m_data));
}

Matrix4::Matrix4(float mat[16])
{
	for(int i = 0; i < 16; ++i)
	{
		m_data[i] = mat[i];
	}
}
Matrix4::Matrix4(const Matrix4 &other)
{
	memcpy(m_data, other.m_data, sizeof(float)*16);
}

Matrix4& Matrix4::operator=(const Matrix4 &other)
{
	if(*this == other)
	{
		return *this;
	}
	memcpy(m_data, other.m_data, sizeof(float)*16);
	return *this;
}

void Matrix4::SetMatrix(float mat[16])
{
	for(int i = 0; i < 16; ++i)
	{
		m_data[i] = mat[i];
	}
}

void Matrix4::nullMat()
{
	memset(m_data, 0, sizeof(m_data));
}

void Matrix4::identityMat()
{
	memset(m_data, 0, sizeof(m_data));
	(*this)(0,0) = 1.0;
	(*this)(1,1) = 1.0;
	(*this)(2,2) = 1.0;
	(*this)(3,3) = 1.0;
	
}

void Matrix4::transposeMat()
{
	
	for(int i = 0; i < 4; i++)
		for(int j = 0; j < i; j++)
			swap((*this)(i,j), (*this)(j,i));
			
}
/*
void Matrix4::operator = (const Matrix4& m1)
{
	
	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < 4; j++)
		{
			(*this)(i,j) = m1(i,j);
		}
	}
	
}
*/
bool Matrix4::operator == (const Matrix4& m2)
{
	
	if( 
		(*this)(0,0) != m2(0,0) || (*this)(0,1) != m2(0,1) || (*this)(0,2) != m2(0,2) || (*this)(0,3) != m2(0,3) ||
		(*this)(1,0) != m2(1,0) || (*this)(1,1) != m2(1,1) || (*this)(1,2) != m2(1,2) || (*this)(1,3) != m2(1,3) ||
		(*this)(2,0) != m2(2,0) || (*this)(2,1) != m2(2,1) || (*this)(2,2) != m2(2,2) || (*this)(2,3) != m2(2,3) ||
		(*this)(3,0) != m2(3,0) || (*this)(3,1) != m2(3,1) || (*this)(3,2) != m2(3,2) || (*this)(3,3) != m2(3,3) )
	{
		return false;
	}
	else
	{
		return true;
	}
	

}

bool Matrix4::operator != (const Matrix4& m2)
{
	if((*this)==m2)
	{
		return false;
	}
	else
	{
		return true;
	}
}

void Matrix4::operator += (const Matrix4& m1)
{	

	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < 4; j++)
		{
			(*this)(i,j) +=  m1(i,j);
		}
	}

}

void Matrix4::operator -= (const Matrix4& m1)
{
	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < 4; j++)
		{
			(*this)(i,j) -=  m1(i,j);
		}
	}
}

void Matrix4::operator *= (float scal)
{

	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < 4; j++)
		{
			(*this)(i,j) *= scal;
		}
	}

}

void Matrix4::operator /= (float scal)
{
	assert(scal != 0);
	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < 4; j++)
		{
			(*this)(i,j) /= scal;
		}
	}
}


Matrix4 Matrix4::operator * (const Matrix4& m1)
{
	Matrix4 mat2;
	int i,j,k;
	float ab;
	for(i = 0; i < 4; i++)
	{
		for(j = 0; j < 4; j++)
		{
			ab = 0.0f;
			for(k = 0; k < 4; k++)
			{
				ab += (*this)(i,k) * m1(k,j);
			}
			mat2(i,j) = ab;
		}
	}
	return mat2;

}

Vector4 Matrix4::operator * (const Vector4& v)
{
	Vector4 ret;

	ret.x = m_data[0]*v.x + m_data[4]*v.y + m_data[8]*v.z + m_data[12]*v.w;
	ret.y = m_data[1]*v.x + m_data[5]*v.y + m_data[9]*v.z + m_data[13]*v.w;
	ret.z = m_data[2]*v.x + m_data[6]*v.y + m_data[10]*v.z + m_data[14]*v.w;
	ret.w = m_data[3]* v.x + m_data[7]*v.y + m_data[11]*v.z + m_data[15]*v.w;

	return ret;

}


void Matrix4::Inverse()
{
	
		//GGemsII, K.Wu, Fast Matrix Inversion 

		Matrix4& m = *this;
		int i,j,k;               
		int pvt_i[4], pvt_j[4];            /* Locations of pivot elements */
		float pvt_val;               /* Value of current pivot element */
		float hold;                  /* Temporary storage */
		float determinat;            

		determinat = 1.0f;
		for (k=0; k<4; k++)  {
			/* Locate k'th pivot element */
			pvt_val=m(k, k);            /* Initialize for search */
			pvt_i[k]=k;
			pvt_j[k]=k;
			for (i=k; i<4; i++) {
				for (j=k; j<4; j++) {
					if (fabs(m(i, j)) > fabs(pvt_val)) {
						pvt_i[k]=i;
						pvt_j[k]=j;
						pvt_val=m(i, j);
					}
				}
			}

			/* Product of pivots, gives determinant when finished */
			determinat*=pvt_val;
			if (fabs(determinat) < EPSILON) {    
				return;  /* Matrix is singular (zero determinant) */
			}

			/* "Interchange" rows (with sign change stuff) */
			i=pvt_i[k];
			if (i!=k) {               /* If rows are different */
				for (j=0; j<4; j++) {
					hold=-m(k, j);
					m(k, j) = m(i, j);
					m(i, j) = hold;
				}
			}

			/* "Interchange" columns */
			j=pvt_j[k];
			if (j!=k) {              /* If columns are different */
				for (i=0; i<4; i++) {
					hold=-m(i, k);
					m(i, k) = m(i, j);
					m(i, j)=hold;
				}
			}

			/* Divide column by minus pivot value */
			for (i=0; i<4; i++) {
				if (i!=k) m(i, k)/=( -pvt_val) ; 
			}
 
			/* Reduce the matrix */
			for (i=0; i<4; i++) {
				hold = m(i, k);
				for (j=0; j<4; j++) {
					if (i!=k && j!=k) 
					{
						m(i, j) += hold * m(k, j);
					}
				}
			}

			/* Divide row by pivot */
			for (j=0; j<4; j++) {
				if (j!=k) 
					m(k, j) /= pvt_val;
			}

			/* Replace pivot by reciprocal (at last we can touch it). */
			m(k, k) = 1.0f / pvt_val;
		}

		/* That was most of the work, one final pass of row/column interchange */
		/* to finish */
		for (k=4-2; k>=0; k--) { /* Don't need to work with 1 by 1 corner*/
			i=pvt_j[k];            /* Rows to swap correspond to pivot COLUMN */
			if (i!=k) {            /* If rows are different */
				for(j=0; j<4; j++) {
					hold = m(k, j);
					m(k, j) =- m(i, j);
					m(i, j) = hold;
				}
			}

			j=pvt_i[k];           /* Columns to swap correspond to pivot ROW */
			if (j!=k)             /* If columns are different */
				for (i=0; i<4; i++) {
					hold=m(i, k);
					m(i, k) =- m(i, j);
					m(i, j) = hold;
				}
		}

}

void Matrix4::GetData(float mat[16])
{
	for(int i = 0; i < 16; ++i)
	{
		 mat[i] = m_data[i];
	}
}




