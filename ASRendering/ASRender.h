#ifndef __AS_RENDER_H__
#define __AS_RENDER_H__

#include <cmath>
#include <GL/glew.h>
#include <GL/glut.h>
#include <AntTweakBar.h>
#include "../Utils/shader/GLSLShader.h"
#include "../Utils/image/PFMImage.h"
#include "../Utils/math/geomath.h"

namespace as_rendering
{

	// GLUT CallBack functions
	void Display(void);
	void Reshape(int width, int height);
	void Terminate(void);

	//! set the AntTweakBar UI
	void SetTweakBar(void);

	//! set GL related resources
	void Init(void);

	void Snap(int camera, int frame);

	typedef enum { DELEGATE_BOX=1, DENSITY, LIGHT } Model_Type;

	// Routine to set a quaternion from a rotation axis and angle
	// ( input axis = float[3] angle = float  output: quat = float[4] )
	inline void SetQuaternionFromAxisAngle(const float *axis, float angle, float *quat)
	{
		float sina2, norm;
		sina2 = (float)sin(0.5f * angle);
		norm = (float)sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
		quat[0] = sina2 * axis[0] / norm;
		quat[1] = sina2 * axis[1] / norm;
		quat[2] = sina2 * axis[2] / norm;
		quat[3] = (float)cos(0.5f * angle);
	}


	// Routine to convert a quaternion to a 4x4 matrix
	// ( input: quat = float[4]  output: mat = float[4*4] )
	inline void ConvertQuaternionToMatrix(const float *quat, float *mat)
	{
		float yy2 = 2.0f * quat[1] * quat[1];
		float xy2 = 2.0f * quat[0] * quat[1];
		float xz2 = 2.0f * quat[0] * quat[2];
		float yz2 = 2.0f * quat[1] * quat[2];
		float zz2 = 2.0f * quat[2] * quat[2];
		float wz2 = 2.0f * quat[3] * quat[2];
		float wy2 = 2.0f * quat[3] * quat[1];
		float wx2 = 2.0f * quat[3] * quat[0];
		float xx2 = 2.0f * quat[0] * quat[0];
		mat[0*4+0] = - yy2 - zz2 + 1.0f;
		mat[0*4+1] = xy2 + wz2;
		mat[0*4+2] = xz2 - wy2;
		mat[0*4+3] = 0;
		mat[1*4+0] = xy2 - wz2;
		mat[1*4+1] = - xx2 - zz2 + 1.0f;
		mat[1*4+2] = yz2 + wx2;
		mat[1*4+3] = 0;
		mat[2*4+0] = xz2 + wy2;
		mat[2*4+1] = yz2 - wx2;
		mat[2*4+2] = - xx2 - yy2 + 1.0f;
		mat[2*4+3] = 0;
		mat[3*4+0] = mat[3*4+1] = mat[3*4+2] = 0;
		mat[3*4+3] = 1;
	}


	// Routine to multiply 2 quaternions (ie, compose rotations)
	// ( input q1 = float[4] q2 = float[4]  output: qout = float[4] )
	inline void MultiplyQuaternions(const float *q1, const float *q2, float *qout)
	{
		float qr[4];
		qr[0] = q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1];
		qr[1] = q1[3]*q2[1] + q1[1]*q2[3] + q1[2]*q2[0] - q1[0]*q2[2];
		qr[2] = q1[3]*q2[2] + q1[2]*q2[3] + q1[0]*q2[1] - q1[1]*q2[0];
		qr[3]  = q1[3]*q2[3] - (q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2]);
		qout[0] = qr[0]; qout[1] = qr[1]; qout[2] = qr[2]; qout[3] = qr[3];
	}

	extern int i_frame;
} // namespace as_rendering

#endif //__AS_RENDER_H__