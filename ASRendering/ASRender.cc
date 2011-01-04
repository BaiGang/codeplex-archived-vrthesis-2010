#include "ASRender.h"
#include <cstdio>
#include "../Utils/Timer/CPUTimer.h"
#include "../Utils/image/BMPImage.h"

using namespace as_rendering;

//#define USE_CAMERA_POS

#ifndef USE_CAMERA_POS
const float BOX_SIZE = 25.0f;
const float g_CenterX = 0;;
const float g_CenterY = -0;
const float g_CenterZ = 0;
#else
const float BOX_SIZE = 17.0f;
const float g_CenterX = 15.0f;
const float g_CenterY = -9.0f;
const float g_CenterZ = 12.0f;
#endif

const int BOX_LENGTH = 128;


static Model_Type g_current_type = DENSITY;

const int CURRENT_CAMERA = 4;
const int FRAME = 20;
int as_rendering::i_frame = 2906;

// Shapes scale
static float g_Zoom = 0.75f;
// Shape orientation (stored as a quaternion)
static float g_Rotation[] = { 0.0f, 0.0f, 0.0f, 1.0f };
// Auto rotate
static int g_AutoRotate = 0;
static int g_RotateTime = 0;
static float g_RotateStart[] = { 0.0f, 0.0f, 0.0f, 1.0f };

//Auto Animate
static int g_AutoAnimate = 0;
//static int g_AnimateTime = 0;
static int g_AnimateFrame = 0;
const int TOTAL_FRAMES = 156;
const int START_FRAME = 2906;
// change texture
static Timer g_FrameTimer;

static GLuint g_vol_tex;
static GLSLShader g_shader_along_x;
static GLSLShader g_shader_along_y;
static GLSLShader g_shader_along_z;

static Vector4 g_CameraPos;
static float g_CameraInv[16];

static bool g_DrawDelegateBox = true;

// Light parameter
static float g_LightMultiplier = 250000.0f;
static float g_LightDist = 195.68271;
static float g_LightPosition[] = { 34.51900f/g_LightDist, -135.97600/g_LightDist, 136.42100/g_LightDist, 1.0f };
// PM properties
static float g_extinction = 0.1f;
static float g_scattering = 0.05f;
static TwBar * g_bar;


////////////////////
Matrix4 g_ModelViews[8];
Matrix4 g_Projections[8];
PFMImage * g_pTexImg;
////////////////////

void set_tex(int frame)
{
	fprintf(stderr, "Frame : %d\n", frame);

	char path_buf[128];
	sprintf(path_buf, "E:/bg data/asmodeling/Data/Results/Frame%08d.pfm", frame);
	g_pTexImg->ReadImage(path_buf);

	//for (int i = 0; i < BOX_LENGTH * BOX_LENGTH; ++i)
	//{
	//	for (int j = 0; j < BOX_LENGTH; ++j)
	//	{
	//		g_pTexImg
	//	}
	//}

	glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, BOX_LENGTH, BOX_LENGTH, BOX_LENGTH, GL_LUMINANCE, GL_FLOAT, g_pTexImg->GetPixelDataBuffer());
	//glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB, BOX_LENGTH, BOX_LENGTH, BOX_LENGTH, 0, GL_LUMINANCE, GL_FLOAT, g_pTexImg->GetPixelDataBuffer());

	// check GL error
	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
	{
		fprintf(stderr, "%s", glewGetErrorString(err));
		return ;
	}
}

void as_rendering::Init(void)
{
	if (!InitGLExtensions())
	{
		fprintf(stderr, "Init OpenGL extensions error.\n");
		return;
	}

	// init shaders
	g_shader_along_x.InitShaders(
		"../Data/GLSLShaders/RayMarchingBlend.vert",
		"../Data/GLSLShaders/RayMarchingBlendXU.frag");
	g_shader_along_y.InitShaders(
		"../Data/GLSLShaders/RayMarchingBlend.vert",
		"../Data/GLSLShaders/RayMarchingBlendYU.frag");
	g_shader_along_z.InitShaders(
		"../Data/GLSLShaders/RayMarchingBlend.vert",
		"../Data/GLSLShaders/RayMarchingBlendZU.frag");

	// load image 
	// and set texture
	g_pTexImg = new PFMImage(BOX_LENGTH, BOX_LENGTH*BOX_LENGTH, 0, NULL);

	// create gl texture for volume data storage
	glGenTextures(1, &g_vol_tex);
	glBindTexture(GL_TEXTURE_3D, g_vol_tex);

	// set basic texture parameters
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	//char pathbuf[128];
	//sprintf(pathbuf, "../Data/Results/Frame%08d.pfm", FRAME);
	//g_pTexImg->ReadImage(pathbuf);

	g_LightPosition[0] = 34.519 / g_LightDist;
	g_LightPosition[1] = -135.976 / g_LightDist;
	g_LightPosition[2] = 136.421 / g_LightDist;
	g_LightPosition[3] = 1.0f;


	g_pTexImg->ReadImage("E:/bg data/asmodeling/Data/Results/Frame00002906.pfm");

	assert(g_pTexImg->GetHeight() == BOX_LENGTH * BOX_LENGTH && g_pTexImg->GetWidth() == BOX_LENGTH);

	//for (int i = 0; i < BOX_LENGTH; ++i)
	//{
	//	for (int j = 0; j < BOX_LENGTH * BOX_LENGTH; ++j)
	//	{
	//		g_pTexImg->GetPixelDataBuffer()[j*BOX_LENGTH+i] *= 100.0f;
	//	}
	//}

	//PFMImage small_img(PFM);
	//small_img.ReadImage("D:/asmodeling2/Data/Results/midnight726/Frame00000000.pfm");
	
	//float * tmp_data = new float [BOX_LENGTH*BOX_LENGTH*BOX_LENGTH];
	//memset(tmp_data, 0, sizeof(float)*BOX_LENGTH*BOX_LENGTH*BOX_LENGTH);
	//for (int z = 0; z < BOX_LENGTH; ++z)
	//{
	//	//if (z < 50 || z > 200)
	//	//	continue;
	//	for (int y = 0; y < BOX_LENGTH; ++y)
	//	{
	//		//if (y < 64 || y > 192)
	//		//	continue;

	//		for (int x = 0; x < BOX_LENGTH; ++x)
	//		{
	//			if (x < 32 || x > 96)
	//				continue;
	//			tmp_data[BOX_LENGTH*BOX_LENGTH*z + BOX_LENGTH*y + x] = small_img.GetPixel(y, BOX_LENGTH * z + x).r * 3.0f;
	//		}
	//	}
	//}

	glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB, BOX_LENGTH, BOX_LENGTH, BOX_LENGTH, 0, GL_LUMINANCE, GL_FLOAT, g_pTexImg->GetPixelDataBuffer());

	// check GL error
	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
	{
		fprintf(stderr, "%s", glewGetErrorString(err));
		return ;
	}

	{
		// LOAD CAMERA DATA
		FILE * fp = fopen("D:/BaiGang/NEWNEW/asmodeling-56327/Data/camera.txt", "r");
		if (NULL == fp)
		{
			fprintf(stderr, "Open camera file error. \n");
			return ;
		}
		Matrix4 Intr, Extr;
		float ftmp;

		fscanf(fp, "%f", &ftmp);
		fprintf(stderr, "num of cameras %f\n", ftmp);
		for (int i = 0; i < 8; ++i)
		{
			// load in intrinsic parameters
			for (int row = 0; row < 3; ++row)
			{
				for (int col = 0; col < 3; ++col)
				{
					if (fscanf(fp, "%f", & ftmp)==1)
					{
						Intr(row,col) = ftmp;
						//fprintf(stderr, "row %d cole %d value %f\n",row, col, ftmp);
					}
					else
					{
						fprintf(stderr, "Error while loading camrea.");
						return;
					}
				} // col
			} // row

			for (int col = 0; col < 4; ++col)
			{
				if (fscanf(fp, "%f", & ftmp)==1)
				{
					Intr(3,col) = ftmp;
				}
				else
				{
					fprintf(stderr, "Error while loading camrea.");
					return;
				}
			}


			// load in extrinsic parameters
			for (int row = 0; row < 4; ++row)
			{
				for (int col = 0; col < 4; ++col)
				{
					if (fscanf(fp, "%f", & ftmp)==1)
					{
						Extr(row, col) = ftmp;
					}
					else
					{
						fprintf(stderr, "Error while loading camrea.");
						return;
					}
				} // col
			} // row

			// set 
			Matrix4 trans;
			trans.identityMat();
			trans(1,1) = -1;
			trans(2,2) = -1;
			g_ModelViews[i] = trans * Extr;

			float proj[16];
			float zn = 0.1f;
			float zf = 1000.0f;
			memset(proj, 0, sizeof(proj));
			proj[0] = 2.0f*Intr(0,0)/1024.0f;
			proj[5] = 2.0f*Intr(1,1)/1024.0f;
			proj[8] = -2.0f*(Intr(0,2)-1024.0f*0.5f)/1024.0f;
			proj[9] = 2.0f*(Intr(1,2)-1024.0f*0.5f)/1024.0f;
			proj[10] = -(zf+zn)/(zf-zn);
			proj[11] = -1.0f;
			proj[14] = -2.0f*(zf*zn)/(zf-zn);
			g_Projections[i].SetMatrix(proj);

		} // for i
	} // LOAD CAMERA DATA

	g_FrameTimer.start();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(40,1, 1, 1000); 

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(30, 40, 0, 0, 0, 0, 0, 0, 1);
	glScalef(0.6, 0.6, 0.6);
}

// call back functions for atw 
void TW_CALL SetAutoRotateCB(const void *value, void *clientData);
void TW_CALL GetAutoRotateCB(void *value, void *clientData);

void TW_CALL SetAutoFrameCB(const void *value, void *clientData);
void TW_CALL GetAutoFrameCB(void *value, void *clientData);

void TW_CALL SetNextFrameCB(const void *value, void *clientData);
void TW_CALL GetNextFrameCB(void *value, void *clientData);
void TW_CALL SetPrevFrameCB(const void *value, void *clientData);
void TW_CALL GetPrevFrameCB(void *value, void *clientData);


void as_rendering::SetTweakBar()
{
	// Initialize AntTweakBar
	// (note that AntTweakBar could also be intialized after GLUT, no matter)
	if( !TwInit(TW_OPENGL, NULL) )
	{
		// A fatal error occured    
		fprintf(stderr, "AntTweakBar initialization failed: %s\n", TwGetLastError());
		return;
	}

	// Set GLUT event callbacks
	// - Directly redirect GLUT mouse button events to AntTweakBar
	glutMouseFunc((GLUTmousebuttonfun)TwEventMouseButtonGLUT);
	// - Directly redirect GLUT mouse motion events to AntTweakBar
	glutMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT);
	// - Directly redirect GLUT mouse "passive" motion events to AntTweakBar (same as MouseMotion)
	glutPassiveMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT);
	// - Directly redirect GLUT key events to AntTweakBar
	glutKeyboardFunc((GLUTkeyboardfun)TwEventKeyboardGLUT);
	// - Directly redirect GLUT special key events to AntTweakBar
	glutSpecialFunc((GLUTspecialfun)TwEventSpecialGLUT);
	// - Send 'glutGetModifers' function pointer to AntTweakBar;
	//   required because the GLUT key event functions do not report key modifiers states.
	TwGLUTModifiersFunc(glutGetModifiers);

	// Create a tweak bar
	g_bar = TwNewBar("Smoke Reconstruction");
	TwDefine(" TweakBar size='240 400' color='96 216 224' "); // change default tweak bar size and color

	// Add 'g_Zoom' to 'bar': this is a modifable (RW) variable of type TW_TYPE_FLOAT. Its key shortcuts are [z] and [Z].
	TwAddVarRW(g_bar, "Zoom", TW_TYPE_FLOAT, &g_Zoom, 
		" min=0.01 max=2.5 step=0.01 keyIncr=z keyDecr=Z help='Scale the object (1=original size).' ");

	// Add 'g_Rotation' to 'bar': this is a variable of type TW_TYPE_QUAT4F which defines the object's orientation
	TwAddVarRW(g_bar, "ObjRotation", TW_TYPE_QUAT4F, &g_Rotation, 
		" label='Object rotation' open help='Change the object orientation.' ");

	// Add callback to toggle auto-rotate mode (callback functions are defined above).
	TwAddVarCB(g_bar, "AutoRotate", TW_TYPE_BOOL32, SetAutoRotateCB, GetAutoRotateCB, NULL, 
		" label='Auto-rotate' key=space help='Toggle auto-rotate mode.' ");

	// Add callback to toggle auto-rotate mode (callback functions are defined above).
	TwAddVarCB(g_bar, "AutoFrame", TW_TYPE_BOOL32, SetAutoFrameCB, GetAutoFrameCB, NULL, 
		" label='Auto-animate' key=enter help='Toggle animate mode.' ");

	// Add callback to toggle NEXT FRAME (callback functions are defined above).
	TwAddVarCB(g_bar, "NextFrame", TW_TYPE_BOOL32, SetNextFrameCB, GetNextFrameCB, NULL, 
		" label='Next-Frame' key=n help='Toggle next frame.' ");

	// Add callback to toggle PREV FRAME (callback functions are defined above).
	TwAddVarCB(g_bar, "PrevFrame", TW_TYPE_BOOL32, SetPrevFrameCB, GetPrevFrameCB, NULL, 
		" label='Prev-Frame' key=p help='Toggle previous frame.' ");

	// Add 'g_LightMultiplier' to 'bar': this is a variable of type TW_TYPE_FLOAT. Its key shortcuts are [+] and [-].
	TwAddVarRW(g_bar, "Intensity", TW_TYPE_FLOAT, &g_LightMultiplier, 
		" label='Light Intensity' min=100000 max=700000.0 step=5000 keyIncr='+' keyDecr='-' help='Increase/decrease the light power.' ");

	// Add 'g_LightMultiplier' to 'bar': this is a variable of type TW_TYPE_FLOAT. Its key shortcuts are [+] and [-].
	TwAddVarRW(g_bar, "Scattering", TW_TYPE_FLOAT, &g_scattering, 
		" label='Scattering coeff' min=0.01 max=0.1 step=0.01 keyIncr=']' keyDecr='[' help='Increase/decrease the scattering.' ");

	// Add 'g_LightMultiplier' to 'bar': this is a variable of type TW_TYPE_FLOAT. Its key shortcuts are [+] and [-].
	TwAddVarRW(g_bar, "Extinction", TW_TYPE_FLOAT, &g_extinction, 
		" label='Extinction coeff' min=0.01 max=0.2 step=0.01 keyIncr=''' keyDecr=';' help='Increase/decrease the extinction.' ");

	// Add 'g_LightDirection' to 'bar': this is a variable of type TW_TYPE_DIR3F which defines the light direction
	TwAddVarRW(g_bar, "LightPos", TW_TYPE_DIR3F, &g_LightPosition, 
		" label='Light Position' open help='Change the light Position.' ");

}

void DrawBox(void)
{
	float pos = 0.5f * BOX_SIZE;

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	glColor3f(1.0f, 1.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(g_CenterX-pos, g_CenterY-pos, g_CenterZ-pos);	glVertex3f(g_CenterX+pos, g_CenterY-pos, g_CenterZ-pos);
	glVertex3f(g_CenterX+pos, g_CenterY-pos, g_CenterZ-pos);	glVertex3f(g_CenterX+pos, g_CenterY+pos, g_CenterZ-pos);
	glVertex3f(g_CenterX+pos, g_CenterY+pos, g_CenterZ-pos);	glVertex3f(g_CenterX-pos, g_CenterY+pos, g_CenterZ-pos);
	glVertex3f(g_CenterX-pos, g_CenterY+pos, g_CenterZ-pos);	glVertex3f(g_CenterX-pos, g_CenterY-pos, g_CenterZ-pos);

	glVertex3f(g_CenterX-pos, g_CenterY-pos, g_CenterZ+pos);	glVertex3f(g_CenterX-pos, g_CenterY-pos, g_CenterZ-pos);
	glVertex3f(g_CenterX-pos, g_CenterY+pos, g_CenterZ+pos);	glVertex3f(g_CenterX-pos, g_CenterY+pos, g_CenterZ-pos);
	glVertex3f(g_CenterX+pos, g_CenterY+pos, g_CenterZ+pos);	glVertex3f(g_CenterX+pos, g_CenterY+pos, g_CenterZ-pos);
	glVertex3f(g_CenterX+pos, g_CenterY-pos, g_CenterZ+pos);	glVertex3f(g_CenterX+pos, g_CenterY-pos, g_CenterZ-pos);

	glVertex3f(g_CenterX-pos, g_CenterY-pos, g_CenterZ+pos);	glVertex3f(g_CenterX+pos, g_CenterY-pos, g_CenterZ+pos);
	glVertex3f(g_CenterX+pos, g_CenterY-pos, g_CenterZ+pos);	glVertex3f(g_CenterX+pos, g_CenterY+pos, g_CenterZ+pos);
	glVertex3f(g_CenterX+pos, g_CenterY+pos, g_CenterZ+pos);	glVertex3f(g_CenterX-pos, g_CenterY+pos, g_CenterZ+pos);
	glVertex3f(g_CenterX-pos, g_CenterY+pos, g_CenterZ+pos);	glVertex3f(g_CenterX-pos, g_CenterY-pos, g_CenterZ+pos);
	glEnd();
}

// 1-x, 2-y, 3-z, 4-error
int GetOrientation()
{
	Matrix4 trn;
	Matrix4 mat;
	Matrix4 cmrE;
	Vector4 cmrPos;
	float arr[16];

	glGetFloatv(GL_MODELVIEW_MATRIX, arr);
	mat.SetMatrix(arr);

	trn.identityMat();
	trn(1, 1) = -1.0f;
	trn(2, 2) = -1.0f;

	cmrE = trn * mat;
	cmrE.Inverse();

	cmrPos.x = cmrE(0,3);
	cmrPos.y = cmrE(1,3);
	cmrPos.z = cmrE(2,3);
	cmrPos.w = 1.0;

	// store the current camera position for rendering 
	g_CameraPos = cmrPos;

	// store the inversed modelview matrix
	mat.Inverse();
	mat.GetData(g_CameraInv);

	// direction
	cmrPos.normaVec();
	int res = 0;
	if (abs(cmrPos.x)>abs(cmrPos.y) && abs(cmrPos.x)>abs(cmrPos.z))
	{
		if (cmrPos.x < 0.0f)
			res = 1;
		else
			res = 4;
	}
	else if (abs(cmrPos.y)>abs(cmrPos.x) && abs(cmrPos.y)>abs(cmrPos.z))
	{
		if (cmrPos.y < 0.0f)
			res = 2;
		else
			res = 5;
	}
	else if (abs(cmrPos.z)>abs(cmrPos.x) && abs(cmrPos.z)>abs(cmrPos.y))
	{
		if (cmrPos.z < 0.0f)
			res = 3;
		else
			res = 6;
	}

	return res;
}

void DrawSmoke_alongxN(void)
{
	g_shader_along_x.Begin();

	g_shader_along_x.SetUniform3f("lightIntensity", g_LightMultiplier, g_LightMultiplier, g_LightMultiplier);
	g_shader_along_x.SetUniform4f("lightPosWorld",
		g_LightPosition[0] * g_LightDist,
		g_LightPosition[1] * g_LightDist,
		g_LightPosition[2] * g_LightDist, 1.0f);
	g_shader_along_x.SetUniform1f("absorptionCoefficient", g_extinction);
	g_shader_along_x.SetUniform1f("scatteringCoefficient", g_scattering);
	
	float step_size = BOX_SIZE / (1.0f * BOX_LENGTH);
	
	g_shader_along_x.SetUniform1f("stepSize", step_size);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, g_vol_tex);
	g_shader_along_x.SetUniform1i("volumeTex", 0);
	g_shader_along_x.SetUniform4f("cameraPos", 
		g_CameraPos.x, g_CameraPos.y,
		g_CameraPos.z, g_CameraPos.w);
	g_shader_along_x.SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, g_CameraInv);

	float pos = 0.5f * BOX_SIZE;

	for (int slice = 0; slice < BOX_LENGTH; ++slice)
	{
		float tex_x_coord = (slice+0.5f) / (1.0f * BOX_LENGTH);
		float geo_x_coord = g_CenterX + tex_x_coord * BOX_SIZE - pos;

		glBegin(GL_QUADS);
		glTexCoord3f(tex_x_coord, 0.0f, 0.0f); glVertex3f(geo_x_coord, g_CenterY - pos, g_CenterZ - pos);
		glTexCoord3f(tex_x_coord, 1.0f, 0.0f); glVertex3f(geo_x_coord, g_CenterY + pos, g_CenterZ - pos);
		glTexCoord3f(tex_x_coord, 1.0f, 1.0f); glVertex3f(geo_x_coord, g_CenterY + pos, g_CenterZ + pos);
		glTexCoord3f(tex_x_coord, 0.0f, 1.0f); glVertex3f(geo_x_coord, g_CenterY - pos, g_CenterZ + pos);
		glEnd();
	}

	g_shader_along_x.End();
}

void DrawSmoke_alongyN(void)
{
	g_shader_along_y.Begin();

	g_shader_along_y.SetUniform3f("lightIntensity", g_LightMultiplier, g_LightMultiplier, g_LightMultiplier);
	g_shader_along_y.SetUniform4f("lightPosWorld",
		g_LightPosition[0] * g_LightDist,
		g_LightPosition[1] * g_LightDist,
		g_LightPosition[2] * g_LightDist, 1.0f);
	g_shader_along_y.SetUniform1f("absorptionCoefficient", g_extinction);
	g_shader_along_y.SetUniform1f("scatteringCoefficient", g_scattering);
	
	float step_size = BOX_SIZE / (1.0f * BOX_LENGTH);
	
	g_shader_along_y.SetUniform1f("stepSize", step_size);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, g_vol_tex);
	g_shader_along_y.SetUniform1i("volumeTex", 0);
	g_shader_along_y.SetUniform4f("cameraPos", 
		g_CameraPos.x, g_CameraPos.y,
		g_CameraPos.z, g_CameraPos.w);
	g_shader_along_y.SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, g_CameraInv);

	float pos = 0.5f * BOX_SIZE;

	for (int slice = 0; slice < BOX_LENGTH; ++slice)
	{
		float tex_y_coord = (slice+0.5f) / (1.0f * BOX_LENGTH);
		float geo_y_coord = g_CenterY + tex_y_coord * BOX_SIZE - pos;

		glBegin(GL_QUADS);
		glTexCoord3f(0.0f, tex_y_coord, 0.0f); glVertex3f(g_CenterX - pos, geo_y_coord, g_CenterZ - pos);
		glTexCoord3f(1.0f, tex_y_coord, 0.0f); glVertex3f(g_CenterX + pos, geo_y_coord, g_CenterZ - pos);
		glTexCoord3f(1.0f, tex_y_coord, 1.0f); glVertex3f(g_CenterX + pos, geo_y_coord, g_CenterZ + pos);
		glTexCoord3f(0.0f, tex_y_coord, 1.0f); glVertex3f(g_CenterX - pos, geo_y_coord, g_CenterZ + pos);
		glEnd();
	}

	g_shader_along_y.End();
}

void DrawSmoke_alongzN(void)
{
	g_shader_along_z.Begin();

	g_shader_along_z.SetUniform3f("lightIntensity", g_LightMultiplier, g_LightMultiplier, g_LightMultiplier);
	g_shader_along_z.SetUniform4f("lightPosWorld",
		g_LightPosition[0] * g_LightDist,
		g_LightPosition[1] * g_LightDist,
		g_LightPosition[2] * g_LightDist, 1.0f);
	g_shader_along_z.SetUniform1f("absorptionCoefficient", g_extinction);
	g_shader_along_z.SetUniform1f("scatteringCoefficient", g_scattering);
	float step_size = BOX_SIZE / (1.0f * BOX_LENGTH);
	g_shader_along_z.SetUniform1f("stepSize", step_size);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, g_vol_tex);
	g_shader_along_z.SetUniform1i("volumeTex", 0);
	g_shader_along_z.SetUniform4f("cameraPos", 
		g_CameraPos.x, g_CameraPos.y,
		g_CameraPos.z, g_CameraPos.w);
	g_shader_along_z.SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, g_CameraInv);

	float pos = 0.5f * BOX_SIZE;

	for (int slice = 0; slice < BOX_LENGTH; ++slice)
	{
		float tex_z_coord = (slice+0.5f) / (1.0f*BOX_LENGTH);
		float geo_z_coord = g_CenterZ + tex_z_coord * BOX_SIZE - pos;

		glBegin(GL_QUADS);
		glTexCoord3f(0.0f, 0.0f, tex_z_coord); glVertex3f(g_CenterX - pos, g_CenterY - pos, geo_z_coord);
		glTexCoord3f(1.0f, 0.0f, tex_z_coord); glVertex3f(g_CenterX + pos, g_CenterY - pos, geo_z_coord);
		glTexCoord3f(1.0f, 1.0f, tex_z_coord); glVertex3f(g_CenterX + pos, g_CenterY + pos, geo_z_coord);
		glTexCoord3f(0.0f, 1.0f, tex_z_coord); glVertex3f(g_CenterX - pos, g_CenterY + pos, geo_z_coord);
		glEnd();
	}

	g_shader_along_z.End();
}

void DrawSmoke_alongxP(void)
{
	g_shader_along_x.Begin();

	g_shader_along_x.SetUniform3f("lightIntensity", g_LightMultiplier, g_LightMultiplier, g_LightMultiplier);
	g_shader_along_x.SetUniform4f("lightPosWorld",
		g_LightPosition[0] * g_LightDist,
		g_LightPosition[1] * g_LightDist,
		g_LightPosition[2] * g_LightDist, 1.0f);
	g_shader_along_x.SetUniform1f("absorptionCoefficient", g_extinction);
	g_shader_along_x.SetUniform1f("scatteringCoefficient", g_scattering);
	float step_size = BOX_SIZE / (1.0f*BOX_LENGTH);
	g_shader_along_x.SetUniform1f("stepSize", step_size);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, g_vol_tex);
	g_shader_along_x.SetUniform1i("volumeTex", 0);
	g_shader_along_x.SetUniform4f("cameraPos", 
		g_CameraPos.x, g_CameraPos.y,
		g_CameraPos.z, g_CameraPos.w);
	g_shader_along_x.SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, g_CameraInv);

	float pos = 0.5f * BOX_SIZE;

	for (int slice = BOX_LENGTH-1; slice >= 0; --slice)
	{
		float tex_x_coord = (slice+0.5f) / (1.0f*BOX_LENGTH);
		float geo_x_coord = g_CenterX + tex_x_coord * BOX_SIZE - pos;

		glBegin(GL_QUADS);
		glTexCoord3f(tex_x_coord, 0.0f, 0.0f); glVertex3f(geo_x_coord, g_CenterY - pos, g_CenterZ - pos);
		glTexCoord3f(tex_x_coord, 1.0f, 0.0f); glVertex3f(geo_x_coord, g_CenterY + pos, g_CenterZ - pos);
		glTexCoord3f(tex_x_coord, 1.0f, 1.0f); glVertex3f(geo_x_coord, g_CenterY + pos, g_CenterZ + pos);
		glTexCoord3f(tex_x_coord, 0.0f, 1.0f); glVertex3f(geo_x_coord, g_CenterY - pos, g_CenterZ + pos);
		glEnd();
	}

	g_shader_along_x.End();
}

void DrawSmoke_alongyP(void)
{
	g_shader_along_y.Begin();

	g_shader_along_y.SetUniform3f("lightIntensity", g_LightMultiplier, g_LightMultiplier, g_LightMultiplier);
	g_shader_along_y.SetUniform4f("lightPosWorld",
		g_LightPosition[0] * g_LightDist,
		g_LightPosition[1] * g_LightDist,
		g_LightPosition[2] * g_LightDist, 1.0f);
	g_shader_along_y.SetUniform1f("absorptionCoefficient", g_extinction);
	g_shader_along_y.SetUniform1f("scatteringCoefficient", g_scattering);
	float step_size = BOX_SIZE / (1.0f*BOX_LENGTH);
	g_shader_along_y.SetUniform1f("stepSize", step_size);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, g_vol_tex);
	g_shader_along_y.SetUniform1i("volumeTex", 0);
	g_shader_along_y.SetUniform4f("cameraPos", 
		g_CameraPos.x, g_CameraPos.y,
		g_CameraPos.z, g_CameraPos.w);
	g_shader_along_y.SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, g_CameraInv);

	float pos = 0.5f * BOX_SIZE;

	for (int slice = BOX_LENGTH-1; slice >= 0; --slice)
	{
		float tex_y_coord = (slice+0.5f) / (1.0f*BOX_LENGTH);
		float geo_y_coord = g_CenterY + tex_y_coord * BOX_SIZE - pos;

		glBegin(GL_QUADS);
		glTexCoord3f(0.0f, tex_y_coord, 0.0f); glVertex3f(g_CenterX - pos, geo_y_coord, g_CenterZ - pos);
		glTexCoord3f(1.0f, tex_y_coord, 0.0f); glVertex3f(g_CenterX + pos, geo_y_coord, g_CenterZ - pos);
		glTexCoord3f(1.0f, tex_y_coord, 1.0f); glVertex3f(g_CenterX + pos, geo_y_coord, g_CenterZ + pos);
		glTexCoord3f(0.0f, tex_y_coord, 1.0f); glVertex3f(g_CenterX - pos, geo_y_coord, g_CenterZ + pos);
		glEnd();
	}

	g_shader_along_y.End();
}

void DrawSmoke_alongzP(void)
{
	g_shader_along_z.Begin();

	g_shader_along_z.SetUniform3f("lightIntensity", g_LightMultiplier, g_LightMultiplier, g_LightMultiplier);
	g_shader_along_z.SetUniform4f("lightPosWorld",
		g_LightPosition[0] * g_LightDist,
		g_LightPosition[1] * g_LightDist,
		g_LightPosition[2] * g_LightDist, 1.0f);
	g_shader_along_z.SetUniform1f("absorptionCoefficient", g_extinction);
	g_shader_along_z.SetUniform1f("scatteringCoefficient", g_scattering);
	float step_size = BOX_SIZE / (1.0f*BOX_LENGTH);
	g_shader_along_z.SetUniform1f("stepSize", step_size);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, g_vol_tex);
	g_shader_along_z.SetUniform1i("volumeTex", 0);
	g_shader_along_z.SetUniform4f("cameraPos", 
		g_CameraPos.x, g_CameraPos.y,
		g_CameraPos.z, g_CameraPos.w);
	g_shader_along_z.SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, g_CameraInv);

	float pos = 0.5f * BOX_SIZE;

	for (int slice = BOX_LENGTH-1; slice >=0; --slice)
	{
		float tex_z_coord = (slice+0.5f) / (1.0f * BOX_LENGTH);
		float geo_z_coord = g_CenterZ + tex_z_coord * BOX_SIZE - pos;

		glBegin(GL_QUADS);
		glTexCoord3f(0.0f, 0.0f, tex_z_coord); glVertex3f(g_CenterX - pos, g_CenterY - pos, geo_z_coord);
		glTexCoord3f(1.0f, 0.0f, tex_z_coord); glVertex3f(g_CenterX + pos, g_CenterY - pos, geo_z_coord);
		glTexCoord3f(1.0f, 1.0f, tex_z_coord); glVertex3f(g_CenterX + pos, g_CenterY + pos, geo_z_coord);
		glTexCoord3f(0.0f, 1.0f, tex_z_coord); glVertex3f(g_CenterX - pos, g_CenterY + pos, geo_z_coord);
		glEnd();
	}

	g_shader_along_z.End();
}


void DrawSmoke(void)
{
	// first determine the camera orientation
	int orn = GetOrientation();

	//fprintf(stderr, "orientation : %d\n", orn);

	if (0 == orn)
		return ;


	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_DST_ALPHA, GL_ZERO);

	// for the corresponding orentation
	// draw the scene
	switch (orn)
	{
	case 1:
		DrawSmoke_alongxN();
		break;
	case 2:
		DrawSmoke_alongyN();
		break;
	case 3:
		DrawSmoke_alongzN();
		break;
	case 4:
		DrawSmoke_alongxP();
		break;
	case 5:
		DrawSmoke_alongyP();
		break;
	case 6:
		DrawSmoke_alongzP();
		break;
	default:
		fprintf(stderr, "Error orientation!\n");
		break;
	}
}

void as_rendering::Display(void)
{
	fprintf(stderr, "light pos: %f %f %f %f\n", g_LightPosition[0], g_LightPosition[1], g_LightPosition[2], g_LightDist);
	//set_tex(18);
	if (g_AutoAnimate)
	{
		if (g_FrameTimer.stop() > 0.05)
		{
			++g_AnimateFrame;
			if (TOTAL_FRAMES == g_AnimateFrame)
			{
				g_AnimateFrame = 0;
			}
			set_tex(g_AnimateFrame + START_FRAME);
			g_FrameTimer.start();
			fprintf(stderr, "Volume Frame %d\n", g_AnimateFrame + START_FRAME);
		}
	}

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Rotate & Draw
	glPushMatrix();

#ifdef USE_CAMERA_POS
	float mv[16], prj[16];
	g_ModelViews[CURRENT_CAMERA].GetData(mv);
	g_Projections[CURRENT_CAMERA].GetData(prj);

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(mv);
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(prj);
#else
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(40,1, 1, 1000); 

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(30, 40, 0, 0, 0, 0, 0, 0, 1);
#endif

	glTranslatef(g_CenterX, g_CenterY, g_CenterZ);

	if (g_AutoRotate)
	{
		//TODO : edit axis to enable arbitrary rotation...
		float axis[3] = {0.0f, 0.0f, 1.0f};
		float angle = (float)(glutGet(GLUT_ELAPSED_TIME)-g_RotateTime)/1000.0f;
		float quat[4];
		SetQuaternionFromAxisAngle(axis, angle, quat);
		MultiplyQuaternions(g_RotateStart, quat, g_Rotation);
	}
	float mat[16];

	ConvertQuaternionToMatrix(g_Rotation, mat);
	glMultMatrixf(mat);
	glScalef(g_Zoom, g_Zoom, g_Zoom);

	glTranslatef(-g_CenterX, -g_CenterY, -g_CenterZ);
	
	// draw smoke volume
	DrawSmoke();

	// draw delegate box
	if (g_DrawDelegateBox)
	{
		DrawBox();
	}

	//// draw light position
	//glPushMatrix();
	//glEnable(GL_DEPTH_TEST);
	//glDisable(GL_BLEND);
	//glTranslatef(g_CenterX+g_LightPosition[0], g_CenterY+g_LightPosition[1], g_CenterZ+g_LightPosition[2]);
	//glColor3f(0.0f, 1.0f, 1.0f);
	//glutSolidSphere(BOX_SIZE * 0.025, 18, 9);
	//glPopMatrix();

	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0,0,0); glVertex3f(10, 0, 0);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0,0,0); glVertex3f(0, 10, 0);
	glColor3f(0.0f, 0.0f, 1.0f);
	glVertex3f(0,0,0); glVertex3f(0, 0, 10);
	glEnd();

	glPopMatrix();

	// Draw tweak bars
	TwDraw();

	// Present frame buffer
	glutSwapBuffers();

	// Recall Display at next frame
	glutPostRedisplay();
}

// capture image
void as_rendering::Snap(int camera, int frame)
{
	set_tex(frame);

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, 1024, 1024);

#ifdef USE_CAMERA_POS
	float mv[16], prj[16];
	g_ModelViews[camera].GetData(mv);
	g_Projections[camera].GetData(prj);

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(mv);

	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(prj);
#else
	float axis[3] = {0.0f, 0.0f, 1.0f};
	float angle = 0.025f;
	float quat[4];
	SetQuaternionFromAxisAngle(axis, angle, quat);
	MultiplyQuaternions(g_RotateStart, quat, g_Rotation);
	float mat[16];
	ConvertQuaternionToMatrix(g_Rotation, mat);
	glMultMatrixf(mat);
#endif

	DrawSmoke();

	DrawBox();

	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0,0,0); glVertex3f(10, 0, 0);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0,0,0); glVertex3f(0, 10, 0);
	glColor3f(0.0f, 0.0f, 1.0f);
	glVertex3f(0,0,0); glVertex3f(0, 0, 10);
	glEnd();
}

void as_rendering::Reshape(int width, int height)
{
	// Set OpenGL viewport and camera
	glViewport(0, 0, width, height);

#ifndef USE_CAMERA_POS
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(40, (double)width/height, 1, 10);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0f,0.0f,4.0f, 0,0,0, 0,1.0f,0.0f);
#else
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(40, (double)width/height, 1, 10);

	float mv[16], prj[16];
	g_ModelViews[CURRENT_CAMERA].GetData(mv);
	g_Projections[CURRENT_CAMERA].GetData(prj);
	glLoadMatrixf(prj);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0f,0.0f,4.0f, 0,0,0, 0,1.0f,0.0f);
	glLoadMatrixf(mv);
#endif

	// Send the new window size to AntTweakBar
	TwWindowSize(width, height);
}

void as_rendering::Terminate(void)
{
	TwTerminate();
}


//  Callback function called when the 'AutoRotate' variable value of the tweak bar has changed
void TW_CALL SetAutoRotateCB(const void *value, void *clientData)
{
	(void)clientData; // unused

	g_AutoRotate = *(const int *)(value); // copy value to g_AutoRotate
	if( g_AutoRotate!=0 ) 
	{
		// init rotation
		g_RotateTime = glutGet(GLUT_ELAPSED_TIME);
		g_RotateStart[0] = g_Rotation[0];
		g_RotateStart[1] = g_Rotation[1];
		g_RotateStart[2] = g_Rotation[2];
		g_RotateStart[3] = g_Rotation[3];

		// make Rotation variable read-only
		TwDefine(" TweakBar/ObjRotation readonly ");
	}
	else
		// make Rotation variable read-write
		TwDefine(" TweakBar/ObjRotation readwrite ");
}


//  Callback function called by the tweak bar to get the 'AutoRotate' value
void TW_CALL GetAutoRotateCB(void *value, void *clientData)
{
	(void)clientData; // unused
	*(int *)(value) = g_AutoRotate; // copy g_AutoRotate to value
}

void TW_CALL SetAutoFrameCB(const void *value, void *clientData)
{
	(void)clientData; // unused

	g_AutoAnimate = *(const int *)value;

	if (g_AutoAnimate != 0)
	{
		g_FrameTimer.start();
	}
	else
	{
		//
	}
}

void TW_CALL GetAutoFrameCB(void *value, void *clientData)
{
	(void)clientData; // unused
	*(int*)(value) = g_AutoAnimate;
}

void TW_CALL SetNextFrameCB(const void *value, void *clientData)
{
	(void)clientData;
	(void)value;

	++ i_frame;
	if (i_frame > 3061)
	{
		i_frame = 2096;
	}
	set_tex(i_frame);
}

void TW_CALL GetNextFrameCB(void *value, void *clientData)
{
	(void)clientData; // unused
	*(int *)(value) = 1; // 
}

void TW_CALL SetPrevFrameCB(const void *value, void *clientData)
{	
	(void)clientData;
	(void)value;

	-- i_frame;
	if (i_frame < 2096)
	{
		i_frame = 3061;
	}
	set_tex(i_frame);

}

void TW_CALL GetPrevFrameCB(void *value, void *clientData)
{
	(void)clientData; // unused
	*(int *)(value) = 1; // 
}