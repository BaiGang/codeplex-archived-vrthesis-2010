#include <cstdio>
#include <cstdlib>
#include "ASRender.h"

#include "../Utils/shader/GLFBO.h"

using namespace as_rendering;

int main(int argc, char * argv[])
{



	// Initialize GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(800, 800);
	glutCreateWindow("Reconstructed Smoke Volume Result.");
	glutCreateMenu(NULL);

    // Set GLUT callbacks
    glutDisplayFunc(Display);
    glutReshapeFunc(Reshape);
    atexit(Terminate);  // Called after glutMainLoop ends

	// Init GL and shaders
	Init();

	////////////////////////////////
	// 
	//  Code for snap each frame
	//
	////////////////////////////////

	//int window_size = 1024;

	//CGLFBO fbo;
	//fbo.Init(window_size, window_size);
	//fbo.CheckFBOErr();

	//PFMImage result(window_size, window_size, 1, NULL);
	//float * data = result.GetPixelDataBuffer();;

	//for (int i_camera = 0; i_camera < 8; ++i_camera)
	//{
	//	for (int i_frame = 2906; i_frame <= 3061; ++i_frame)
	//	{
	//		fprintf(stderr, "Camera %d Frame %d \n", i_camera, i_frame);

	//		fbo.BeginDraw2FBO();

	//		Snap(i_camera, i_frame);

	//		fbo.EndDraw2FBO();

	//		float * fdata = fbo.ReadPixels();

	//		for (int v = 0; v < window_size; ++v)
	//		{
	//			for (int u = 0; u < window_size; ++u)
	//			{
	//				for (int c = 0; c < 3; ++c)
	//				{
	//					data[(v*window_size+u)*3+c] = fdata[(v*window_size+u)*4+c];
	//				}
	//			}
	//		}

	//		char buf[128];
	//		sprintf(buf, "../Data/RenderResults/Camera%d/Frame%d.pfm", i_camera, i_frame);
	//		result.WriteImage(buf);
	//	}
	//}
	//return 0;


	// Set ATB UIs
	SetTweakBar();

	// Call the GLUT main loop
	glutMainLoop();

	return 0;
}