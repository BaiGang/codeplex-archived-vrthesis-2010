// 3DSLoaderView.cpp : implementation of the CPmModelTool class
//

//#include "stdafx.h"
//#include "3DSLoader.h"
//
//#include "3DSLoaderDoc.h"
//#include "3DSLoaderView.h"
#include "stdafx.h"
// SHARED_HANDLERS 可以在实现预览、缩略图和搜索筛选器句柄的
// ATL 项目中进行定义，并允许与该项目共享文档代码。
#ifndef SHARED_HANDLERS
#include "test.h"
#endif

#include "testDoc.h"
#include "pmModelTool.h"
#include "MainFrm.h"

#include "math/SimpleQuat.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CPmModelTool

float g_CenterX = 0.0f;
float g_CenterY = 0.0f;
float g_CenterZ = 0.0f;

IMPLEMENT_DYNCREATE(CPmModelTool, CView)

BEGIN_MESSAGE_MAP(CPmModelTool, CView)
	//{{AFX_MSG_MAP(CPmModelTool)
	ON_WM_CREATE()
	ON_WM_DESTROY()
	ON_WM_SIZE()
	ON_WM_TIMER()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_MOUSEMOVE()
	ON_WM_RBUTTONDOWN()
	ON_WM_RBUTTONUP()
	ON_COMMAND(ID_FILE_SAVE, OnFileSave)
	//}}AFX_MSG_MAP
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, CPmModelTool::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, CPmModelTool::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, CPmModelTool::OnFilePrintPreview)
	ON_BN_CLICKED(IDC_RADIO_CAMERA, &CPmModelTool::OnBnClickedRadioCamera)
	ON_BN_CLICKED(IDC_RADIO_LIGHT, &CPmModelTool::OnBnClickedRadioLight)
	ON_COMMAND(ID_LAST, &CPmModelTool::OnBnClickedLast)
	ON_COMMAND(ID_NEXT, &CPmModelTool::OnBnClickedNext)
	ON_COMMAND(ID_BUTTON_CAMERA0, &CPmModelTool::OnButtonCamera0)
	ON_COMMAND(ID_BUTTON_CAMERA3, &CPmModelTool::OnButtonCamera3)
	ON_COMMAND(ID_BUTTON_CAMERA1, &CPmModelTool::OnButtonCamera1)
	ON_COMMAND(ID_BUTTON_CAMERA2, &CPmModelTool::OnButtonCamera2)
	ON_COMMAND(ID_BUTTON_CAMERA4, &CPmModelTool::OnButtonCamera4)
	ON_COMMAND(ID_BUTTON_CAMERA5, &CPmModelTool::OnButtonCamera5)
	ON_COMMAND(ID_BUTTON_CAMERA6, &CPmModelTool::OnButtonCamera6)
	ON_COMMAND(ID_BUTTON_CAMERA7, &CPmModelTool::OnButtonCamera7)
	ON_COMMAND(ID_VOL_SLIDER1, &CPmModelTool::OnSlider1)
	ON_COMMAND(ID_VOL_SLIDER2, &CPmModelTool::OnSlider2)
	ON_COMMAND(ID_VOL_SLIDER3, &CPmModelTool::OnSlider3)
	ON_COMMAND(ID_VOL_EDIT1, &CPmModelTool::OnVolEdit1)
	ON_COMMAND(ID_VOL_EDIT2, &CPmModelTool::OnVolEdit2)
	ON_COMMAND(ID_VOL_EDIT3, &CPmModelTool::OnVolEdit3)
	ON_COMMAND(ID_BUTTON_START, &CPmModelTool::OnButtonStart)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CPmModelTool construction/destruction

CPmModelTool::CPmModelTool()
{
	// TODO: add construction code here

}

CPmModelTool::~CPmModelTool()
{
}

BOOL CPmModelTool::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs
////////////////////////////////////////////////////////////////
//设置窗口类型
	cs.style |=WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
////////////////////////////////////////////////////////////////
	return CView::PreCreateWindow(cs);
}

/////////////////////////////////////////////////////////////////////////////
// CPmModelTool drawing

void CPmModelTool::OnDraw(CDC* pDC)
{
	CtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	// TODO: add draw code for native data here
//////////////////////////////////////////////////////////////////
	RenderScene();	//渲染场景
//////////////////////////////////////////////////////////////////

}

/////////////////////////////////////////////////////////////////////////////
// CPmModelTool printing

BOOL CPmModelTool::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CPmModelTool::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CPmModelTool::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}

/////////////////////////////////////////////////////////////////////////////
// CPmModelTool diagnostics

#ifdef _DEBUG
void CPmModelTool::AssertValid() const
{
	CView::AssertValid();
}

void CPmModelTool::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CtestDoc* CPmModelTool::GetDocument() // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CtestDoc)));
	return (CtestDoc*)m_pDocument;
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CPmModelTool message handlers

int CPmModelTool::OnCreate(LPCREATESTRUCT lpCreateStruct) 
{
	if (CView::OnCreate(lpCreateStruct) == -1)
		return -1;
	
	// TODO: Add your specialized creation code here
//////////////////////////////////////////////////////////////////
//初始化OpenGL和设置定时器
	m_pDC = new CClientDC(this);
	SetTimer(1, 20, NULL);
	InitializeOpenGL(m_pDC);
//////////////////////////////////////////////////////////////////
	return 0;
}

void CPmModelTool::OnDestroy() 
{
	CView::OnDestroy();
	
	// TODO: Add your message handler code here
/////////////////////////////////////////////////////////////////
//删除调色板和渲染上下文、定时器
	::wglMakeCurrent(0,0);
	::wglDeleteContext( m_hRC);
	if (m_hPalette)
	    DeleteObject(m_hPalette);
	if ( m_pDC )
	{
		delete m_pDC;
	}
	KillTimer(1);		
/////////////////////////////////////////////////////////////////
	
}

void CPmModelTool::OnSize(UINT nType, int cx, int cy) 
{
	CView::OnSize(nType, cx, cy);
	
	// TODO: Add your message handler code here
/////////////////////////////////////////////////////////////////
//添加窗口缩放时的图形变换函数
	glViewport(0,0,cx,cy);
/////////////////////////////////////////////////////////////////
	GLdouble aspect_ratio;
	aspect_ratio = (GLdouble)cx/(GLdouble)cy;
	::glMatrixMode(GL_PROJECTION);
	::glLoadIdentity();
	gluPerspective(40.0F, aspect_ratio, 1.0F, 1000.0F);
	::glMatrixMode(GL_MODELVIEW);
	//::glLoadIdentity();

	setExePos();

}

void CPmModelTool::OnTimer(UINT nIDEvent) 
{
	// TODO: Add your message handler code here and/or call default
/////////////////////////////////////////////////////////////////
//添加定时器响应函数和场景更新函数
	Invalidate(FALSE);	
/////////////////////////////////////////////////////////////////
	
	CView::OnTimer(nIDEvent);
}

/////////////////////////////////////////////////////////////////////
//	                  设置逻辑调色板
//////////////////////////////////////////////////////////////////////
void CPmModelTool::SetLogicalPalette(void)
{
    struct
    {
        WORD Version;
        WORD NumberOfEntries;
        PALETTEENTRY aEntries[256];
    } logicalPalette = { 0x300, 256 };

	BYTE reds[] = {0, 36, 72, 109, 145, 182, 218, 255};
	BYTE greens[] = {0, 36, 72, 109, 145, 182, 218, 255};
	BYTE blues[] = {0, 85, 170, 255};

    for (int colorNum=0; colorNum<256; ++colorNum)
    {
        logicalPalette.aEntries[colorNum].peRed =
            reds[colorNum & 0x07];
        logicalPalette.aEntries[colorNum].peGreen =
            greens[(colorNum >> 0x03) & 0x07];
        logicalPalette.aEntries[colorNum].peBlue =
            blues[(colorNum >> 0x06) & 0x03];
        logicalPalette.aEntries[colorNum].peFlags = 0;
    }

    m_hPalette = CreatePalette ((LOGPALETTE*)&logicalPalette);
}


//////////////////////////////////////////////////////////
//						初始化openGL场景
//////////////////////////////////////////////////////////
BOOL CPmModelTool::InitializeOpenGL(CDC* pDC)
{
	m_pDC = pDC;
	SetupPixelFormat();
	//生成绘制描述表
	m_hRC = ::wglCreateContext(m_pDC->GetSafeHdc());
	//置当前绘制描述表
	::wglMakeCurrent(m_pDC->GetSafeHdc(), m_hRC);

	InitGL();
	Init();

	GLenum err = glGetError();
	
	if (err != GL_NO_ERROR)
	{
		char buf[256];
		sprintf(buf, "%s", glewGetErrorString(err));
		
		AfxMessageBox(char2cstring(buf));
		return false;
	}

	return TRUE;
}

//////////////////////////////////////////////////////////
//						设置像素格式
//////////////////////////////////////////////////////////
BOOL CPmModelTool::SetupPixelFormat()
{
	PIXELFORMATDESCRIPTOR pfd = { 
	    sizeof(PIXELFORMATDESCRIPTOR),    // pfd结构的大小 
	    1,                                // 版本号 
	    PFD_DRAW_TO_WINDOW |              // 支持在窗口中绘图 
	    PFD_SUPPORT_OPENGL |              // 支持 OpenGL 
	    PFD_DOUBLEBUFFER,                 // 双缓存模式 
	    PFD_TYPE_RGBA,                    // RGBA 颜色模式 
	    24,                               // 24 位颜色深度 
	    0, 0, 0, 0, 0, 0,                 // 忽略颜色位 
	    0,                                // 没有非透明度缓存 
	    0,                                // 忽略移位位 
	    0,                                // 无累加缓存 
	    0, 0, 0, 0,                       // 忽略累加位 
	    32,                               // 32 位深度缓存     
	    0,                                // 无模板缓存 
	    0,                                // 无辅助缓存 
	    PFD_MAIN_PLANE,                   // 主层 
	    0,                                // 保留 
	    0, 0, 0                           // 忽略层,可见性和损毁掩模 
	}; 	
	int pixelformat;
	pixelformat = ::ChoosePixelFormat(m_pDC->GetSafeHdc(), &pfd);//选择像素格式
	::SetPixelFormat(m_pDC->GetSafeHdc(), pixelformat, &pfd);	//设置像素格式
	if(pfd.dwFlags & PFD_NEED_PALETTE)
		SetLogicalPalette();	//设置逻辑调色板
	return TRUE;
}



//////////////////////////////////////////////////////////
//						场景绘制与渲染
//////////////////////////////////////////////////////////
BOOL CPmModelTool::RenderScene() 
{
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Rotate & Draw
	glPushMatrix();

	//
	float axisX[3] = {1.0f, 0.0f, 0.0f};
	float axisY[3] = {0.0f, 1.0f, 0.0f};
	float axisZ[3] = {0.0f, 0.0f, 1.0f};
	float quatRZ[4];
	float quatRX[4];
	float quatRY[4];
	float quatTmp[4];
	float quatRotate[4];
	float matRotate[16];

	////////// get quaternion for x and y rotation
	////////SetQuaternionFromAxisAngle(axisY, static_cast<float>(m_mouse_x)*0.0001f, quatRX);
	////////SetQuaternionFromAxisAngle(axisX, static_cast<float>(m_mouse_y)*0.0001f, quatRY);
	////////// combine the rotation
	////////MultiplyQuaternions(m_quatRotate_start, quatRX, quatTmp);
	////////MultiplyQuaternions(quatTmp, quatRY, quatRotate);
	////////// get mv mat from quat
	////////ConvertQuaternionToMatrix(quatRotate, matRotate);

	if (m_mouse_x < 0.0)
	{
		SetQuaternionFromAxisAngle(axisZ, -0.03, quatRZ);
	}
	else
	{
		SetQuaternionFromAxisAngle(axisZ, 0.03, quatRZ);
	}
	if (m_mouse_y < 0.0)
	{
		SetQuaternionFromAxisAngle(axisY, -0.03, quatRY);
	}
	else
	{
		SetQuaternionFromAxisAngle(axisY, 0.03, quatRY);
	}
	// RX = RY * RZ
	MultiplyQuaternions(quatRY, quatRZ, quatRX);
	MultiplyQuaternions(m_quatRotate_start, quatRX, m_quatRotate_update);
	ConvertQuaternionToMatrix(m_quatRotate_update, matRotate);

	glMatrixMode(GL_MODELVIEW);
	glMultMatrixf(matRotate);

	DrawSmoke();
	DrawAxis();
	DrawBox();
		
	glPopMatrix();
	
	SwapBuffers(m_pDC->GetSafeHdc());		//交互缓冲区
	return TRUE;
}

//////////////////////////////////////////////////////////
//							DrawAxis()
//////////////////////////////////////////////////////////
void CPmModelTool::DrawAxis()
{
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0,0,0); glVertex3f(10, 0, 0);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0,0,0); glVertex3f(0, 10, 0);
	glColor3f(0.0f, 0.0f, 1.0f);
	glVertex3f(0,0,0); glVertex3f(0, 0, 10);
	glEnd();

////	::glMatrixMode(GL_MODELVIEW);
////	::glLoadIdentity();
//	glPushMatrix();
//	glScalef(50000.0,50000.0,50000.0);
//	glBegin( GL_LINES);
//			// x轴
//			glColor3f(1.0F, 0.0F, 0.0F);
//			glVertex3f(-3.0f, 0.0f, 0.0f);
//			glVertex3f( 3.0f, 0.0f, 0.0f);
//			glVertex3f( 2.5f, 0.5f, 0.0f);
//			glVertex3f( 3.0f, 0.0f, 0.0f);
//			glVertex3f( 2.5f,-0.5f,-0.0f);
//			glVertex3f( 3.0f, 0.0f, 0.0f);
//			
//			// y轴
//			glColor3f(0.0F, 1.0F, 0.0F);
//			glVertex3f( 0.0f, -3.0f, 0.0f);
//			glVertex3f( 0.0f,  3.0f, 0.0f);
//			glVertex3f(-0.5f,  2.5f, 0.0f);
//			glVertex3f( 0.0f,  3.0f, 0.0f);
//			glVertex3f( 0.5f,  2.5f, 0.0f);
//			glVertex3f( 0.0f,  3.0f, 0.0f);
//
//			// z轴
//			glColor3f(0.0F, 0.0F, 1.0F);
//			glVertex3f( 0.0f, 0.0f, -3.0f);
//			glVertex3f( 0.0f, 0.0f,  3.0f);
//			glVertex3f(-0.5f, 0.0f,  2.5f);
//			glVertex3f( 0.0f, 0.0f,  3.0f);
//			glVertex3f( 0.5f, 0.0f,  2.5f);
//			glVertex3f( 0.0f, 0.0f,  3.0f);
//	glEnd();
//	glPopMatrix();
}

//////////////////////////////////////////////////////////
//							Draw3ds()
//////////////////////////////////////////////////////////
void CPmModelTool::Draw3ds()
{
	/*if (m_3dsLoaded) 
	{
			m_triList.drawGL();
	}*/
}

void CPmModelTool::Init(GLvoid)
{

//	m_3dsLoaded  = FALSE;
//
	camPos[0]	 = 0.0f;
	camPos[1]	 = 0.0f;
	camPos[2]	 = -100.0f;
	camRot[0]	 = 20.0f;
	camRot[1]	 = -20.0f;
	camRot[2]	 = 0.0f;

	scenePos[0]	 = 0.0f;
	scenePos[1]	 = 0.0f;
	scenePos[2]	 = 0.0f;
	sceneRot[0]	 = 0.0f;
	sceneRot[1]	 = 0.0f;
	sceneRot[2]	 = 0.0f;
	mouseprevpoint.x = 0;
	mouseprevpoint.y = 0;
	mouserightdown = FALSE;
	mouseleftdown = FALSE;
//
//
////	m_triList.Init();
//	
//	::glShadeModel(GL_FLAT);
//	
//	::glClearColor(0.0F, 0.0F, 0.0F, 0.0F);
//	
//	::glClearDepth(1.0F);
//
//	::glEnable(GL_DEPTH_TEST);
//
//	::glEnable(GL_CULL_FACE);
//
//	//Change Light by  cxf
//	GLfloat ambientLight[] = { 0.23f, 0.23f, 0.23f, 1.0f};
//	GLfloat diffuseLight[] = { 0.027f, 0.027f, 0.027f, 1.0f};
//	GLfloat lightPos[]     = {0.0f,3000.0f,0.0f, 1.0f};
//
//	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
//	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
//	glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
//	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientLight);
//	
//	glEnable(GL_COLOR_MATERIAL);
//	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
//	glEnable(GL_LIGHTING);
//	glEnable(GL_LIGHT0);

}

void CPmModelTool::OnRButtonUp(UINT nFlags, CPoint point) 
{
	// TODO: Add your message handler code here and/or call default
	ReleaseCapture( );
	mouserightdown = FALSE;
//	SetCamPos(2, (point.y - mouseprevpoint.y) , TRUE, TRUE);
	
	CView::OnRButtonUp(nFlags, point);
}

void CPmModelTool::OnRButtonDown(UINT nFlags, CPoint point) 
{
	// TODO: Add your message handler code here and/or call default
	SetCapture( );
	mouserightdown = TRUE;
	mouseprevpoint.x = point.x;
	mouseprevpoint.y = point.y;
	
	CView::OnRButtonDown(nFlags, point);

}

void CPmModelTool::OnMouseMove(UINT nFlags, CPoint point) 
{
	// TODO: Add your message handler code here and/or call default
	
	//if(mouserightdown)
	//{
	//	SetCamPos(2, -(point.y - mouseprevpoint.y) , TRUE,TRUE);
	//}
	//else if(mouseleftdown)
	//{	
	//	SetSceneRot(0, (point.y - mouseprevpoint.y) , TRUE,TRUE);
	//	SetSceneRot(2, (point.x - mouseprevpoint.x) , TRUE,TRUE);
	//}
	//CView::OnMouseMove(nFlags, point);

	//mouseprevpoint.x = point.x;
	//mouseprevpoint.y = point.y;	

	// TODO: 在此添加消息处理程序代码和/或调用默认值

	CPoint mouseDiff = point - m_mouseLast;

	//左键按下
	if(nFlags & MK_LBUTTON)
	{
		m_mouse_x = mouseDiff.x;
		m_mouse_x /= 50;
		m_mouse_y = mouseDiff.y;
		m_mouse_y /= 50;

		m_quatRotate_start[0] = m_quatRotate_update[0];
		m_quatRotate_start[1] = m_quatRotate_update[1];
		m_quatRotate_start[2] = m_quatRotate_update[2];
		m_quatRotate_start[3] = m_quatRotate_update[3];

		RenderScene();
	}

	if(nFlags & MK_MBUTTON)
	{
		m_mouse_x = mouseDiff.x;
		m_mouse_x /= 50;
		m_mouse_y = mouseDiff.y;
		m_mouse_y /= 1000;

		if (m_mouse_y > 0.0)
		{
			glScalef(0.95, 0.95, 0.95);
		}
		else
		{
			glScalef(1.05, 1.05, 1.05);
		}

		RenderScene();
		//SetCamPos(2, m_mouse_y , TRUE, TRUE);
	}

	if(nFlags & MK_RBUTTON)
	{
		m_mouse_x = mouseDiff.x;
		m_mouse_x /= 100;
		m_mouse_y = mouseDiff.y;
		m_mouse_y /= 100;

		glTranslatef(-m_mouse_x, -m_mouse_y, 0.0f);

		RenderScene();
		//SetCamPos(2, m_mouse_y , TRUE, TRUE);
	}

	m_mouseLast = point;

	CView::OnMouseMove(nFlags, point);

}

void CPmModelTool::OnLButtonUp(UINT nFlags, CPoint point) 
{
	// TODO: Add your message handler code here and/or call default
	ReleaseCapture( );
	mouseleftdown = FALSE;

	//SetSceneRot(0, (point.y - mouseprevpoint.y) , TRUE, TRUE);
	//SetSceneRot(2, (point.x - mouseprevpoint.x) , TRUE, TRUE);
	
	CView::OnLButtonUp(nFlags, point);
}

void CPmModelTool::OnLButtonDown(UINT nFlags, CPoint point) 
{
	// TODO: Add your message handler code here and/or call default
	SetCapture( );
	mouseleftdown = TRUE;
	mouseprevpoint.x = point.x;
	mouseprevpoint.y = point.y;	
	CView::OnLButtonDown(nFlags, point);
}

void CPmModelTool::SetCamPos(int axis, int value, BOOL increment, BOOL apply)
{
	if(increment)
	{
		camPos[axis] += (float)value*camPos[axis]/100;
	}
	else
	{
		camPos[axis] = (float)value/2;
	}

	::glMatrixMode(GL_MODELVIEW);
	::glLoadIdentity();

	RenderScene();

}

void CPmModelTool::SetSceneRot(int axis, int value, BOOL increment, BOOL apply)
{
	if(increment)
		sceneRot[axis] += (sceneRot[axis] >=360) ? (-360 + value/2): value/2;
	else
		sceneRot[axis] = (sceneRot[axis] >=360) ? (-360 + value/2): value/2;
	
	RenderScene();
}

BOOL CPmModelTool::OpenFile(LPCTSTR lpszPathName)
{
	//char* file = new char[strlen(lpszPathName)];
	//strcpy(file, lpszPathName);	

	//C3dsReader Loader;
	//BOOL result;
	//if( m_triList.getNumObjects() > 0 ) //如果文档已经有了模型，先去掉所有
	//									//如果是新打开的文档则，m_triList.getNumObjects()为0;	
	//	m_triList.removeAllObjects();
	//
	//result = Loader.Reader(file, &m_triList);
	//if( result) 
	//{
	//	m_3dsLoaded = TRUE;
	//	m_triList.doAfterMath();
	//}
	
//	return result;
	return TRUE;
}

void CPmModelTool::OnFileSave() 
{
	//CString s;
	//s.LoadString(IDS_FD_3DS);
	//CFileDialog fd(FALSE,"3ds",0,OFN_OVERWRITEPROMPT|OFN_HIDEREADONLY|OFN_PATHMUSTEXIST,s);

	//if(fd.DoModal()==IDOK)
	//{
	//	BeginWaitCursor();
	//	if (exp.export((char *)(LPCSTR)fd.GetPathName())==0)
	//		AfxMessageBox(IDS_ERRORSAVING,MB_OK|MB_ICONEXCLAMATION);
	//	EndWaitCursor();
	//}	
}

void CPmModelTool::OnBnClickedLast()
{
	if(itemHasLast(picIndex))
	{
		picIndex -= 1;
		loadPic();
	}
	else
		AfxMessageBox(_T("It has been the first Picture!"));
}

void CPmModelTool::OnBnClickedNext()
{
	if(itemHasNext(picIndex))
	{
		picIndex += 1;
		loadPic();
	}
	else
		AfxMessageBox(_T("It has been the last Picture!"));
}

void CPmModelTool::OnBnClickedRadioCamera()
{
	// TODO: 在此添加控件通知处理程序代码
	currentRBState = CAMERA;
	((CMainFrame*)theApp.m_pMainWnd)->setRadio(true);
	Invalidate();
}


void CPmModelTool::OnBnClickedRadioLight()
{
	// TODO: 在此添加控件通知处理程序代码
	currentRBState = LIGHT;
	((CMainFrame*)theApp.m_pMainWnd)->setRadio(false);
	Invalidate();
}

void CPmModelTool::loadPic()
{
	// TODO: 在此添加命令处理程序代码
	CString filestr;

	if(itemHasLast(picIndex))
	{
		filestr.Format(_T("E:/bg data/asmodeling/Data/Results/Frame%08d.pfm"), 2906+picIndex-1);
		m_pLastPic->ReadImage(cstring2char(filestr));
	}

	filestr.Format(_T("E:/bg data/asmodeling/Data/Results/Frame%08d.pfm"), 2906+picIndex);
	m_pCurrentPic->ReadImage(cstring2char(filestr));

	if(itemHasNext(picIndex))
	{
		filestr.Format(_T("E:/bg data/asmodeling/Data/Results/Frame%08d.pfm"), 2906+picIndex+1);
		m_pNextPic->ReadImage(cstring2char(filestr));
	}
}

char* CPmModelTool::cstring2char (CString cstr)
{
	int len = WideCharToMultiByte(CP_UTF8, 0, cstr.AllocSysString(), -1, NULL, 0, NULL, NULL);  
	char * szUtf8=new char[len + 1];
	WideCharToMultiByte (CP_UTF8, 0, cstr.AllocSysString(), -1, szUtf8, len, NULL,NULL);
	return szUtf8;
}

CString CPmModelTool::char2cstring(char* cstr)
{	
	CString str;
	str = cstr;
	return str;
}

bool CPmModelTool::itemHasNext(int index)
{
	if (index < 2096+ getFileCount(theApp.rootPath+_T("E:/bg data/asmodeling/Data/Results")))
		return true;
	else 
		return false;
}

bool CPmModelTool::itemHasLast(int index)
{
	if (index >= 2096+getFileCount(theApp.rootPath+_T("E:/bg data/asmodeling/Data/Results")))
		return true;
	else 
		return false;
}

int CPmModelTool::getFileCount(CString csFolderName)
{   
	int   i=0;   
	CFileFind   f; 
	BOOL   bFind=f.FindFile(csFolderName+ "\\*.* "); 
	while(bFind) 
	{   
		bFind   =   f.FindNextFile(); 
		i++; 
	} 
	return   i; 
}

void CPmModelTool::outputText(char* str)
{
	((CMainFrame*)theApp.m_pMainWnd)->outputText(char2cstring(str));
}

bool CPmModelTool::InitGL(void)
{
	// init glew
	if (!InitGLExtensions())
	{
		AfxMessageBox(_T("InitGLExtensions failed!"));
		return false;
	}

	// init shaders
	m_shader_along_x.InitShaders("../Data/RayMarchingBlend.vert","../Data/RayMarchingBlendXU.frag");
	m_shader_along_y.InitShaders("../Data/RayMarchingBlend.vert","../Data/RayMarchingBlendYU.frag");
	m_shader_along_z.InitShaders("../Data/RayMarchingBlend.vert","../Data/RayMarchingBlendZU.frag");

	// init volume texture
	glGenTextures(1, &m_tex3d_id);
	glBindTexture(GL_TEXTURE_3D, m_tex3d_id);

	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// load camera parameters
	// and convert to mv & prj matrices
	LoadCameras();

	// init images
	m_pLastPic     = new PFMImage(BOX_LENGTH, BOX_LENGTH*BOX_LENGTH, 0, NULL);
	m_pCurrentPic  = new PFMImage(BOX_LENGTH, BOX_LENGTH*BOX_LENGTH, 0, NULL);
	m_pNextPic     = new PFMImage(BOX_LENGTH, BOX_LENGTH*BOX_LENGTH, 0, NULL);

	// TODO : path specification...
	m_pCurrentPic->ReadImage("E:/bg data/asmodeling/Data/Results/Frame00002906.pfm");
	
	PixelFormat pf;
	pf.r = pf.g = pf.b = pf.alpha = 1.0f;
	for (int i = 0; i < BOX_LENGTH; ++i)
	{
		for (int j = 0; j < BOX_LENGTH * BOX_LENGTH; ++j)
		{
			m_pCurrentPic->SetPixel(i, j, pf);
		}
	}
	glBindTexture(GL_TEXTURE_3D, m_tex3d_id);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB, BOX_LENGTH, BOX_LENGTH, BOX_LENGTH,
		0, GL_LUMINANCE, GL_FLOAT, m_pCurrentPic->GetPixelDataBuffer());

	m_LightMultiplier = 50000000.0f;
	m_LightDist = 195.68271;
	m_LightPosition[0] = 34.51900f  / m_LightDist;
	m_LightPosition[0] = -135.97600 / m_LightDist;
	m_LightPosition[0] = 136.42100  / m_LightDist;
	m_LightPosition[0] = 1.0f;


	// set rotate start quat
	m_quatRotate_start[0] = 0.0f;
	m_quatRotate_start[1] = 0.0f;
	m_quatRotate_start[2] = 0.0f;
	m_quatRotate_start[3] = 1.0f;

	m_quatRotate_update[0] = 0.0f;
	m_quatRotate_update[1] = 0.0f;
	m_quatRotate_update[2] = 0.0f;
	m_quatRotate_update[3] = 1.0f;

	m_extinction = 0.1f;
	m_scattering = 0.05f;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(40,1, 1, 1000); 

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(30, 40, 0, 0, 0, 0, 0, 0, 1);

	return true;
}

bool CPmModelTool::ReshapeGL(int width, int height)
{
	// view port
	glViewport(0, 0, width, height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(40, (double)width/height, 1, 100);

	// modelview
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0f,0.0f,4.0f, 0,0,0, 0,1.0f,0.0f);

	return true;
}


bool CPmModelTool::DisplayGL(void)
{
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Rotate & Draw
	glPushMatrix();

	//
	float axisX[3] = {1.0f, 0.0f, 0.0f};
	float axisY[3] = {0.0f, 1.0f, 0.0f};
	float quatRX[4];
	float quatRY[4];
	float quatTmp[4];
	float quatRotate[4];
	float matRotate[16];

	// get quaternion for x and y rotation
	SetQuaternionFromAxisAngle(axisY, static_cast<float>(m_mouse_x), quatRX);
	SetQuaternionFromAxisAngle(axisX, static_cast<float>(m_mouse_y), quatRY);
	// combine the rotation
	MultiplyQuaternions(m_quatRotate_start, quatRX, quatTmp);
	MultiplyQuaternions(quatTmp, quatRY, quatRotate);
	// get mv mat from quat
	ConvertQuaternionToMatrix(quatRotate, matRotate);
	glMatrixMode(GL_MODELVIEW);
	glMultMatrixf(matRotate);

	// Draw Delegate Box
	DrawBox();

	// Draw Smoke
	DrawSmoke();

	glPopMatrix();

	return true;
}

bool CPmModelTool::LoadCameras(void)
{
	FILE * fp = fopen("../Data/camera.txt", "r");
	if (NULL == fp)
	{
		fprintf(stderr, "Open camera file error. \n");
		return false;
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
					return false;
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
				return false;
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
					return false;
				}
			} // col
		} // row

		// set 
		Matrix4 trans;
		trans.identityMat();
		trans(1,1) = -1;
		trans(2,2) = -1;
		m_modelview_mats[i] = trans * Extr;

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
		m_projection_mats[i].SetMatrix(proj);

	} // for i

	return true;
}

// Draw Delegate Box
void CPmModelTool::DrawBox()
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

// Draw Smoke
void CPmModelTool::DrawSmoke()
{
	// first determine the camera orientation
	int orn = GetOrientation();

	//char orn_buf[36];
	//sprintf(orn_buf, "Orientation : %d\n", orn);
	//AfxMessageBox(char2cstring(orn_buf));

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

void CPmModelTool::SetCamera(int i)
{
	char tmp[128];
	sprintf(tmp, "Set camera %d.\n", i);
	outputText(tmp);
	
	//CRect rect;
	//GetWindowRect(&rect);
	//glViewport(0,0,rect.Width(),rect.Height());
	//glViewport(0,0,1024,1024);

	float mv[16];
	m_projection_mats[i].GetData(mv);
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(mv);

	m_modelview_mats[i].GetData(mv);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(mv);

	// update box position
	g_CenterX = 15.0f;
	g_CenterY = -9.0f;
	g_CenterZ = 12.0f;

	/////////////////////
	// for debugging

}
void CPmModelTool::OnButtonCamera0()
{
	// TODO: 在此添加命令处理程序代码
	SetCamera(0);
}


void CPmModelTool::OnButtonCamera3()
{
	// TODO: 在此添加命令处理程序代码
	SetCamera(3);
}


void CPmModelTool::OnButtonCamera1()
{
	// TODO: 在此添加命令处理程序代码
	SetCamera(1);
}


void CPmModelTool::OnButtonCamera2()
{
	// TODO: 在此添加命令处理程序代码
	SetCamera(2);
}


void CPmModelTool::OnButtonCamera4()
{
	// TODO: 在此添加命令处理程序代码
	SetCamera(4);
}


void CPmModelTool::OnButtonCamera5()
{
	// TODO: 在此添加命令处理程序代码
	SetCamera(5);
}


void CPmModelTool::OnButtonCamera6()
{
	// TODO: 在此添加命令处理程序代码
	SetCamera(6);
}


void CPmModelTool::OnButtonCamera7()
{
	// TODO: 在此添加命令处理程序代码
	SetCamera(7);
}


void CPmModelTool::OnSlider1()
{
	// TODO: 在此添加命令处理程序代码
	int n;
	CString str;
	CMainFrame *pMain=(CMainFrame *)AfxGetApp()->m_pMainWnd;
	CMFCRibbonSlider* pSlider = DYNAMIC_DOWNCAST(CMFCRibbonSlider, pMain->m_wndRibbonBar.FindByID(ID_VOL_SLIDER1));
	n = pSlider->GetPos();
	str.Format(_T("%d"), n);
	CMFCRibbonRichEditCtrl *pEdit = (CMFCRibbonRichEditCtrl *)(pMain->m_wndRibbonBar.GetDlgItem(ID_VOL_EDIT1));
	pEdit->SetWindowTextW(str);

	// update intensity
	m_LightMultiplier = ((float)n);
}


void CPmModelTool::OnSlider2()
{
	// TODO: 在此添加命令处理程序代码
	int n;
	CString str;
	CMainFrame *pMain=(CMainFrame *)AfxGetApp()->m_pMainWnd;
	CMFCRibbonSlider* pSlider = DYNAMIC_DOWNCAST(CMFCRibbonSlider, pMain->m_wndRibbonBar.FindByID(ID_VOL_SLIDER2));
	n = pSlider->GetPos();
	str.Format(_T("%.2f"), ((float)n)/100);
	CMFCRibbonRichEditCtrl *pEdit = (CMFCRibbonRichEditCtrl *)(pMain->m_wndRibbonBar.GetDlgItem(ID_VOL_EDIT2));
	pEdit->SetWindowTextW(str);

	// update scattering
	m_scattering = (float)n / 100.0f;
}


void CPmModelTool::OnSlider3()
{
	// TODO: 在此添加命令处理程序代码
	int n;
	CString str;
	CMainFrame *pMain=(CMainFrame *)AfxGetApp()->m_pMainWnd;
	CMFCRibbonSlider* pSlider = DYNAMIC_DOWNCAST(CMFCRibbonSlider, pMain->m_wndRibbonBar.FindByID(ID_VOL_SLIDER3));
	n = pSlider->GetPos();
	str.Format(_T("%.2f"), ((float)n)/100);
	CMFCRibbonRichEditCtrl *pEdit = (CMFCRibbonRichEditCtrl *)(pMain->m_wndRibbonBar.GetDlgItem(ID_VOL_EDIT3));
	pEdit->SetWindowTextW(str);

	// update absorption
	m_extinction = (float)n / 100.0f;

}


bool CPmModelTool::setVolume(int id, int n)
{
	int max,min;
	CString str;
	CMainFrame *pMain=(CMainFrame *)AfxGetApp()->m_pMainWnd;
	CMFCRibbonSlider* p = DYNAMIC_DOWNCAST(CMFCRibbonSlider, pMain->m_wndRibbonBar.FindByID(id));
	max = p->GetRangeMax();
	min = p->GetRangeMin();

	if (n>max || n<min)
	{
		str.Format(_T("%d is Out of Range (%d, %d)"), n, min, max);
		AfxMessageBox(str);
		return false;
	}



	p->SetPos(n);
	return true;
}


void CPmModelTool::OnVolEdit1()
{
	// TODO: 在此添加命令处理程序代码
	CMainFrame *pMain=(CMainFrame *)AfxGetApp()->m_pMainWnd;
	CMFCRibbonRichEditCtrl *p = (CMFCRibbonRichEditCtrl *)(pMain->m_wndRibbonBar.GetDlgItem(ID_VOL_EDIT1));
	CString str;
	int n;
	p->GetWindowTextW(str);
	n = atoi(cstring2char(str)) ;
	setVolume(ID_VOL_SLIDER1, n);
}


void CPmModelTool::OnVolEdit2()
{
	// TODO: 在此添加命令处理程序代码
	CMainFrame *pMain=(CMainFrame *)AfxGetApp()->m_pMainWnd;
	CMFCRibbonRichEditCtrl *p = (CMFCRibbonRichEditCtrl *)(pMain->m_wndRibbonBar.GetDlgItem(ID_VOL_EDIT2));
	CString str;
	int n;
	p->GetWindowTextW(str);
	float value = _wtof(str);
	n = value*100;
	setVolume(ID_VOL_SLIDER2, n);
}


void CPmModelTool::OnVolEdit3()
{
	// TODO: 在此添加命令处理程序代码
	CMainFrame *pMain=(CMainFrame *)AfxGetApp()->m_pMainWnd;
	CMFCRibbonRichEditCtrl *p = (CMFCRibbonRichEditCtrl *)(pMain->m_wndRibbonBar.GetDlgItem(ID_VOL_EDIT3));
	CString str;
	int n;
	p->GetWindowTextW(str);
	float value = _wtof(str);
	n = value*100;
	setVolume(ID_VOL_SLIDER3, n);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

int CPmModelTool::GetOrientation()
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
	m_CameraPos = cmrPos;

	// store the inversed modelview matrix
	mat.Inverse();
	mat.GetData(m_CameraInv);

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


void CPmModelTool::DrawSmoke_alongxN(void)
{
	m_shader_along_x.Begin();

	m_shader_along_x.SetUniform3f("lightIntensity", m_LightMultiplier, m_LightMultiplier, m_LightMultiplier);
	m_shader_along_x.SetUniform4f("lightPosWorld",
		m_LightPosition[0] * m_LightDist,
		m_LightPosition[1] * m_LightDist,
		m_LightPosition[2] * m_LightDist, 1.0f);
	m_shader_along_x.SetUniform1f("absorptionCoefficient", m_extinction);
	m_shader_along_x.SetUniform1f("scatteringCoefficient", m_scattering);
	
	float step_size = BOX_SIZE / (1.0f * BOX_LENGTH);
	
	m_shader_along_x.SetUniform1f("stepSize", step_size);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, m_tex3d_id);
	m_shader_along_x.SetUniform1i("volumeTex", 0);
	m_shader_along_x.SetUniform4f("cameraPos", 
		m_CameraPos.x, m_CameraPos.y,
		m_CameraPos.z, m_CameraPos.w);
	m_shader_along_x.SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, m_CameraInv);

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

	m_shader_along_x.End();
}

void CPmModelTool::DrawSmoke_alongyN(void)
{
	m_shader_along_y.Begin();

	m_shader_along_y.SetUniform3f("lightIntensity", m_LightMultiplier, m_LightMultiplier, m_LightMultiplier);
	m_shader_along_y.SetUniform4f("lightPosWorld",
		m_LightPosition[0] * m_LightDist,
		m_LightPosition[1] * m_LightDist,
		m_LightPosition[2] * m_LightDist, 1.0f);
	m_shader_along_y.SetUniform1f("absorptionCoefficient", m_extinction);
	m_shader_along_y.SetUniform1f("scatteringCoefficient", m_scattering);
	
	float step_size = BOX_SIZE / (1.0f * BOX_LENGTH);
	
	m_shader_along_y.SetUniform1f("stepSize", step_size);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, m_tex3d_id);
	m_shader_along_y.SetUniform1i("volumeTex", 0);
	m_shader_along_y.SetUniform4f("cameraPos", 
		m_CameraPos.x, m_CameraPos.y,
		m_CameraPos.z, m_CameraPos.w);
	m_shader_along_y.SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, m_CameraInv);

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

	m_shader_along_y.End();
}

void CPmModelTool::DrawSmoke_alongzN(void)
{
	m_shader_along_z.Begin();

	m_shader_along_z.SetUniform3f("lightIntensity", m_LightMultiplier, m_LightMultiplier, m_LightMultiplier);
	m_shader_along_z.SetUniform4f("lightPosWorld",
		m_LightPosition[0] * m_LightDist,
		m_LightPosition[1] * m_LightDist,
		m_LightPosition[2] * m_LightDist, 1.0f);
	m_shader_along_z.SetUniform1f("absorptionCoefficient", m_extinction);
	m_shader_along_z.SetUniform1f("scatteringCoefficient", m_scattering);
	float step_size = BOX_SIZE / (1.0f * BOX_LENGTH);
	m_shader_along_z.SetUniform1f("stepSize", step_size);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, m_tex3d_id);
	m_shader_along_z.SetUniform1i("volumeTex", 0);
	m_shader_along_z.SetUniform4f("cameraPos", 
		m_CameraPos.x, m_CameraPos.y,
		m_CameraPos.z, m_CameraPos.w);
	m_shader_along_z.SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, m_CameraInv);

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

	m_shader_along_z.End();
}

void CPmModelTool::DrawSmoke_alongxP(void)
{
	m_shader_along_x.Begin();

	m_shader_along_x.SetUniform3f("lightIntensity", m_LightMultiplier, m_LightMultiplier, m_LightMultiplier);
	m_shader_along_x.SetUniform4f("lightPosWorld",
		m_LightPosition[0] * m_LightDist,
		m_LightPosition[1] * m_LightDist,
		m_LightPosition[2] * m_LightDist, 1.0f);
	m_shader_along_x.SetUniform1f("absorptionCoefficient", m_extinction);
	m_shader_along_x.SetUniform1f("scatteringCoefficient", m_scattering);
	float step_size = BOX_SIZE / (1.0f*BOX_LENGTH);
	m_shader_along_x.SetUniform1f("stepSize", step_size);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, m_tex3d_id);
	m_shader_along_x.SetUniform1i("volumeTex", 0);
	m_shader_along_x.SetUniform4f("cameraPos", 
		m_CameraPos.x, m_CameraPos.y,
		m_CameraPos.z, m_CameraPos.w);
	m_shader_along_x.SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, m_CameraInv);

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

	m_shader_along_x.End();
}

void CPmModelTool::DrawSmoke_alongyP(void)
{
	m_shader_along_y.Begin();

	m_shader_along_y.SetUniform3f("lightIntensity", m_LightMultiplier, m_LightMultiplier, m_LightMultiplier);
	m_shader_along_y.SetUniform4f("lightPosWorld",
		m_LightPosition[0] * m_LightDist,
		m_LightPosition[1] * m_LightDist,
		m_LightPosition[2] * m_LightDist, 1.0f);
	m_shader_along_y.SetUniform1f("absorptionCoefficient", m_extinction);
	m_shader_along_y.SetUniform1f("scatteringCoefficient", m_scattering);
	float step_size = BOX_SIZE / (1.0f*BOX_LENGTH);
	m_shader_along_y.SetUniform1f("stepSize", step_size);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, m_tex3d_id);
	m_shader_along_y.SetUniform1i("volumeTex", 0);
	m_shader_along_y.SetUniform4f("cameraPos", 
		m_CameraPos.x, m_CameraPos.y,
		m_CameraPos.z, m_CameraPos.w);
	m_shader_along_y.SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, m_CameraInv);

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

	m_shader_along_y.End();
}

void CPmModelTool::DrawSmoke_alongzP(void)
{
	m_shader_along_z.Begin();

	m_shader_along_z.SetUniform3f("lightIntensity", m_LightMultiplier, m_LightMultiplier, m_LightMultiplier);
	m_shader_along_z.SetUniform4f("lightPosWorld",
		m_LightPosition[0] * m_LightDist,
		m_LightPosition[1] * m_LightDist,
		m_LightPosition[2] * m_LightDist, 1.0f);
	m_shader_along_z.SetUniform1f("absorptionCoefficient", m_extinction);
	m_shader_along_z.SetUniform1f("scatteringCoefficient", m_scattering);
	float step_size = BOX_SIZE / (1.0f*BOX_LENGTH);
	m_shader_along_z.SetUniform1f("stepSize", step_size);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, m_tex3d_id);
	m_shader_along_z.SetUniform1i("volumeTex", 0);
	m_shader_along_z.SetUniform4f("cameraPos", 
		m_CameraPos.x, m_CameraPos.y,
		m_CameraPos.z, m_CameraPos.w);
	m_shader_along_z.SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, m_CameraInv);

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

	m_shader_along_z.End();
}



void CPmModelTool::OnButtonStart()
{
	// TODO: 在此添加命令处理程序代码
}

void CPmModelTool::setExePos()
{
	CRect rect;
	GetWindowRect(&rect);

	CWnd * hwnd;	
	hwnd = FindWindow(NULL, _T("Reconstructed Smoke Volume Result."));

	if(hwnd)
	{
		hwnd->MoveWindow(rect, true);
		hwnd->SetWindowPos(&CWnd::wndTopMost , rect.left,rect.top, rect.Width(), rect.Height(), SWP_NOSIZE|SWP_SHOWWINDOW );
	}
}