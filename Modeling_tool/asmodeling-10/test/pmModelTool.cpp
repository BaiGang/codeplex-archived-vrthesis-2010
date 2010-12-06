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

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CPmModelTool

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
	ON_COMMAND(ID_FILE_PRINT, CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, CView::OnFilePrintPreview)
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
	Init();	
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
	gluPerspective(40.0F, aspect_ratio, 1.0F, 10000.0F);
	::glMatrixMode(GL_MODELVIEW);
	::glLoadIdentity();

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
	::glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	::glMatrixMode(GL_MODELVIEW);
	::glLoadIdentity();

	::glTranslatef( camPos[0], camPos[1], camPos[2] );
	::glRotatef( camRot[0], 1.0F, 0.0F, 0.0F );
	::glRotatef( camRot[1], 0.0F, 1.0F, 0.0F );
	::glRotatef( camRot[2], 0.0F, 0.0F, 1.0F );
	
	::glPushMatrix();
	::glScalef(0.0005,0.0005,0.0005);//缩小模型 by cxf
	::glTranslatef(scenePos[0], scenePos[1], scenePos[2]);
	::glRotatef( sceneRot[0], 1.0F, 0.0F, 0.0F );
	::glRotatef( sceneRot[1], 0.0F, 1.0F, 0.0F );
	::glRotatef( sceneRot[2], 0.0F, 0.0F, 1.0F );

	DrawAxis();
	Draw3ds();
	
	::glPopMatrix();

	::SwapBuffers(m_pDC->GetSafeHdc());		//交互缓冲区
	return TRUE;
}

//////////////////////////////////////////////////////////
//							DrawAxis()
//////////////////////////////////////////////////////////
void CPmModelTool::DrawAxis()
{
//	::glMatrixMode(GL_MODELVIEW);
//	::glLoadIdentity();
	glPushMatrix();
	glScalef(50000.0,50000.0,50000.0);
	glBegin( GL_LINES);
			// x轴
			glColor3f(1.0F, 0.0F, 0.0F);
			glVertex3f(-3.0f, 0.0f, 0.0f);
			glVertex3f( 3.0f, 0.0f, 0.0f);
			glVertex3f( 2.5f, 0.5f, 0.0f);
			glVertex3f( 3.0f, 0.0f, 0.0f);
			glVertex3f( 2.5f,-0.5f,-0.0f);
			glVertex3f( 3.0f, 0.0f, 0.0f);
			
			// y轴
			glColor3f(0.0F, 1.0F, 0.0F);
			glVertex3f( 0.0f, -3.0f, 0.0f);
			glVertex3f( 0.0f,  3.0f, 0.0f);
			glVertex3f(-0.5f,  2.5f, 0.0f);
			glVertex3f( 0.0f,  3.0f, 0.0f);
			glVertex3f( 0.5f,  2.5f, 0.0f);
			glVertex3f( 0.0f,  3.0f, 0.0f);

			// z轴
			glColor3f(0.0F, 0.0F, 1.0F);
			glVertex3f( 0.0f, 0.0f, -3.0f);
			glVertex3f( 0.0f, 0.0f,  3.0f);
			glVertex3f(-0.5f, 0.0f,  2.5f);
			glVertex3f( 0.0f, 0.0f,  3.0f);
			glVertex3f( 0.5f, 0.0f,  2.5f);
			glVertex3f( 0.0f, 0.0f,  3.0f);
	glEnd();
	glPopMatrix();
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

	m_3dsLoaded  = FALSE;

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


//	m_triList.Init();
	
	::glShadeModel(GL_FLAT);
	
	::glClearColor(0.0F, 0.0F, 0.0F, 0.0F);
	
	::glClearDepth(1.0F);

	::glEnable(GL_DEPTH_TEST);

	::glEnable(GL_CULL_FACE);

	//Change Light by  cxf
	GLfloat ambientLight[] = { 0.23f, 0.23f, 0.23f, 1.0f};
	GLfloat diffuseLight[] = { 0.027f, 0.027f, 0.027f, 1.0f};
	GLfloat lightPos[]     = {0.0f,3000.0f,0.0f, 1.0f};

	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientLight);
	
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

}

void CPmModelTool::OnRButtonUp(UINT nFlags, CPoint point) 
{
	// TODO: Add your message handler code here and/or call default
	ReleaseCapture( );
	mouserightdown = FALSE;
	SetCamPos(2, (point.y - mouseprevpoint.y) , TRUE, TRUE);
	
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
	
	if(mouserightdown)
	{
		SetCamPos(2, -(point.y - mouseprevpoint.y) , TRUE,TRUE);
	}
	else if(mouseleftdown)
	{	
		SetSceneRot(0, (point.y - mouseprevpoint.y) , TRUE,TRUE);
		SetSceneRot(2, (point.x - mouseprevpoint.x) , TRUE,TRUE);
	}
	CView::OnMouseMove(nFlags, point);

	mouseprevpoint.x = point.x;
	mouseprevpoint.y = point.y;	
	CView::OnMouseMove(nFlags, point);
}

void CPmModelTool::OnLButtonUp(UINT nFlags, CPoint point) 
{
	// TODO: Add your message handler code here and/or call default
	ReleaseCapture( );
	mouseleftdown = FALSE;
	SetSceneRot(0, (point.y - mouseprevpoint.y) , TRUE, TRUE);
	SetSceneRot(2, (point.x - mouseprevpoint.x) , TRUE, TRUE);
	
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
