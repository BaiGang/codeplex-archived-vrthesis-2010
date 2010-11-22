// This MFC Samples source code demonstrates using MFC Microsoft Office Fluent User Interface 
// (the "Fluent UI") and is provided only as referential material to supplement the 
// Microsoft Foundation Classes Reference and related electronic documentation 
// included with the MFC C++ library software.  
// License terms to copy, use or distribute the Fluent UI are available separately.  
// To learn more about our Fluent UI licensing program, please visit 
// http://msdn.microsoft.com/officeui.
//
// Copyright (C) Microsoft Corporation
// All rights reserved.

// asmodelingView.cpp : implementation of the CasmodelingView class
//

#include "stdafx.h"
#include "asmodeling.h"

#include "asmodelingDoc.h"
#include "asmodelingView.h"
#include "MainFrm.h"
#include "OutputWnd.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CasmodelingView

IMPLEMENT_DYNCREATE(CasmodelingView, CView)

BEGIN_MESSAGE_MAP(CasmodelingView, CView)
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CasmodelingView::OnFilePrintPreview)
	ON_WM_CREATE()
	ON_WM_DESTROY()
	ON_WM_SIZE()
	ON_WM_PAINT()
	ON_WM_MOUSEMOVE()
	ON_COMMAND(ID_LAST, &CasmodelingView::OnBnClickedLast)
	ON_COMMAND(ID_NEXT, &CasmodelingView::OnBnClickedNext)
END_MESSAGE_MAP()

// CasmodelingView construction/destruction

CasmodelingView::CasmodelingView()
: m_GLPixelIndex(0)
{
	// TODO: add construction code here
	this->m_GLPixelIndex = 0;
	this->m_hGLContext = NULL;


	picIndex = 0;
	//assume
	float length = 320;
	float * imgdata1 = new float [length * length*length];
	float * imgdata2 = new float [length * length*length];
	float * imgdata3 = new float [length * length*length];
	lastPic =  new PFMImage(length, length*length, 0, imgdata1);
	currentPic = new PFMImage(length, length*length, 0, imgdata2);
	nextPic = new PFMImage(length, length*length, 0, imgdata3);

	loadPic();

}

CasmodelingView::~CasmodelingView()
{
}

BOOL CasmodelingView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	cs.style |= (WS_CLIPCHILDREN | WS_CLIPSIBLINGS);
	return CView::PreCreateWindow(cs);
}

// CasmodelingView drawing

void CasmodelingView::OnDraw(CDC* /*pDC*/)
{
	CasmodelingDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: add draw code for native data here
	
}


// CasmodelingView printing


void CasmodelingView::OnFilePrintPreview()
{
	AFXPrintPreview(this);
}

BOOL CasmodelingView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CasmodelingView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CasmodelingView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}

void CasmodelingView::OnRButtonUp(UINT nFlags, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void CasmodelingView::OnContextMenu(CWnd* pWnd, CPoint point)
{
	theApp.GetContextMenuManager()->ShowPopupMenu(IDR_POPUP_EDIT, point.x, point.y, this, TRUE);
}


// CasmodelingView diagnostics

#ifdef _DEBUG
void CasmodelingView::AssertValid() const
{
	CView::AssertValid();
}

void CasmodelingView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CasmodelingDoc* CasmodelingView::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CasmodelingDoc)));
	return (CasmodelingDoc*)m_pDocument;
}
#endif //_DEBUG


// CasmodelingView message handlers

BOOL CasmodelingView::SetWindowPixelFormat(HDC hDC)
{
	PIXELFORMATDESCRIPTOR pixelDesc=
	{
		sizeof(PIXELFORMATDESCRIPTOR),
		1,
		PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|
		PFD_DOUBLEBUFFER|PFD_SUPPORT_GDI,
		PFD_TYPE_RGBA,
		24,
		0,0,0,0,0,0,
		0,
		0,
		0,
		0,0,0,0,
		32,
		0,
		0,
		PFD_MAIN_PLANE,
		0,
		0,0,0
	};

	this->m_GLPixelIndex = ChoosePixelFormat(hDC,&pixelDesc);
	if(this->m_GLPixelIndex==0)
	{
		this->m_GLPixelIndex = 1;
		if(DescribePixelFormat(hDC,this->m_GLPixelIndex,sizeof(PIXELFORMATDESCRIPTOR),&pixelDesc)==0)
		{
			return FALSE;
		}
	}

	if(SetPixelFormat(hDC,this->m_GLPixelIndex,&pixelDesc)==FALSE)
	{
		return FALSE;
	}
	return TRUE;
}

BOOL CasmodelingView::CreateViewGLContext(HDC hDC)
{
	this->m_hGLContext = wglCreateContext(hDC);
	if(this->m_hGLContext==NULL)
	{//创建失败
		return FALSE;
	}

	if(wglMakeCurrent(hDC,this->m_hGLContext)==FALSE)
	{//选为当前RC失败
		return FALSE;
	}

	return TRUE;
}

int CasmodelingView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CView::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  Add your specialized creation code here


	HWND hWnd = this->GetSafeHwnd();    
	HDC hDC = ::GetDC(hWnd);
	if(this->SetWindowPixelFormat(hDC)==FALSE)
	{
		AfxMessageBox(_T("setwindowpixelFormat failed!"));
		return 0;
	}
	if(this->CreateViewGLContext(hDC)==FALSE)
	{
		AfxMessageBox(_T("createViewGLContext failed!"));
		return 0;
	}
	return 0;

}

void CasmodelingView::OnDestroy()
{
	CView::OnDestroy();

	// TODO: Add your message handler code here
	if(wglGetCurrentContext()!=NULL)
	{
		wglMakeCurrent(NULL,NULL);
	}
	if(this->m_hGLContext!=NULL)
	{
		wglDeleteContext(this->m_hGLContext);
		this->m_hGLContext = NULL;
	}
}

void CasmodelingView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	// TODO: Add your message handler code here
	GLsizei width,height;
	GLdouble aspect;
	width = cx;
	height = cy;
	if(cy==0)
	{
		aspect = (GLdouble)width;
	}
	else
	{
		aspect = (GLdouble)width/(GLdouble)height;
	}
	glViewport(0,0,width,height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0,500.0*aspect,0.0,500.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}


void CasmodelingView::OnPaint()
{
	CPaintDC dc(this); // device context for painting
	// TODO: Add your message handler code here
	// Do not call CView::OnPaint() for painting messages

	CPen   pen(PS_SOLID,1,RGB(0,0,0)); 
	CPen   *pOldPen=dc.SelectObject   (&pen); 
	dc.MoveTo(50,50); 
	dc.LineTo(200,200); 
	dc.SelectObject(pOldPen); 


	glLoadIdentity();
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_POLYGON);
	glColor4f(1.0f,0.0f,0.0f,1.0f);
	glVertex2f(100.0f,50.0f);
	glColor4f(0.0f,1.0f,0.0f,1.0f);
	glVertex2f(450.0f,400.0f);
	glColor4f(0.0f,0.0f,1.0f,1.0f);
	glVertex2f(450.0f,50.0f);
	glEnd();
	glFlush();
}



void CasmodelingView::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值

	CPoint mouseDiff = point - m_mouseLast;
	double x,y;
	x = mouseDiff.x;
	x /= 50;
	y = mouseDiff.y;
	y /= 50;

	//左键按下
	if(nFlags & MK_LBUTTON)
	{

	}

	m_mouseLast = point;

	CView::OnMouseMove(nFlags, point);
}

void CasmodelingView::SetRBstate(bool flag)
{
	if(flag)
		currentRBState = CAMERA;
	else
		currentRBState = LIGHT;
}

void CasmodelingView::OnBnClickedLast()
{
	if(itemHasLast(picIndex))
	{
		picIndex -= 1;
		loadPic();
	}
	else
		AfxMessageBox(_T("It has been the first Picture!"));
}

void CasmodelingView::OnBnClickedNext()
{
	if(itemHasNext(picIndex))
	{
		picIndex += 1;
		loadPic();
	}
	else
		AfxMessageBox(_T("It has been the last Picture!"));
}

void CasmodelingView::loadPic()
{
	// TODO: 在此添加命令处理程序代码
	CString filestr;

	if(itemHasLast(picIndex))
	{
		filestr.Format(_T("./PFM/%d.pfm"), picIndex-1);
		lastPic->ReadImage(cstring2char(filestr));
	}
	
	filestr.Format(_T("./PFM/%d.pfm"), picIndex);
	currentPic->ReadImage(cstring2char(filestr));

	if(itemHasNext(picIndex))
	{
		filestr.Format(_T("./PFM/%d.pfm"), picIndex+1);
		nextPic->ReadImage(cstring2char(filestr));
	}
}

char* CasmodelingView::cstring2char (CString cstr)
{
	int len = WideCharToMultiByte(CP_UTF8, 0, cstr.AllocSysString(), -1, NULL, 0, NULL, NULL);  
	char * szUtf8=new char[len + 1];
	WideCharToMultiByte (CP_UTF8, 0, cstr.AllocSysString(), -1, szUtf8, len, NULL,NULL);
	return szUtf8;
}

CString CasmodelingView::char2cstring(char* cstr)
{	
	CString str;
	str = cstr;
	return str;
}

bool CasmodelingView::itemHasNext(int index)
{
	if (index < getFileCount(theApp.rootPath+_T("./PFM/")))
		return true;
	else 
		return false;
}

bool CasmodelingView::itemHasLast(int index)
{
	if (index >= getFileCount(theApp.rootPath+_T("./PFM/")))
		return true;
	else 
		return false;
}

int CasmodelingView::getFileCount(CString csFolderName)
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

void CasmodelingView::outputText(char* str)
{
	((CMainFrame*)theApp.m_pMainWnd)->outputText(char2cstring(str));
}