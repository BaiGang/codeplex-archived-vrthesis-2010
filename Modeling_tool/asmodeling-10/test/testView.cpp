// ��� MFC ʾ��Դ������ʾ���ʹ�� MFC Microsoft Office Fluent �û����� 
// (��Fluent UI��)����ʾ�������ο���
// ���Բ��䡶Microsoft ������ο����� 
// MFC C++ ������渽����ص����ĵ���
// ���ơ�ʹ�û�ַ� Fluent UI ����������ǵ����ṩ�ġ�
// ��Ҫ�˽��й� Fluent UI ��ɼƻ�����ϸ��Ϣ�������  
// http://msdn.microsoft.com/officeui��
//
// ��Ȩ����(C) Microsoft Corporation
// ��������Ȩ����

// testView.cpp : CtestView ���ʵ��
//

#include "stdafx.h"
// SHARED_HANDLERS ������ʵ��Ԥ��������ͼ������ɸѡ�������
// ATL ��Ŀ�н��ж��壬�����������Ŀ�����ĵ����롣
#ifndef SHARED_HANDLERS
#include "test.h"
#endif

#include "testDoc.h"
#include "testView.h"
#include "MainFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CtestView

IMPLEMENT_DYNCREATE(CtestView, CFormView)

BEGIN_MESSAGE_MAP(CtestView, CFormView)
	ON_WM_CONTEXTMENU()
	ON_WM_RBUTTONUP()
	ON_WM_MOUSEMOVE()
	ON_BN_CLICKED(IDC_RADIO_CAMERA, &CtestView::OnBnClickedRadioCamera)
	ON_BN_CLICKED(IDC_RADIO_LIGHT, &CtestView::OnBnClickedRadioLight)
	ON_COMMAND(ID_LAST, &CtestView::OnBnClickedLast)
	ON_COMMAND(ID_NEXT, &CtestView::OnBnClickedNext)
	ON_WM_CREATE()
	ON_WM_PAINT()
	ON_WM_DESTROY()
	ON_WM_SIZE()
END_MESSAGE_MAP()

// CtestView ����/����

CtestView::CtestView()
	: CFormView(CtestView::IDD)
{
	// TODO: �ڴ˴���ӹ������
	this->m_GLPixelIndex = 0;
	this->m_hGLContext = NULL;
/*
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
	*/
}	

CtestView::~CtestView()
{
}

void CtestView::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
}

BOOL CtestView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: �ڴ˴�ͨ���޸�
	//  CREATESTRUCT cs ���޸Ĵ��������ʽ
	cs.style |= (WS_CLIPCHILDREN | WS_CLIPSIBLINGS);
	return CFormView::PreCreateWindow(cs);
}

void CtestView::OnInitialUpdate()
{
	CFormView::OnInitialUpdate();
	GetParentFrame()->RecalcLayout();
	ResizeParentToFit();

}

BOOL CtestView::SetWindowPixelFormat(HDC hDC)
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

BOOL CtestView::CreateViewGLContext(HDC hDC)
{
	this->m_hGLContext = wglCreateContext(hDC);
	if(this->m_hGLContext==NULL)
	{//����ʧ��
		return FALSE;
	}

	if(wglMakeCurrent(hDC,this->m_hGLContext)==FALSE)
	{//ѡΪ��ǰRCʧ��
		return FALSE;
	}

	return TRUE;
}

void CtestView::OnRButtonUp(UINT /* nFlags */, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void CtestView::OnContextMenu(CWnd* /* pWnd */, CPoint point)
{
#ifndef SHARED_HANDLERS
	theApp.GetContextMenuManager()->ShowPopupMenu(IDR_POPUP_EDIT, point.x, point.y, this, TRUE);
#endif
}


// CtestView ���

#ifdef _DEBUG
void CtestView::AssertValid() const
{
	CFormView::AssertValid();
}

void CtestView::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}

CtestDoc* CtestView::GetDocument() const // �ǵ��԰汾��������
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CtestDoc)));
	return (CtestDoc*)m_pDocument;
}
#endif //_DEBUG


// CtestView ��Ϣ�������


void CtestView::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: �ڴ������Ϣ�����������/�����Ĭ��ֵ
	
	CPoint mouseDiff = point - m_mouseLast;
	double x,y;
	x = mouseDiff.x;
	x /= 50;
	y = mouseDiff.y;
	y /= 50;

	//�������
	if(nFlags & MK_LBUTTON)
	{
		
	}

	m_mouseLast = point;

	CFormView::OnMouseMove(nFlags, point);
}

void CtestView::OnBnClickedLast()
{
	if(itemHasLast(picIndex))
	{
		picIndex -= 1;
		loadPic();
	}
	else
		AfxMessageBox(_T("It has been the first Picture!"));
}

void CtestView::OnBnClickedNext()
{
	if(itemHasNext(picIndex))
	{
		picIndex += 1;
		loadPic();
	}
	else
		AfxMessageBox(_T("It has been the last Picture!"));
}

void CtestView::OnBnClickedRadioCamera()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	currentRBState = CAMERA;
	((CMainFrame*)theApp.m_pMainWnd)->setRadio(true);
	Invalidate();
}


void CtestView::OnBnClickedRadioLight()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	currentRBState = LIGHT;
	((CMainFrame*)theApp.m_pMainWnd)->setRadio(false);
	Invalidate();
}

void CtestView::loadPic()
{
	// TODO: �ڴ���������������
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

char* CtestView::cstring2char (CString cstr)
{
	int len = WideCharToMultiByte(CP_UTF8, 0, cstr.AllocSysString(), -1, NULL, 0, NULL, NULL);  
	char * szUtf8=new char[len + 1];
	WideCharToMultiByte (CP_UTF8, 0, cstr.AllocSysString(), -1, szUtf8, len, NULL,NULL);
	return szUtf8;
}

CString CtestView::char2cstring(char* cstr)
{	
	CString str;
	str = cstr;
	return str;
}

bool CtestView::itemHasNext(int index)
{
	if (index < getFileCount(theApp.rootPath+_T("./PFM/")))
		return true;
	else 
		return false;
}

bool CtestView::itemHasLast(int index)
{
	if (index >= getFileCount(theApp.rootPath+_T("./PFM/")))
		return true;
	else 
		return false;
}

int CtestView::getFileCount(CString csFolderName)
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

void CtestView::outputText(char* str)
{
	((CMainFrame*)theApp.m_pMainWnd)->outputText(char2cstring(str));
}

int CtestView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
/*	

	if (CFormView::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  �ڴ������ר�õĴ�������
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
	*/
	return 0;
}


void CtestView::OnPaint()
{
	CPaintDC dc(this); // device context for painting
	// TODO: �ڴ˴������Ϣ����������
	// ��Ϊ��ͼ��Ϣ���� CFormView::OnPaint()
	
}


void CtestView::OnDestroy()
{
	CFormView::OnDestroy();

	// TODO: �ڴ˴������Ϣ����������
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


void CtestView::OnSize(UINT nType, int cx, int cy)
{
	CFormView::OnSize(nType, cx, cy);

	// TODO: �ڴ˴������Ϣ����������
/*
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
	*/
}