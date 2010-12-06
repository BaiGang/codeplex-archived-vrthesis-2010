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

#include "math/SimpleQuat.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CtestView

IMPLEMENT_DYNCREATE(CtestView, CView)

	BEGIN_MESSAGE_MAP(CtestView, CView)
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
	: CView()
{
}	

CtestView::~CtestView()
{
}

void CtestView::DoDataExchange(CDataExchange* pDX)
{
	CView::DoDataExchange(pDX);
}

BOOL CtestView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: �ڴ˴�ͨ���޸�
	//  CREATESTRUCT cs ���޸Ĵ��������ʽ
	cs.style |= (WS_CLIPCHILDREN | WS_CLIPSIBLINGS);
	return CView::PreCreateWindow(cs);
}

void CtestView::OnInitialUpdate()
{
	CView::OnInitialUpdate();
	GetParentFrame()->RecalcLayout();
	//ResizeParentToFit();

}

BOOL CtestView::SetWindowPixelFormat(HDC hDC)
{
	PIXELFORMATDESCRIPTOR pixelDesc=
	{
		sizeof(PIXELFORMATDESCRIPTOR),
		1,
		PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER,
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
	CView::AssertValid();
}

void CtestView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
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

	//�������
	if(nFlags & MK_LBUTTON)
	{
		m_mouse_x = mouseDiff.x;
		m_mouse_x /= 50;
		m_mouse_y = mouseDiff.y;
		m_mouse_y /= 50;
	}

	m_mouseLast = point;

	CView::OnMouseMove(nFlags, point);
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
		m_pLastPic->ReadImage(cstring2char(filestr));
	}

	filestr.Format(_T("./PFM/%d.pfm"), picIndex);
	m_pCurrentPic->ReadImage(cstring2char(filestr));

	if(itemHasNext(picIndex))
	{
		filestr.Format(_T("./PFM/%d.pfm"), picIndex+1);
		m_pNextPic->ReadImage(cstring2char(filestr));
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
	if (CView::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  �ڴ������ר�õĴ�������
	
	this->m_hDC=::GetDC(this->m_hWnd);//��ȡ�豸�������

	ASSERT(this->m_hDC);

	//�������ظ�ʽ Pixel Format
	static PIXELFORMATDESCRIPTOR pfdWnd=
	{sizeof(PIXELFORMATDESCRIPTOR),
	1,
	PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER,
	PFD_TYPE_RGBA,
	24,
	0,0,0,0,0,0,
	0,0,0,0,0,0,0,
	32,
	0,0,
	PFD_MAIN_PLANE,
	0,
	0,0,0};

	//���豸������ѡȡ����������ƥ������ظ�ʽ
	int pixelformat;
	pixelformat=ChoosePixelFormat(m_hDC,&pfdWnd);

	//Ϊ�豸�����������ظ�ʽ
	ASSERT(SetPixelFormat(m_hDC,pixelformat,&pfdWnd));

	//������Ⱦ����
	m_hGLContext=wglCreateContext(m_hDC);

	//ѡ����Ⱦ����m_hRCΪ��ǰ����
	VERIFY(wglMakeCurrent(m_hDC,m_hGLContext));

	////��ʼ����Ⱦ����
	//GLProcess::GetInstance()->GLSetupRC(m_hDC);

	//�ر���Ⱦ����
	VERIFY(wglMakeCurrent(NULL,NULL));


	InitGL();

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
	CView::OnDestroy();

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
	CView::OnSize(nType, cx, cy);

	ReshapeGL(cx, cy);
}

bool CtestView::InitGL(void)
{
	// init glew
	if (!InitGLExtensions())
	{
		AfxMessageBox(_T("InitGLExtensions failed!"));
		return 0;
	}

	// init shaders
	m_shader_alongX.InitShaders("../Data/RayMarchingBlend.vert","../Data/RayMarchingBlendXU.frag");
	m_shader_alongY.InitShaders("../Data/RayMarchingBlend.vert","../Data/RayMarchingBlendYU.frag");
	m_shader_alongZ.InitShaders("../Data/RayMarchingBlend.vert","../Data/RayMarchingBlendZU.frag");

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
	m_pLastPic     = new PFMImage(256, 256*256, 0, NULL);
	m_pCurrentPic  = new PFMImage(256, 256*256, 0, NULL);
	m_pNextPic     = new PFMImage(256, 256*256, 0, NULL);

	// set rotate start quat
	m_quatRotate_start[0] = 0.0f;
	m_quatRotate_start[1] = 0.0f;
	m_quatRotate_start[2] = 0.0f;
	m_quatRotate_start[3] = 1.0f;


	return true;
}

bool CtestView::ReshapeGL(int width, int height)
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


bool CtestView::DisplayGL(void)
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

bool CtestView::LoadCameras(void)
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



void CtestView::OnDraw(CDC* /*pDC*/)
{
	// TODO: �ڴ����ר�ô����/����û���
	DisplayGL();
}


void CtestView::DrawBox(void)
{
	glBegin(GL_POINTS);
	
	glEnd();
}

void CtestView::DrawSmoke(void)
{
	glBegin(GL_POINTS);
	
	glEnd();
}