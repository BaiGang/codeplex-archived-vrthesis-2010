// 3DSLoaderView.h : interface of the CPmModelTool class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_MYSDOPENGLVIEW_H__75C5AAEC_37B0_4A8B_9132_9A0C663F6DDC__INCLUDED_)
#define AFX_MYSDOPENGLVIEW_H__75C5AAEC_37B0_4A8B_9132_9A0C663F6DDC__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
//
//#include "glStructures.h"	//数据结构定义
//#include "3dsReader.h"		//C3dsReader说明文件
//#include "TriList.h"		//CTriList说明文件
//#include "header.h"
#include "shader/GLSLShader.h"
#include "math/geomath.h"

#define BOX_SIZE 17.0f
#define BOX_LENGTH 128



class CPmModelTool : public CView
{
protected: // create from serialization only
	CPmModelTool();
	DECLARE_DYNCREATE(CPmModelTool)

// Attributes
public:
	CtestDoc* GetDocument();

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CPmModelTool)
	public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CPmModelTool();
/////////////////////////////////////////////////////////////////
//添加成员函数与成员变量
	BOOL RenderScene();
	BOOL SetupPixelFormat(void);
	void SetLogicalPalette(void);
	BOOL InitializeOpenGL(CDC* pDC);

	HGLRC		m_hRC;			//OpenGL绘制描述表
	HPALETTE	m_hPalette;		//OpenGL调色板
	CDC*	    m_pDC;			//OpenGL设备描述表
/////////////////////////////////////////////////////////////////
	void Init(GLvoid);
	void Draw3ds();
	void DrawAxis();
	void SetSceneRot(int axis, int value, BOOL increment, BOOL apply);
	void SetCamPos(int axis, int value, BOOL increment, BOOL apply);
	BOOL OpenFile(LPCTSTR lpszPathName);

/*
	C3dsExport  exp;
	CTriList	m_triList;	*/	
	//BOOL		m_3dsLoaded;	
	float		camRot[3];		
	float		camPos[3];		
	float		sceneRot[3];	
	float		scenePos[3];	
	BOOL		mouserightdown;	
	BOOL		mouseleftdown;	
	CPoint		mouseprevpoint; 

	// utility functions for setting opengl
	bool InitGL(void);
	bool ReshapeGL(int width, int height);
	bool DisplayGL(void);
	// load in camera parameters
	bool LoadCameras(void);

	void DrawBox();
	void DrawSmoke();

	int GetOrientation();
	void DrawSmoke_alongxN(void);
	void DrawSmoke_alongyN(void);
	void DrawSmoke_alongzN(void);
	void DrawSmoke_alongxP(void);
	void DrawSmoke_alongyP(void);
	void DrawSmoke_alongzP(void);

	
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	//{{AFX_MSG(CPmModelTool)
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnDestroy();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnTimer(UINT nIDEvent);
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg void OnRButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnFileSave();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
protected:
	afx_msg void OnBnClickedRadioCamera();
	afx_msg void OnBnClickedRadioLight();

	PFMImage * m_pLastPic;
	PFMImage * m_pCurrentPic;
	PFMImage * m_pNextPic;
	int picIndex;
	CPoint					m_mouseLast;			// Mouse Last point
	enum RBState {CAMERA, LIGHT};
	RBState currentRBState;

	void loadPic();
	char* cstring2char (CString cstr);
	CString char2cstring(char* cstr);

	bool itemHasNext(int index);
	bool itemHasLast(int index);
	int getFileCount(CString csFolderName);

	void outputText(char* str);

	afx_msg void OnBnClickedLast();
	afx_msg void OnBnClickedNext();

	//////////////////////////////////////
	//  member variables for rendering
	//////////////////////////////////////

	static const int NUM_CAMERAS = 8;

	// GLSL shaders for volumetric rendering
	GLSLShader m_shader_along_x;
	GLSLShader m_shader_along_y;
	GLSLShader m_shader_along_z;

	Vector4 m_CameraPos;
	float m_CameraInv[16];
	float m_LightPosition[4];
	float m_LightDist;
	float m_LightMultiplier;
	float m_extinction;
	float m_scattering;

	// mv and prj matrix for each camera
	Matrix4 m_modelview_mats[NUM_CAMERAS];
	Matrix4 m_projection_mats[NUM_CAMERAS];

	// GL 3D texture for volume data
	GLuint m_tex3d_id;

	// quaternion for rotation

	// mouse motion on screen
	double m_mouse_x;
	double m_mouse_y;
	float m_quatRotate_start[4];
	float m_quatRotate_update[4];
public:
	void SetCamera(int i);
	afx_msg void OnButtonCamera0();
	afx_msg void OnButtonCamera3();
	afx_msg void OnButtonCamera1();
	afx_msg void OnButtonCamera2();
	afx_msg void OnButtonCamera4();
	afx_msg void OnButtonCamera5();
	afx_msg void OnButtonCamera6();
	afx_msg void OnButtonCamera7();
	afx_msg void OnSlider1();
	afx_msg void OnSlider2();
	afx_msg void OnSlider3();
	bool setVolume(int id, int n);
	afx_msg void OnVolEdit1();
	afx_msg void OnVolEdit2();
	afx_msg void OnVolEdit3();
	afx_msg void OnButtonStart();

	// set .exe  position
	void setExePos();
};

#ifndef _DEBUG  // debug version in 3DSLoaderView.cpp
inline CMy3DSLoaderDoc* CPmModelTool::GetDocument()
   { return (CMy3DSLoaderDoc*)m_pDocument; }
#endif

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_MYSDOPENGLVIEW_H__75C5AAEC_37B0_4A8B_9132_9A0C663F6DDC__INCLUDED_)
