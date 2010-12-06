// 这段 MFC 示例源代码演示如何使用 MFC Microsoft Office Fluent 用户界面 
// (“Fluent UI”)。该示例仅供参考，
// 用以补充《Microsoft 基础类参考》和 
// MFC C++ 库软件随附的相关电子文档。
// 复制、使用或分发 Fluent UI 的许可条款是单独提供的。
// 若要了解有关 Fluent UI 许可计划的详细信息，请访问  
// http://msdn.microsoft.com/officeui。
//
// 版权所有(C) Microsoft Corporation
// 保留所有权利。

// testView.h : CtestView 类的接口
//

#pragma once

#include "resource.h"

#include "shader/GLSLShader.h"
#include "math/geomath.h"


class CtestView : public CView
{
protected: // 仅从序列化创建
	CtestView();
	DECLARE_DYNCREATE(CtestView)

public:
	enum{ IDD = IDD_TEST_FORM };

// 特性
public:
	CtestDoc* GetDocument() const;

// 操作
public:

// 重写
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持
	virtual void OnInitialUpdate(); // 构造后第一次调用
	CPoint					m_mouseLast;			// Mouse Last point
	enum RBState {CAMERA, LIGHT};
	RBState currentRBState;
// 实现
public:
	virtual ~CtestView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:
	// utility functions for setting opengl
	bool InitGL(void);
	bool ReshapeGL(int width, int height);
	bool DisplayGL(void);
	// load in camera parameters
	bool LoadCameras(void);

	void DrawBox();
	void DrawSmoke();

// 生成的消息映射函数
protected:
	afx_msg void OnFilePrintPreview();
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnContextMenu(CWnd* pWnd, CPoint point);
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg void OnBnClickedRadioCamera();
	afx_msg void OnBnClickedRadioLight();

	PFMImage * m_pLastPic;
	PFMImage * m_pCurrentPic;
	PFMImage * m_pNextPic;
	int picIndex;

	void loadPic();
	char* cstring2char (CString cstr);
	CString char2cstring(char* cstr);

	bool itemHasNext(int index);
	bool itemHasLast(int index);
	int getFileCount(CString csFolderName);

	void outputText(char* str);

	afx_msg void OnBnClickedLast();
	afx_msg void OnBnClickedNext();

	BOOL SetWindowPixelFormat(HDC hDC);
	int m_GLPixelIndex;
	BOOL CreateViewGLContext(HDC hDC);
	HGLRC m_hGLContext;
	HDC m_hDC;
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnPaint();
	afx_msg void OnDestroy();
	afx_msg void OnSize(UINT nType, int cx, int cy);

protected:
	//////////////////////////////////////
	//  member variables for rendering
	//////////////////////////////////////
	
	static const int NUM_CAMERAS = 8;

	// GLSL shaders for volumetric rendering
	GLSLShader m_shader_alongX;
	GLSLShader m_shader_alongY;
	GLSLShader m_shader_alongZ;

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



	virtual void OnDraw(CDC* /*pDC*/);
};

#ifndef _DEBUG  // testView.cpp 中的调试版本
inline CtestDoc* CtestView::GetDocument() const
   { return reinterpret_cast<CtestDoc*>(m_pDocument); }
#endif

