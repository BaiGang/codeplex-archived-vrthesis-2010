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

// asmodelingView.h : interface of the CasmodelingView class
//


#pragma once

#include "image\Image.h"
#include "image\PFMImage.h"

class CasmodelingView : public CView
{
protected: // create from serialization only
	CasmodelingView();
	DECLARE_DYNCREATE(CasmodelingView)

// Attributes
public:
	CasmodelingDoc* GetDocument() const;

// Operations
public:

// Overrides
public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// Implementation
public:
	virtual ~CasmodelingView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	afx_msg void OnFilePrintPreview();
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnContextMenu(CWnd* pWnd, CPoint point);
	DECLARE_MESSAGE_MAP()
public:
	BOOL SetWindowPixelFormat(HDC hDC);
	int m_GLPixelIndex;
	BOOL CreateViewGLContext(HDC hDC);
	HGLRC m_hGLContext;
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnDestroy();
	afx_msg void OnSize(UINT nType, int cx, int cy);
protected:

public:
	afx_msg void OnPaint();

	CPoint					m_mouseLast;			// Mouse Last point
	enum RBState {CAMERA, LIGHT};
	RBState currentRBState;

	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	void SetRBstate(bool flag);

	afx_msg void OnBnClickedLast();
	afx_msg void OnBnClickedNext();

	PFMImage* lastPic;
	PFMImage* currentPic;
	PFMImage* nextPic;
	int picIndex;

	void loadPic();
	char* cstring2char (CString cstr);
	CString char2cstring(char* cstr);

	bool itemHasNext(int index);
	bool itemHasLast(int index);
	int getFileCount(CString csFolderName);
	
	void outputText(char* str);
};

#ifndef _DEBUG  // debug version in asmodelingView.cpp
inline CasmodelingDoc* CasmodelingView::GetDocument() const
   { return reinterpret_cast<CasmodelingDoc*>(m_pDocument); }
#endif

