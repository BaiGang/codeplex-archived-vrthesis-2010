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
END_MESSAGE_MAP()

// CasmodelingView construction/destruction

CasmodelingView::CasmodelingView()
{
	// TODO: add construction code here

}

CasmodelingView::~CasmodelingView()
{
}

BOOL CasmodelingView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

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
