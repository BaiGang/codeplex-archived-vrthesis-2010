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

// testView.cpp : CtestView 类的实现
//

#include "stdafx.h"
// SHARED_HANDLERS 可以在实现预览、缩略图和搜索筛选器句柄的
// ATL 项目中进行定义，并允许与该项目共享文档代码。
#ifndef SHARED_HANDLERS
#include "test.h"
#endif

#include "testDoc.h"
#include "testView.h"

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
END_MESSAGE_MAP()

// CtestView 构造/析构

CtestView::CtestView()
	: CFormView(CtestView::IDD)
{
	// TODO: 在此处添加构造代码

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
	// TODO: 在此处通过修改
	//  CREATESTRUCT cs 来修改窗口类或样式

	return CFormView::PreCreateWindow(cs);
}

void CtestView::OnInitialUpdate()
{
	CFormView::OnInitialUpdate();
	GetParentFrame()->RecalcLayout();
	ResizeParentToFit();

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


// CtestView 诊断

#ifdef _DEBUG
void CtestView::AssertValid() const
{
	CFormView::AssertValid();
}

void CtestView::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}

CtestDoc* CtestView::GetDocument() const // 非调试版本是内联的
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CtestDoc)));
	return (CtestDoc*)m_pDocument;
}
#endif //_DEBUG


// CtestView 消息处理程序


void CtestView::OnMouseMove(UINT nFlags, CPoint point)
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

	CFormView::OnMouseMove(nFlags, point);
}


void CtestView::OnBnClickedRadioCamera()
{
	// TODO: 在此添加控件通知处理程序代码
	currentRBState = CAMERA;
	Invalidate();
}


void CtestView::OnBnClickedRadioLight()
{
	// TODO: 在此添加控件通知处理程序代码
	currentRBState = LIGHT;
	Invalidate();
}
