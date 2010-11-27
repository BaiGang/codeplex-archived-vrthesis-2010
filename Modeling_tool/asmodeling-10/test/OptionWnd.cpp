// OptionWnd.cpp : 实现文件
//

#include "stdafx.h"
#include "test.h"
#include "OptionWnd.h"


// COptionWnd

IMPLEMENT_DYNAMIC(COptionWnd, CDockablePane)

COptionWnd::COptionWnd()
{

}

COptionWnd::~COptionWnd()
{
}


BEGIN_MESSAGE_MAP(COptionWnd, CDockablePane)
	ON_WM_CREATE()
	ON_WM_SIZE()
END_MESSAGE_MAP()



// COptionWnd 消息处理程序




int COptionWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CDockablePane::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  在此添加您专用的创建代码

	m_font.CreateFontW (16, 0, 0, 0, FW_NORMAL, 0, 0, 0, ANSI_CHARSET, OUT_STROKE_PRECIS,
		CLIP_STROKE_PRECIS , DRAFT_QUALITY,
		VARIABLE_PITCH|FF_SWISS, _T("Arial") );

	radioCamera.Create(_T("Camera"), WS_VISIBLE | WS_CHILD | BS_RADIOBUTTON, CRect
		(20,20,70,40), this, IDC_RADIO_CAMERA);
	radioCamera.SetFont(&m_font);

	radioLight.Create(_T("Light"), WS_VISIBLE | WS_CHILD | BS_RADIOBUTTON, CRect
		(20,40,70,40), this, IDC_RADIO_LIGHT);
	radioLight.SetFont(&m_font);

	blankBn.Create(_T(""),WS_VISIBLE | WS_CHILD | BS_PUSHBOX,CRect
		(20,60,70,40), this, NULL);

	return 0;
}


void COptionWnd::OnSize(UINT nType, int cx, int cy)
{
	CDockablePane::OnSize(nType, cx, cy);

	// TODO: 在此处添加消息处理程序代码
	if (radioCamera.GetSafeHwnd() != NULL)
	{
		CRect rectClient;
		GetClientRect(rectClient);
		radioCamera.SetWindowPos(NULL, rectClient.left, rectClient.top, rectClient.Width(),rectClient.top+20, SWP_NOACTIVATE | SWP_NOZORDER);
	}

	if (radioLight.GetSafeHwnd() != NULL)
	{
		CRect rectClient;
		GetClientRect(rectClient);
		radioLight.SetWindowPos(NULL, rectClient.left, rectClient.top+20, rectClient.Width(),rectClient.top + 40, SWP_NOACTIVATE | SWP_NOZORDER);
	}

	if (blankBn.GetSafeHwnd() != NULL)
	{
		CRect rectClient;
		GetClientRect(rectClient);
		blankBn.SetWindowPos(NULL, rectClient.left, rectClient.top+55, rectClient.Width(),rectClient.Height(), SWP_NOACTIVATE | SWP_NOZORDER);
	}

}


void COptionWnd::setCamera()
{
	// TODO: 在此添加控件通知处理程序代码
	radioCamera.SetCheck(true);
	radioLight.SetCheck(false);
	Invalidate();
}


void COptionWnd::setLight()
{
	// TODO: 在此添加控件通知处理程序代码
	radioCamera.SetCheck(false);
	radioLight.SetCheck(true);
	Invalidate();
}