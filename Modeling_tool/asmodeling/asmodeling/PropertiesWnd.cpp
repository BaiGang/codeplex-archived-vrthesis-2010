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

#include "stdafx.h"

#include "PropertiesWnd.h"
#include "Resource.h"
#include "MainFrm.h"
#include "asmodeling.h"

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

/////////////////////////////////////////////////////////////////////////////
// CResourceViewBar

CPropertiesWnd::CPropertiesWnd()
{
	pGroup1 = new CMFCPropertyGridProperty(_T("PMedia"));
	pGroup2 = new CMFCPropertyGridProperty(_T("Render"));
	pGroup3 = new CMFCPropertyGridProperty(_T("Light"));
	pGroup4 = new CMFCPropertyGridProperty(_T("Volume"));
	pGroup5 = new CMFCPropertyGridProperty(_T("LBFGSB"));
}

CPropertiesWnd::~CPropertiesWnd()
{
}

BEGIN_MESSAGE_MAP(CPropertiesWnd, CDockablePane)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_COMMAND(ID_EXPAND_ALL, OnExpandAllProperties)
	ON_UPDATE_COMMAND_UI(ID_EXPAND_ALL, OnUpdateExpandAllProperties)
	ON_COMMAND(ID_SORTPROPERTIES, OnSortProperties)
	ON_UPDATE_COMMAND_UI(ID_SORTPROPERTIES, OnUpdateSortProperties)
	ON_COMMAND(ID_PROPERTIES1, OnProperties1)
	ON_UPDATE_COMMAND_UI(ID_PROPERTIES1, OnUpdateProperties1)
	ON_COMMAND(ID_PROPERTIES2, OnProperties2)
	ON_UPDATE_COMMAND_UI(ID_PROPERTIES2, OnUpdateProperties2)
	ON_WM_SETFOCUS()
	ON_WM_SETTINGCHANGE()
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CResourceViewBar message handlers

void CPropertiesWnd::AdjustLayout()
{
	if (GetSafeHwnd() == NULL)
	{
		return;
	}

	CRect rectClient,rectCombo;
	GetClientRect(rectClient);
/*
	m_wndObjectCombo.GetWindowRect(&rectCombo);

	int cyCmb = rectCombo.Size().cy;
	int cyTlb = m_wndToolBar.CalcFixedLayout(FALSE, TRUE).cy;

	m_wndObjectCombo.SetWindowPos(NULL, rectClient.left, rectClient.top, rectClient.Width(), 200, SWP_NOACTIVATE | SWP_NOZORDER);
	*/
	m_wndToolBar.SetWindowPos(NULL, rectClient.left, rectClient.top , rectClient.Width(), 0, SWP_NOACTIVATE | SWP_NOZORDER);
	m_wndPropList.SetWindowPos(NULL, rectClient.left, rectClient.top , rectClient.Width(), rectClient.Height(), SWP_NOACTIVATE | SWP_NOZORDER);
}

int CPropertiesWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CDockablePane::OnCreate(lpCreateStruct) == -1)
		return -1;

	CRect rectDummy;
	rectDummy.SetRectEmpty();
/*
	// Create combo:
	const DWORD dwViewStyle = WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_BORDER | CBS_SORT | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;

	if (!m_wndObjectCombo.Create(dwViewStyle, rectDummy, this, 1))
	{
		TRACE0("Failed to create Properties Combo \n");
		return -1;      // fail to create
	}

	m_wndObjectCombo.AddString(_T("Application"));
	m_wndObjectCombo.AddString(_T("Properties Window"));
	m_wndObjectCombo.SetFont(CFont::FromHandle((HFONT) GetStockObject(DEFAULT_GUI_FONT)));
	m_wndObjectCombo.SetCurSel(0);
*/
	if (!m_wndPropList.Create(WS_VISIBLE | WS_CHILD, rectDummy, this, 2))
	{
		TRACE0("Failed to create Properties Grid \n");
		return -1;      // fail to create
	}

	InitPropList();

	m_wndToolBar.Create(this, AFX_DEFAULT_TOOLBAR_STYLE, IDR_PROPERTIES);
	m_wndToolBar.LoadToolBar(IDR_PROPERTIES, 0, 0, TRUE /* Is locked */);
	m_wndToolBar.CleanUpLockedImages();
	m_wndToolBar.LoadBitmap(theApp.m_bHiColorIcons ? IDB_PROPERTIES_HC : IDR_PROPERTIES, 0, 0, TRUE /* Locked */);

	m_wndToolBar.SetPaneStyle(m_wndToolBar.GetPaneStyle() | CBRS_TOOLTIPS | CBRS_FLYBY);
	m_wndToolBar.SetPaneStyle(m_wndToolBar.GetPaneStyle() & ~(CBRS_GRIPPER | CBRS_SIZE_DYNAMIC | CBRS_BORDER_TOP | CBRS_BORDER_BOTTOM | CBRS_BORDER_LEFT | CBRS_BORDER_RIGHT));
	m_wndToolBar.SetOwner(this);

	// All commands will be routed via this control , not via the parent frame:
	m_wndToolBar.SetRouteCommandsViaFrame(FALSE);

	AdjustLayout();
	return 0;
}

void CPropertiesWnd::OnSize(UINT nType, int cx, int cy)
{
	CDockablePane::OnSize(nType, cx, cy);
	AdjustLayout();
}

void CPropertiesWnd::OnExpandAllProperties()
{
	m_wndPropList.ExpandAll();
}

void CPropertiesWnd::OnUpdateExpandAllProperties(CCmdUI* pCmdUI)
{
}

void CPropertiesWnd::OnSortProperties()
{
	m_wndPropList.SetAlphabeticMode(!m_wndPropList.IsAlphabeticMode());
}

void CPropertiesWnd::OnUpdateSortProperties(CCmdUI* pCmdUI)
{
	pCmdUI->SetCheck(m_wndPropList.IsAlphabeticMode());
}

void CPropertiesWnd::OnProperties1()
{
	// TODO: Add your command handler code here
}

void CPropertiesWnd::OnUpdateProperties1(CCmdUI* /*pCmdUI*/)
{
	// TODO: Add your command update UI handler code here
}

void CPropertiesWnd::OnProperties2()
{
	// TODO: Add your command handler code here
}

void CPropertiesWnd::OnUpdateProperties2(CCmdUI* /*pCmdUI*/)
{
	// TODO: Add your command update UI handler code here
}

void CPropertiesWnd::InitPropList()
{
	SetPropListFont();

	m_wndPropList.EnableHeaderCtrl(FALSE);
//	m_wndPropList.EnableDescriptionArea();
	/*
	m_wndPropList.SetVSDotNetLook();
	m_wndPropList.MarkModifiedProperties();
*/

	pGroup1->AddSubItem(new CMFCPropertyGridProperty(_T("extinction"), (_variant_t) theApp.currentXML.PMedia.extinction, _T("指定窗口标题栏中显示的文本")));

	pGroup1->AddSubItem(new CMFCPropertyGridProperty(_T("scattering"), (_variant_t) theApp.currentXML.PMedia.scattering, _T("指定窗口标题栏中显示的文本")));

	pGroup1->AddSubItem(new CMFCPropertyGridProperty(_T("alaph"), (_variant_t) theApp.currentXML.PMedia.alaph, _T("指定窗口标题栏中显示的文本")));

	m_wndPropList.AddProperty(pGroup1);
	////////////////////////////////////////////////////////////////


	pGroup2->AddSubItem(new CMFCPropertyGridProperty(_T("CurrentView"), (_variant_t) theApp.currentXML.Render.CurrentView, _T("指定窗口标题栏中显示的文本")));

	pGroup2->AddSubItem(new CMFCPropertyGridProperty(_T("width"), (_variant_t) theApp.currentXML.Render.width, _T("指定窗口标题栏中显示的文本")));

	pGroup2->AddSubItem(new CMFCPropertyGridProperty(_T("height"), (_variant_t) theApp.currentXML.Render.height, _T("指定窗口标题栏中显示的文本")));

	CMFCPropertyGridProperty* pGroup21 = new CMFCPropertyGridProperty(_T("RenderInterval"));
	pGroup2->AddSubItem(pGroup21);
	pGroup21->AddSubItem(new CMFCPropertyGridProperty(_T("Level5"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level5, _T("此为说明")));
	pGroup21->AddSubItem(new CMFCPropertyGridProperty(_T("Level6"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level6, _T("此为说明")));
	pGroup21->AddSubItem(new CMFCPropertyGridProperty(_T("Level7"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level7, _T("此为说明")));
	pGroup21->AddSubItem(new CMFCPropertyGridProperty(_T("Level8"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level8, _T("此为说明")));
	pGroup21->Expand(true);
	m_wndPropList.AddProperty(pGroup2);
	///////////////////////////////////////////////////


	pGroup3->AddSubItem(new CMFCPropertyGridProperty(_T("LightType"), (_variant_t) theApp.currentXML.Light.LightType, _T("指定窗口标题栏中显示的文本")));

	pGroup3->AddSubItem(new CMFCPropertyGridProperty(_T("LightIntensityR"), (_variant_t) theApp.currentXML.Light.LightIntensityR, _T("指定窗口标题栏中显示的文本")));

	pGroup3->AddSubItem(new CMFCPropertyGridProperty(_T("LightX"), (_variant_t) theApp.currentXML.Light.LightX, _T("指定窗口标题栏中显示的文本")));

	pGroup3->AddSubItem(new CMFCPropertyGridProperty(_T("Lighty"), (_variant_t) theApp.currentXML.Light.LightY, _T("指定窗口标题栏中显示的文本")));

	pGroup3->AddSubItem(new CMFCPropertyGridProperty(_T("Lightz"), (_variant_t) theApp.currentXML.Light.LightZ, _T("指定窗口标题栏中显示的文本")));

	m_wndPropList.AddProperty(pGroup3);
	///////////////////////////////////////////////////////


	pGroup4->AddSubItem(new CMFCPropertyGridProperty(_T("BoxSize"), (_variant_t) theApp.currentXML.Volume.BoxSize, _T("指定窗口标题栏中显示的文本")));

	pGroup4->AddSubItem(new CMFCPropertyGridProperty(_T("TransX"), (_variant_t) theApp.currentXML.Volume.TransX, _T("指定窗口标题栏中显示的文本")));

	pGroup4->AddSubItem(new CMFCPropertyGridProperty(_T("TransY"), (_variant_t) theApp.currentXML.Volume.TransY, _T("指定窗口标题栏中显示的文本")));

	pGroup4->AddSubItem(new CMFCPropertyGridProperty(_T("TransZ"), (_variant_t) theApp.currentXML.Volume.TransZ, _T("指定窗口标题栏中显示的文本")));

	pGroup4->AddSubItem(new CMFCPropertyGridProperty(_T("VolumeInitiallLevel"), (_variant_t) theApp.currentXML.Volume.VolumeInitialLevel, _T("指定窗口标题栏中显示的文本")));

	pGroup4->AddSubItem(new CMFCPropertyGridProperty(_T("VolumeMaxLevel"), (_variant_t) theApp.currentXML.Volume.VolumeMaxLevel, _T("指定窗口标题栏中显示的文本")));

	CMFCPropertyGridProperty* pGroup41 = new CMFCPropertyGridProperty(_T("VolInterval"));

	pGroup4->AddSubItem(pGroup41);
	pGroup41->AddSubItem(new CMFCPropertyGridProperty(_T("Level5"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level5, _T("此为说明")));
	pGroup41->AddSubItem(new CMFCPropertyGridProperty(_T("Level6"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level6, _T("此为说明")));
	pGroup41->AddSubItem(new CMFCPropertyGridProperty(_T("Level7"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level7, _T("此为说明")));
	pGroup41->AddSubItem(new CMFCPropertyGridProperty(_T("Level8"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level8, _T("此为说明")));
	pGroup41->Expand(true);

	m_wndPropList.AddProperty(pGroup4);
	/////////////////////////////////////////////////////



	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("disturb"), (_variant_t) theApp.currentXML.LBFGSB.disturb, _T("指定窗口标题栏中显示的文本")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("EpsG"), (_variant_t) theApp.currentXML.LBFGSB.EpsG, _T("指定窗口标题栏中显示的文本")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("EpsF"), (_variant_t) theApp.currentXML.LBFGSB.EpsF, _T("指定窗口标题栏中显示的文本")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("EpsX"), (_variant_t) theApp.currentXML.LBFGSB.EpsX, _T("指定窗口标题栏中显示的文本")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("MaxIts"), (_variant_t) theApp.currentXML.LBFGSB.MaxIts, _T("指定窗口标题栏中显示的文本")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("m"), (_variant_t) theApp.currentXML.LBFGSB.m, _T("指定窗口标题栏中显示的文本")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("ConstrainType"), (_variant_t) theApp.currentXML.LBFGSB.ConstrainType, _T("指定窗口标题栏中显示的文本")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("LowerBound"), (_variant_t) theApp.currentXML.LBFGSB.LowerBound, _T("指定窗口标题栏中显示的文本")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("UpperBound"), (_variant_t) theApp.currentXML.LBFGSB.UpperBound, _T("指定窗口标题栏中显示的文本")));

	m_wndPropList.AddProperty(pGroup5);
}

void CPropertiesWnd::OnSetFocus(CWnd* pOldWnd)
{
	CDockablePane::OnSetFocus(pOldWnd);
	m_wndPropList.SetFocus();
}

void CPropertiesWnd::OnSettingChange(UINT uFlags, LPCTSTR lpszSection)
{
	CDockablePane::OnSettingChange(uFlags, lpszSection);
	SetPropListFont();
}

void CPropertiesWnd::SetPropListFont()
{
	::DeleteObject(m_fntPropList.Detach());

	LOGFONT lf;
	afxGlobalData.fontRegular.GetLogFont(&lf);

	NONCLIENTMETRICS info;
	info.cbSize = sizeof(info);

	afxGlobalData.GetNonClientMetrics(info);

	lf.lfHeight = info.lfMenuFont.lfHeight;
	lf.lfWeight = info.lfMenuFont.lfWeight;
	lf.lfItalic = info.lfMenuFont.lfItalic;

	m_fntPropList.CreateFontIndirect(&lf);

	m_wndPropList.SetFont(&m_fntPropList);
}

void CPropertiesWnd::reflesh()
{
	pGroup1->GetSubItem(0)->SetValue((_variant_t) theApp.currentXML.PMedia.extinction);
	pGroup1->GetSubItem(1)->SetValue((_variant_t) theApp.currentXML.PMedia.scattering);
	pGroup1->GetSubItem(2)->SetValue((_variant_t) theApp.currentXML.PMedia.alaph);

	////////////////////////////////////////////////////////////////


	pGroup2->GetSubItem(0)->SetValue((_variant_t) theApp.currentXML.Render.CurrentView);
	pGroup2->GetSubItem(1)->SetValue((_variant_t) theApp.currentXML.Render.width);
	pGroup2->GetSubItem(2)->SetValue((_variant_t) theApp.currentXML.Render.height);

	pGroup2->GetSubItem(3)->GetSubItem(0)->SetValue((_variant_t) theApp.currentXML.Render.RenderInterval.Level5);
	pGroup2->GetSubItem(3)->GetSubItem(1)->SetValue((_variant_t) theApp.currentXML.Render.RenderInterval.Level6);
	pGroup2->GetSubItem(3)->GetSubItem(2)->SetValue((_variant_t) theApp.currentXML.Render.RenderInterval.Level7);
	pGroup2->GetSubItem(3)->GetSubItem(3)->SetValue((_variant_t) theApp.currentXML.Render.RenderInterval.Level8);

	///////////////////////////////////////////////////


	pGroup3->GetSubItem(0)->SetValue((_variant_t) theApp.currentXML.Light.LightType);
	pGroup3->GetSubItem(1)->SetValue((_variant_t) theApp.currentXML.Light.LightIntensityR);
	pGroup3->GetSubItem(2)->SetValue((_variant_t) theApp.currentXML.Light.LightX);
	pGroup3->GetSubItem(3)->SetValue((_variant_t) theApp.currentXML.Light.LightY);
	pGroup3->GetSubItem(4)->SetValue((_variant_t) theApp.currentXML.Light.LightZ);

	///////////////////////////////////////////////////////


	pGroup4->GetSubItem(0)->SetValue((_variant_t) theApp.currentXML.Volume.BoxSize);
	pGroup4->GetSubItem(1)->SetValue((_variant_t) theApp.currentXML.Volume.TransX);
	pGroup4->GetSubItem(2)->SetValue((_variant_t) theApp.currentXML.Volume.TransY);
	pGroup4->GetSubItem(3)->SetValue((_variant_t) theApp.currentXML.Volume.TransZ);
	pGroup4->GetSubItem(4)->SetValue((_variant_t) theApp.currentXML.Volume.VolumeInitialLevel);
	pGroup4->GetSubItem(5)->SetValue((_variant_t) theApp.currentXML.Volume.VolumeMaxLevel);

	pGroup4->GetSubItem(6)->GetSubItem(0)->SetValue((_variant_t) theApp.currentXML.Render.RenderInterval.Level5);
	pGroup4->GetSubItem(6)->GetSubItem(1)->SetValue((_variant_t) theApp.currentXML.Render.RenderInterval.Level6);
	pGroup4->GetSubItem(6)->GetSubItem(2)->SetValue((_variant_t) theApp.currentXML.Render.RenderInterval.Level7);
	pGroup4->GetSubItem(6)->GetSubItem(3)->SetValue((_variant_t) theApp.currentXML.Render.RenderInterval.Level8);

	/////////////////////////////////////////////////////



	pGroup5->GetSubItem(0)->SetValue((_variant_t) theApp.currentXML.LBFGSB.disturb);
	pGroup5->GetSubItem(1)->SetValue((_variant_t) theApp.currentXML.LBFGSB.EpsG);
	pGroup5->GetSubItem(2)->SetValue((_variant_t) theApp.currentXML.LBFGSB.EpsF);
	pGroup5->GetSubItem(3)->SetValue((_variant_t) theApp.currentXML.LBFGSB.EpsX);
	pGroup5->GetSubItem(4)->SetValue((_variant_t) theApp.currentXML.LBFGSB.MaxIts);
	pGroup5->GetSubItem(5)->SetValue((_variant_t) theApp.currentXML.LBFGSB.m);
	pGroup5->GetSubItem(6)->SetValue((_variant_t) theApp.currentXML.LBFGSB.ConstrainType);
	pGroup5->GetSubItem(7)->SetValue((_variant_t) theApp.currentXML.LBFGSB.LowerBound);
	pGroup5->GetSubItem(8)->SetValue((_variant_t) theApp.currentXML.LBFGSB.UpperBound);

}