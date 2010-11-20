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

#include "stdafx.h"

#include "PropertiesWnd.h"
#include "Resource.h"
#include "MainFrm.h"
#include "test.h"

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
// CResourceViewBar ��Ϣ�������

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
	m_wndToolBar.SetWindowPos(NULL, rectClient.left, rectClient.top, rectClient.Width(), 0, SWP_NOACTIVATE | SWP_NOZORDER);
	m_wndPropList.SetWindowPos(NULL, rectClient.left, rectClient.top  , rectClient.Width(), rectClient.Height(), SWP_NOACTIVATE | SWP_NOZORDER);
}

int CPropertiesWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CDockablePane::OnCreate(lpCreateStruct) == -1)
		return -1;

	CRect rectDummy;
	rectDummy.SetRectEmpty();

	// �������:
	const DWORD dwViewStyle = WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_BORDER | CBS_SORT | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;

/*
	if (!m_wndObjectCombo.Create(dwViewStyle, rectDummy, this, 1))
	{
		TRACE0("δ�ܴ���������� \n");
		return -1;      // δ�ܴ���
	}

	m_wndObjectCombo.AddString(_T("Ӧ�ó���"));
	m_wndObjectCombo.AddString(_T("���Դ���"));
	m_wndObjectCombo.SetCurSel(0);
*/
	if (!m_wndPropList.Create(WS_VISIBLE | WS_CHILD, rectDummy, this, 2))
	{
		TRACE0("δ�ܴ�����������\n");
		return -1;      // δ�ܴ���
	}

	InitPropList();

	m_wndToolBar.Create(this, AFX_DEFAULT_TOOLBAR_STYLE, IDR_PROPERTIES);
	m_wndToolBar.LoadToolBar(IDR_PROPERTIES, 0, 0, TRUE /* ������*/);
	m_wndToolBar.CleanUpLockedImages();
	m_wndToolBar.LoadBitmap(theApp.m_bHiColorIcons ? IDB_PROPERTIES_HC : IDR_PROPERTIES, 0, 0, TRUE /* ����*/);

	m_wndToolBar.SetPaneStyle(m_wndToolBar.GetPaneStyle() | CBRS_TOOLTIPS | CBRS_FLYBY);
	m_wndToolBar.SetPaneStyle(m_wndToolBar.GetPaneStyle() & ~(CBRS_GRIPPER | CBRS_SIZE_DYNAMIC | CBRS_BORDER_TOP | CBRS_BORDER_BOTTOM | CBRS_BORDER_LEFT | CBRS_BORDER_RIGHT));
	m_wndToolBar.SetOwner(this);

	

	// �������ͨ���˿ؼ�·�ɣ�������ͨ�������·��:
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

void CPropertiesWnd::OnUpdateExpandAllProperties(CCmdUI* /* pCmdUI */)
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
	// TODO: �ڴ˴���������������
}

void CPropertiesWnd::OnUpdateProperties1(CCmdUI* /*pCmdUI*/)
{
	// TODO: �ڴ˴����������� UI ����������
}

void CPropertiesWnd::OnProperties2()
{
	// TODO: �ڴ˴���������������
}

void CPropertiesWnd::OnUpdateProperties2(CCmdUI* /*pCmdUI*/)
{
	// TODO: �ڴ˴����������� UI ����������
}

void CPropertiesWnd::InitPropList()
{
	SetPropListFont();

	m_wndPropList.EnableHeaderCtrl(FALSE);
	m_wndPropList.EnableDescriptionArea(FALSE);
//	m_wndPropList.SetVSDotNetLook();
//	m_wndPropList.MarkModifiedProperties();

	

	pGroup1->AddSubItem(new CMFCPropertyGridProperty(_T("extinction"), (_variant_t) theApp.currentXML.PMedia.extinction, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup1->AddSubItem(new CMFCPropertyGridProperty(_T("scattering"), (_variant_t) theApp.currentXML.PMedia.scattering, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup1->AddSubItem(new CMFCPropertyGridProperty(_T("alaph"), (_variant_t) theApp.currentXML.PMedia.alaph, _T("ָ�����ڱ���������ʾ���ı�")));

	m_wndPropList.AddProperty(pGroup1);
////////////////////////////////////////////////////////////////


	pGroup2->AddSubItem(new CMFCPropertyGridProperty(_T("CurrentView"), (_variant_t) theApp.currentXML.Render.CurrentView, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup2->AddSubItem(new CMFCPropertyGridProperty(_T("width"), (_variant_t) theApp.currentXML.Render.width, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup2->AddSubItem(new CMFCPropertyGridProperty(_T("height"), (_variant_t) theApp.currentXML.Render.height, _T("ָ�����ڱ���������ʾ���ı�")));

	CMFCPropertyGridProperty* pGroup21 = new CMFCPropertyGridProperty(_T("RenderInterval"));
	pGroup2->AddSubItem(pGroup21);
	pGroup21->AddSubItem(new CMFCPropertyGridProperty(_T("Level5"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level5, _T("��Ϊ˵��")));
	pGroup21->AddSubItem(new CMFCPropertyGridProperty(_T("Level6"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level6, _T("��Ϊ˵��")));
	pGroup21->AddSubItem(new CMFCPropertyGridProperty(_T("Level7"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level7, _T("��Ϊ˵��")));
	pGroup21->AddSubItem(new CMFCPropertyGridProperty(_T("Level8"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level8, _T("��Ϊ˵��")));
	pGroup21->Expand(true);
	m_wndPropList.AddProperty(pGroup2);
///////////////////////////////////////////////////
	

	pGroup3->AddSubItem(new CMFCPropertyGridProperty(_T("LightType"), (_variant_t) theApp.currentXML.Light.LightType, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup3->AddSubItem(new CMFCPropertyGridProperty(_T("LightIntensityR"), (_variant_t) theApp.currentXML.Light.LightIntensityR, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup3->AddSubItem(new CMFCPropertyGridProperty(_T("LightX"), (_variant_t) theApp.currentXML.Light.LightX, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup3->AddSubItem(new CMFCPropertyGridProperty(_T("Lighty"), (_variant_t) theApp.currentXML.Light.LightY, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup3->AddSubItem(new CMFCPropertyGridProperty(_T("Lightz"), (_variant_t) theApp.currentXML.Light.LightZ, _T("ָ�����ڱ���������ʾ���ı�")));

	m_wndPropList.AddProperty(pGroup3);
///////////////////////////////////////////////////////


	pGroup4->AddSubItem(new CMFCPropertyGridProperty(_T("BoxSize"), (_variant_t) theApp.currentXML.Volume.BoxSize, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup4->AddSubItem(new CMFCPropertyGridProperty(_T("TransX"), (_variant_t) theApp.currentXML.Volume.TransX, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup4->AddSubItem(new CMFCPropertyGridProperty(_T("TransY"), (_variant_t) theApp.currentXML.Volume.TransY, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup4->AddSubItem(new CMFCPropertyGridProperty(_T("TransZ"), (_variant_t) theApp.currentXML.Volume.TransZ, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup4->AddSubItem(new CMFCPropertyGridProperty(_T("VolumeInitiallLevel"), (_variant_t) theApp.currentXML.Volume.VolumeInitialLevel, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup4->AddSubItem(new CMFCPropertyGridProperty(_T("VolumeMaxLevel"), (_variant_t) theApp.currentXML.Volume.VolumeMaxLevel, _T("ָ�����ڱ���������ʾ���ı�")));

	CMFCPropertyGridProperty* pGroup41 = new CMFCPropertyGridProperty(_T("VolInterval"));

	pGroup4->AddSubItem(pGroup41);
	pGroup41->AddSubItem(new CMFCPropertyGridProperty(_T("Level5"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level5, _T("��Ϊ˵��")));
	pGroup41->AddSubItem(new CMFCPropertyGridProperty(_T("Level6"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level6, _T("��Ϊ˵��")));
	pGroup41->AddSubItem(new CMFCPropertyGridProperty(_T("Level7"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level7, _T("��Ϊ˵��")));
	pGroup41->AddSubItem(new CMFCPropertyGridProperty(_T("Level8"), (_variant_t) theApp.currentXML.Render.RenderInterval.Level8, _T("��Ϊ˵��")));
	pGroup41->Expand(true);

	m_wndPropList.AddProperty(pGroup4);
	/////////////////////////////////////////////////////



	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("disturb"), (_variant_t) theApp.currentXML.LBFGSB.disturb, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("EpsG"), (_variant_t) theApp.currentXML.LBFGSB.EpsG, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("EpsF"), (_variant_t) theApp.currentXML.LBFGSB.EpsF, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("EpsX"), (_variant_t) theApp.currentXML.LBFGSB.EpsX, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("MaxIts"), (_variant_t) theApp.currentXML.LBFGSB.MaxIts, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("m"), (_variant_t) theApp.currentXML.LBFGSB.m, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("ConstrainType"), (_variant_t) theApp.currentXML.LBFGSB.ConstrainType, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("LowerBound"), (_variant_t) theApp.currentXML.LBFGSB.LowerBound, _T("ָ�����ڱ���������ʾ���ı�")));

	pGroup5->AddSubItem(new CMFCPropertyGridProperty(_T("UpperBound"), (_variant_t) theApp.currentXML.LBFGSB.UpperBound, _T("ָ�����ڱ���������ʾ���ı�")));

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
//	m_wndObjectCombo.SetFont(&m_fntPropList);
}

void CPropertiesWnd::reflesh()
{
	MSXML2::IXMLDOMDocumentPtr pDoc;
	HRESULT hr;
	hr=pDoc.CreateInstance(__uuidof(MSXML2::DOMDocument40));
	if(FAILED(hr))
	{ 
		AfxMessageBox(_T("�޷�����DOMDocument���������Ƿ�װ��MS XML Parser ���п�!")); 
	} 

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