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

// test.cpp : ����Ӧ�ó��������Ϊ��
//

#include "stdafx.h"
#include "afxwinappex.h"
#include "afxdialogex.h"
#include "test.h"
#include "MainFrm.h"

#include "testDoc.h"
//#include "testView.h"
#include "pmModelTool.h"

#include "INPUTDLG.h"



#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CtestApp

BEGIN_MESSAGE_MAP(CtestApp, CWinAppEx)
	ON_COMMAND(ID_APP_ABOUT, &CtestApp::OnAppAbout)
	// �����ļ��ı�׼�ĵ�����
	ON_COMMAND(ID_FILE_NEW, &CWinAppEx::OnFileNew)
	ON_COMMAND(ID_FILE_OPEN, &CWinAppEx::OnFileOpen)
	ON_COMMAND(ID_ROOT, &CtestApp::OnSetRoot)
	ON_COMMAND(ID_BUTTON6, &CtestApp::OnButtonLastPic)
	ON_COMMAND(ID_BUTTON7, &CtestApp::OnButtonNextPic)
END_MESSAGE_MAP()


// CtestApp ����

CtestApp::CtestApp()
{
	m_bHiColorIcons = TRUE;

	// ֧����������������
	m_dwRestartManagerSupportFlags = AFX_RESTART_MANAGER_SUPPORT_ALL_ASPECTS;
#ifdef _MANAGED
	// ���Ӧ�ó��������ù�����������ʱ֧��(/clr)�����ģ���:
	//     1) �����д˸������ã�������������������֧�ֲ�������������
	//     2) ��������Ŀ�У������밴������˳���� System.Windows.Forms ������á�
	System::Windows::Forms::Application::SetUnhandledExceptionMode(System::Windows::Forms::UnhandledExceptionMode::ThrowException);
#endif

	// TODO: ������Ӧ�ó��� ID �ַ����滻ΪΨһ�� ID �ַ�����������ַ�����ʽ
	//Ϊ CompanyName.ProductName.SubProduct.VersionInformation
	SetAppID(_T("test.AppID.NoVersion"));

	// TODO: �ڴ˴���ӹ�����룬
	// ��������Ҫ�ĳ�ʼ�������� InitInstance ��
}

// Ψһ��һ�� CtestApp ����

CtestApp theApp;


// CtestApp ��ʼ��

BOOL CtestApp::InitInstance()
{
	// ���һ�������� Windows XP �ϵ�Ӧ�ó����嵥ָ��Ҫ
	// ʹ�� ComCtl32.dll �汾 6 ����߰汾�����ÿ��ӻ���ʽ��
	//����Ҫ InitCommonControlsEx()�����򣬽��޷��������ڡ�
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	// ��������Ϊ��������Ҫ��Ӧ�ó�����ʹ�õ�
	// �����ؼ��ࡣ
	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);

	CWinAppEx::InitInstance();


	// ��ʼ�� OLE ��
	if (!AfxOleInit())
	{
		AfxMessageBox(IDP_OLE_INIT_FAILED);
		return FALSE;
	}

	EnableTaskbarInteraction(FALSE);

	// ʹ�� RichEdit �ؼ���Ҫ  AfxInitRichEdit2()	
	// AfxInitRichEdit2();

	// ��׼��ʼ��
	// ���δʹ����Щ���ܲ�ϣ����С
	// ���տ�ִ���ļ��Ĵ�С����Ӧ�Ƴ�����
	// ����Ҫ���ض���ʼ������
	// �������ڴ洢���õ�ע�����
	// TODO: Ӧ�ʵ��޸ĸ��ַ�����
	// �����޸�Ϊ��˾����֯��
	SetRegistryKey(_T("Ӧ�ó��������ɵı���Ӧ�ó���"));
	LoadStdProfileSettings(4);  // ���ر�׼ INI �ļ�ѡ��(���� MRU)


	InitContextMenuManager();

	InitKeyboardManager();

	InitTooltipManager();
	CMFCToolTipInfo ttParams;
	ttParams.m_bVislManagerTheme = TRUE;
	theApp.GetTooltipManager()->SetTooltipParams(AFX_TOOLTIP_TYPE_ALL,
		RUNTIME_CLASS(CMFCToolTipCtrl), &ttParams);

	// ע��Ӧ�ó�����ĵ�ģ�塣�ĵ�ģ��
	// �������ĵ�����ܴ��ں���ͼ֮�������
	CSingleDocTemplate* pDocTemplate;
	pDocTemplate = new CSingleDocTemplate(
		IDR_MAINFRAME,
		RUNTIME_CLASS(CtestDoc),
		RUNTIME_CLASS(CMainFrame),       // �� SDI ��ܴ���
		RUNTIME_CLASS(CPmModelTool));
	if (!pDocTemplate)
		return FALSE;
	AddDocTemplate(pDocTemplate);


	// ������׼ shell ���DDE�����ļ�������������
	CCommandLineInfo cmdInfo;
	ParseCommandLine(cmdInfo);



	// ��������������ָ����������
	// �� /RegServer��/Register��/Unregserver �� /Unregister ����Ӧ�ó����򷵻� FALSE��
	if (!ProcessShellCommand(cmdInfo))
		return FALSE;

	// Ψһ��һ�������ѳ�ʼ���������ʾ����������и���
	m_pMainWnd->ShowWindow(SW_SHOW);
	m_pMainWnd->UpdateWindow();
	// �������к�׺ʱ�ŵ��� DragAcceptFiles
	//  �� SDI Ӧ�ó����У���Ӧ�� ProcessShellCommand ֮����


	//picIndex = 0;
	////assume
	//float length = 320;
	//float * imgdata1 = new float [length * length*length];
	//float * imgdata2 = new float [length * length*length];
	//float * imgdata3 = new float [length * length*length];
	//lastPic =  new PFMImage(length, length*length, 0, imgdata1);
	//currentPic = new PFMImage(length, length*length, 0, imgdata2);
	//nextPic = new PFMImage(length, length*length, 0, imgdata3);


	return TRUE;
}

int CtestApp::ExitInstance()
{
	//TODO: �����������ӵĸ�����Դ
	AfxOleTerm(FALSE);

	return CWinAppEx::ExitInstance();
}

// CtestApp ��Ϣ�������


// ����Ӧ�ó��򡰹��ڡ��˵���� CAboutDlg �Ի���

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// �Ի�������
	enum { IDD = IDD_ABOUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

// ʵ��
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnComboView();
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()

// �������жԻ����Ӧ�ó�������
void CtestApp::OnAppAbout()
{
	CAboutDlg aboutDlg;
	aboutDlg.DoModal();
	
}

void CtestApp::OnSetRoot()
{
	INPUTDLG inDlg;
	inDlg.DoModal();
}


// CtestApp �Զ������/���淽��

void CtestApp::PreLoadState()
{
	BOOL bNameValid;
	CString strName;
	bNameValid = strName.LoadString(IDS_EDIT_MENU);
	ASSERT(bNameValid);
	GetContextMenuManager()->AddMenu(strName, IDR_POPUP_EDIT);
}

void CtestApp::LoadCustomState()
{
}

void CtestApp::SaveCustomState()
{
}

// CtestApp ��Ϣ�������





void CtestApp::OnButtonLastPic()
{
	// TODO: �ڴ���������������

}


void CtestApp::OnButtonNextPic()
{
	// TODO: �ڴ���������������
}

void CtestApp::loadPic()
{
	// TODO: �ڴ���������������
	CString filestr;
	filestr.Format(_T("%d"), picIndex);
	currentPic->ReadImage(cstring2char(filestr));
}

char* CtestApp::cstring2char (CString cstr)
{
	int len = WideCharToMultiByte(CP_UTF8, 0, cstr.AllocSysString(), -1, NULL, 0, NULL, NULL);  
	char * szUtf8=new char[len + 1];
	WideCharToMultiByte (CP_UTF8, 0, cstr.AllocSysString(), -1, szUtf8, len, NULL,NULL);
	return szUtf8;
}

void CtestApp::loadXml()
{
	if(FAILED(::CoInitialize(NULL)))  
	{
		AfxMessageBox(_T("failed"));
	}

	MSXML2::IXMLDOMDocumentPtr pDoc;
	HRESULT hr;
	hr=pDoc.CreateInstance(__uuidof(MSXML2::DOMDocument40));
	if(FAILED(hr))
	{ 
		AfxMessageBox(_T("�޷�����DOMDocument���������Ƿ�װ��MS XML Parser ���п�!")); 
		return;
	} 

	//�����ļ� 
	CString filepath;
	filepath = rootPath + _T("\\data\\configure.xml");
	if(pDoc->load(filepath.GetBuffer(0)) == false)
	{
		AfxMessageBox(_T("Failed when read \\data\\configure.xml"));
		return;
	}

	MSXML2::IXMLDOMNodePtr pNode;

	//�����в�����ΪPMedia�Ľڵ�,"//"��ʾ������һ����� 
	pNode=pDoc->selectSingleNode("//Parameters");

	//�ڵ����� 
	CString strData;

	//�ڵ�����,���������� 
	MSXML2::IXMLDOMNamedNodeMapPtr pAttrMap=NULL;
	MSXML2::IXMLDOMNodePtr   pAttrItem;
	_variant_t variantvalue;

	MSXML2::IXMLDOMNodeListPtr pNodeList;
	MSXML2::IXMLDOMNodeListPtr m_pNodeList;
	MSXML2::IXMLDOMNodePtr pNodet;
	MSXML2::IXMLDOMNodePtr m_pNodet;
	pNodeList = pNode->childNodes;
////////////////////////////// PMedia /////////////////////////
	pNodet = pNodeList->item[0];
	m_pNodeList = pNodet->childNodes;


	m_pNodet = m_pNodeList->item[0];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.PMedia.extinction = strData;

	m_pNodet = m_pNodeList->item[1];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.PMedia.scattering = strData;

	m_pNodet = m_pNodeList->item[2];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.PMedia.alaph = strData;
////////////////////////////// Render ////////////////////////
	
	pNodet = pNodeList->item[1];
	m_pNodeList = pNodet->childNodes;


	m_pNodet = m_pNodeList->item[0];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.CurrentView = strData;

	m_pNodet = m_pNodeList->item[1];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.width = strData;

	m_pNodet = m_pNodeList->item[2];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.height = strData;
////////////////////////////// Render ////////////////////////////
//////////////////////////////////////// IntervalLevel //////////
	m_pNodet = m_pNodeList->item[3];
	m_pNodeList = m_pNodet->childNodes;

	m_pNodet = m_pNodeList->item[0];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.RenderInterval.Level5 = strData;

	m_pNodet = m_pNodeList->item[1];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.RenderInterval.Level6 = strData;

	m_pNodet = m_pNodeList->item[2];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.RenderInterval.Level7 = strData;

	m_pNodet = m_pNodeList->item[3];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.RenderInterval.Level8 = strData;
///////////////////////////////// Render //////////////////////////
////////////////////////////////////////// IntervalLevel End //////
	pNodet = pNodeList->item[1];
	m_pNodeList = pNodet->childNodes;

	m_pNodet = m_pNodeList->item[4];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.RotAngle = strData;

////////////////////////////// Light ////////////////////////
	pNodet = pNodeList->item[2];
	m_pNodeList = pNodet->childNodes;


	m_pNodet = m_pNodeList->item[0];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Light.LightType = strData;

	m_pNodet = m_pNodeList->item[1];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Light.LightIntensityR = strData;

	m_pNodet = m_pNodeList->item[2];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Light.LightX = strData;

	m_pNodet = m_pNodeList->item[3];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Light.LightY = strData;

	m_pNodet = m_pNodeList->item[4];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Light.LightZ = strData;

////////////////////////////// Volume ////////////////////////
	pNodet = pNodeList->item[3];
	m_pNodeList = pNodet->childNodes;


	m_pNodet = m_pNodeList->item[3];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.BoxSize = strData;

	m_pNodet = m_pNodeList->item[4];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.TransX = strData;

	m_pNodet = m_pNodeList->item[5];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.TransY = strData;

	m_pNodet = m_pNodeList->item[6];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.TransZ = strData;

	m_pNodet = m_pNodeList->item[8];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.VolumeInitialLevel = strData;

	m_pNodet = m_pNodeList->item[9];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.VolumeMaxLevel = strData;
////////////////////////////// Volume ////////////////////////////
//////////////////////////////////////// IntervalLevel //////////
	m_pNodet = m_pNodeList->item[10];
	m_pNodeList = m_pNodet->childNodes;

	m_pNodet = m_pNodeList->item[0];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.VolInterval.Level5 = strData;

	m_pNodet = m_pNodeList->item[1];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.VolInterval.Level6 = strData;

	m_pNodet = m_pNodeList->item[2];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.VolInterval.Level7 = strData;

	m_pNodet = m_pNodeList->item[3];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.VolInterval.Level8 = strData;
///////////////////////////////// Volume //////////////////////////
////////////////////////////////////////// IntervalLevel End //////

////////////////////////////// LBFGSB ////////////////////////
	pNodet = pNodeList->item[4];
	m_pNodeList = pNodet->childNodes;


	m_pNodet = m_pNodeList->item[7];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.disturb = strData;

	m_pNodet = m_pNodeList->item[8];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.EpsG = strData;

	m_pNodet = m_pNodeList->item[9];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.EpsF = strData;

	m_pNodet = m_pNodeList->item[10];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.EpsX = strData;

	m_pNodet = m_pNodeList->item[11];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.MaxIts = strData;

	m_pNodet = m_pNodeList->item[12];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.m = strData;

	m_pNodet = m_pNodeList->item[13];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.ConstrainType = strData;

	m_pNodet = m_pNodeList->item[14];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.LowerBound = strData;

	m_pNodet = m_pNodeList->item[15];
	m_pNodet->get_attributes(&pAttrMap);
	//��õ�0������
	pAttrMap->get_item(0,&pAttrItem);
	//ȡ�ýڵ��ֵ
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.UpperBound = strData;
}