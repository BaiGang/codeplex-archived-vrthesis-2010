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

// test.cpp : 定义应用程序的类行为。
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
	// 基于文件的标准文档命令
	ON_COMMAND(ID_FILE_NEW, &CWinAppEx::OnFileNew)
	ON_COMMAND(ID_FILE_OPEN, &CWinAppEx::OnFileOpen)
	ON_COMMAND(ID_ROOT, &CtestApp::OnSetRoot)
	ON_COMMAND(ID_BUTTON6, &CtestApp::OnButtonLastPic)
	ON_COMMAND(ID_BUTTON7, &CtestApp::OnButtonNextPic)
	ON_COMMAND(ID_BUTTON_EXE, &CtestApp::OnButtonExe)
END_MESSAGE_MAP()


// CtestApp 构造

CtestApp::CtestApp()
{
	m_bHiColorIcons = TRUE;

	// 支持重新启动管理器
	m_dwRestartManagerSupportFlags = AFX_RESTART_MANAGER_SUPPORT_ALL_ASPECTS;
#ifdef _MANAGED
	// 如果应用程序是利用公共语言运行时支持(/clr)构建的，则:
	//     1) 必须有此附加设置，“重新启动管理器”支持才能正常工作。
	//     2) 在您的项目中，您必须按照生成顺序向 System.Windows.Forms 添加引用。
	System::Windows::Forms::Application::SetUnhandledExceptionMode(System::Windows::Forms::UnhandledExceptionMode::ThrowException);
#endif

	// TODO: 将以下应用程序 ID 字符串替换为唯一的 ID 字符串；建议的字符串格式
	//为 CompanyName.ProductName.SubProduct.VersionInformation
	SetAppID(_T("test.AppID.NoVersion"));

	// TODO: 在此处添加构造代码，
	// 将所有重要的初始化放置在 InitInstance 中
}

// 唯一的一个 CtestApp 对象

CtestApp theApp;


// CtestApp 初始化

BOOL CtestApp::InitInstance()
{
	// 如果一个运行在 Windows XP 上的应用程序清单指定要
	// 使用 ComCtl32.dll 版本 6 或更高版本来启用可视化方式，
	//则需要 InitCommonControlsEx()。否则，将无法创建窗口。
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	// 将它设置为包括所有要在应用程序中使用的
	// 公共控件类。
	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);

	CWinAppEx::InitInstance();


	// 初始化 OLE 库
	if (!AfxOleInit())
	{
		AfxMessageBox(IDP_OLE_INIT_FAILED);
		return FALSE;
	}

	EnableTaskbarInteraction(FALSE);

	// 使用 RichEdit 控件需要  AfxInitRichEdit2()	
	// AfxInitRichEdit2();

	// 标准初始化
	// 如果未使用这些功能并希望减小
	// 最终可执行文件的大小，则应移除下列
	// 不需要的特定初始化例程
	// 更改用于存储设置的注册表项
	// TODO: 应适当修改该字符串，
	// 例如修改为公司或组织名
	SetRegistryKey(_T("应用程序向导生成的本地应用程序"));
	LoadStdProfileSettings(4);  // 加载标准 INI 文件选项(包括 MRU)


	InitContextMenuManager();

	InitKeyboardManager();

	InitTooltipManager();
	CMFCToolTipInfo ttParams;
	ttParams.m_bVislManagerTheme = TRUE;
	theApp.GetTooltipManager()->SetTooltipParams(AFX_TOOLTIP_TYPE_ALL,
		RUNTIME_CLASS(CMFCToolTipCtrl), &ttParams);

	// 注册应用程序的文档模板。文档模板
	// 将用作文档、框架窗口和视图之间的连接
	CSingleDocTemplate* pDocTemplate;
	pDocTemplate = new CSingleDocTemplate(
		IDR_MAINFRAME,
		RUNTIME_CLASS(CtestDoc),
		RUNTIME_CLASS(CMainFrame),       // 主 SDI 框架窗口
		RUNTIME_CLASS(CPmModelTool));
	if (!pDocTemplate)
		return FALSE;
	AddDocTemplate(pDocTemplate);


	// 分析标准 shell 命令、DDE、打开文件操作的命令行
	CCommandLineInfo cmdInfo;
	ParseCommandLine(cmdInfo);



	// 调度在命令行中指定的命令。如果
	// 用 /RegServer、/Register、/Unregserver 或 /Unregister 启动应用程序，则返回 FALSE。
	if (!ProcessShellCommand(cmdInfo))
		return FALSE;

	// 唯一的一个窗口已初始化，因此显示它并对其进行更新
	m_pMainWnd->ShowWindow(SW_SHOW);
	m_pMainWnd->UpdateWindow();
	// 仅当具有后缀时才调用 DragAcceptFiles
	//  在 SDI 应用程序中，这应在 ProcessShellCommand 之后发生


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
	//TODO: 处理可能已添加的附加资源
	AfxOleTerm(FALSE);

	CWnd * hwnd;
	hwnd = (CWnd*)FindWindow(NULL, _T("Reconstructed Smoke Volume Result."));

	if(hwnd)
	{
		// here to add code to exit exe
//		hwnd->CloseWindow();  //this program could not exit as normal, so this function works bad = =
	}

	return CWinAppEx::ExitInstance();
}

// CtestApp 消息处理程序


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
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

// 用于运行对话框的应用程序命令
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


// CtestApp 自定义加载/保存方法

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

// CtestApp 消息处理程序





void CtestApp::OnButtonLastPic()
{
	// TODO: 在此添加命令处理程序代码

}


void CtestApp::OnButtonNextPic()
{
	// TODO: 在此添加命令处理程序代码
}

void CtestApp::loadPic()
{
	// TODO: 在此添加命令处理程序代码
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
		AfxMessageBox(_T("无法创建DOMDocument对象，请检查是否安装了MS XML Parser 运行库!")); 
		return;
	} 

	//加载文件 
	CString filepath;
	filepath = rootPath + _T("\\data\\configure.xml");
	if(pDoc->load(filepath.GetBuffer(0)) == false)
	{
		AfxMessageBox(_T("Failed when read \\data\\configure.xml"));
		return;
	}

	MSXML2::IXMLDOMNodePtr pNode;

	//在树中查找名为PMedia的节点,"//"表示在任意一层查找 
	pNode=pDoc->selectSingleNode("//Parameters");

	//节点数据 
	CString strData;

	//节点属性,放在链表中 
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
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.PMedia.extinction = strData;

	m_pNodet = m_pNodeList->item[1];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.PMedia.scattering = strData;

	m_pNodet = m_pNodeList->item[2];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.PMedia.alaph = strData;
////////////////////////////// Render ////////////////////////
	
	pNodet = pNodeList->item[1];
	m_pNodeList = pNodet->childNodes;


	m_pNodet = m_pNodeList->item[0];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.CurrentView = strData;

	m_pNodet = m_pNodeList->item[1];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.width = strData;

	m_pNodet = m_pNodeList->item[2];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.height = strData;
////////////////////////////// Render ////////////////////////////
//////////////////////////////////////// IntervalLevel //////////
	m_pNodet = m_pNodeList->item[3];
	m_pNodeList = m_pNodet->childNodes;

	m_pNodet = m_pNodeList->item[0];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.RenderInterval.Level5 = strData;

	m_pNodet = m_pNodeList->item[1];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.RenderInterval.Level6 = strData;

	m_pNodet = m_pNodeList->item[2];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.RenderInterval.Level7 = strData;

	m_pNodet = m_pNodeList->item[3];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.RenderInterval.Level8 = strData;
///////////////////////////////// Render //////////////////////////
////////////////////////////////////////// IntervalLevel End //////
	pNodet = pNodeList->item[1];
	m_pNodeList = pNodet->childNodes;

	m_pNodet = m_pNodeList->item[4];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Render.RotAngle = strData;

////////////////////////////// Light ////////////////////////
	pNodet = pNodeList->item[2];
	m_pNodeList = pNodet->childNodes;


	m_pNodet = m_pNodeList->item[0];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Light.LightType = strData;

	m_pNodet = m_pNodeList->item[1];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Light.LightIntensityR = strData;

	m_pNodet = m_pNodeList->item[2];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Light.LightX = strData;

	m_pNodet = m_pNodeList->item[3];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Light.LightY = strData;

	m_pNodet = m_pNodeList->item[4];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Light.LightZ = strData;

////////////////////////////// Volume ////////////////////////
	pNodet = pNodeList->item[3];
	m_pNodeList = pNodet->childNodes;


	m_pNodet = m_pNodeList->item[3];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.BoxSize = strData;

	m_pNodet = m_pNodeList->item[4];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.TransX = strData;

	m_pNodet = m_pNodeList->item[5];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.TransY = strData;

	m_pNodet = m_pNodeList->item[6];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.TransZ = strData;

	m_pNodet = m_pNodeList->item[8];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.VolumeInitialLevel = strData;

	m_pNodet = m_pNodeList->item[9];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.VolumeMaxLevel = strData;
////////////////////////////// Volume ////////////////////////////
//////////////////////////////////////// IntervalLevel //////////
	m_pNodet = m_pNodeList->item[10];
	m_pNodeList = m_pNodet->childNodes;

	m_pNodet = m_pNodeList->item[0];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.VolInterval.Level5 = strData;

	m_pNodet = m_pNodeList->item[1];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.VolInterval.Level6 = strData;

	m_pNodet = m_pNodeList->item[2];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.Volume.VolInterval.Level7 = strData;

	m_pNodet = m_pNodeList->item[3];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
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
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.disturb = strData;

	m_pNodet = m_pNodeList->item[8];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.EpsG = strData;

	m_pNodet = m_pNodeList->item[9];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.EpsF = strData;

	m_pNodet = m_pNodeList->item[10];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.EpsX = strData;

	m_pNodet = m_pNodeList->item[11];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.MaxIts = strData;

	m_pNodet = m_pNodeList->item[12];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.m = strData;

	m_pNodet = m_pNodeList->item[13];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.ConstrainType = strData;

	m_pNodet = m_pNodeList->item[14];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.LowerBound = strData;

	m_pNodet = m_pNodeList->item[15];
	m_pNodet->get_attributes(&pAttrMap);
	//获得第0个属性
	pAttrMap->get_item(0,&pAttrItem);
	//取得节点的值
	pAttrItem->get_nodeTypedValue(&variantvalue);
	strData = (char *)(_bstr_t)variantvalue;
	currentXML.LBFGSB.UpperBound = strData;
}

void CtestApp::OnButtonExe()
{
	// TODO: 在此添加命令处理程序代码
	WinExec("ASRendering.exe",SW_SHOW);
	Sleep(1000);
	CMainFrame   *pMain=(CMainFrame   *)AfxGetApp()->m_pMainWnd;   
    CPmModelTool   *pView=(CPmModelTool   *)pMain->GetActiveView();
	pView->setExePos();
}
