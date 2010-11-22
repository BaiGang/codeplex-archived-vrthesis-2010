// INPUTDLG.cpp : implementation file
//

#include "stdafx.h"
#include "asmodeling.h"
#include "INPUTDLG.h"


// INPUTDLG dialog

IMPLEMENT_DYNAMIC(INPUTDLG, CDialog)

INPUTDLG::INPUTDLG(CWnd* pParent /*=NULL*/)
	: CDialog(INPUTDLG::IDD, pParent)
{

}

INPUTDLG::~INPUTDLG()
{
}

void INPUTDLG::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_EDIT_ROOTPATH, mPath);
}


BEGIN_MESSAGE_MAP(INPUTDLG, CDialog)
	ON_BN_CLICKED(IDOK, &INPUTDLG::OnBnClickedOk)
	ON_BN_CLICKED(IDC_BUTTON_BROWSE, &INPUTDLG::OnBnClickedButtonBrowse)
END_MESSAGE_MAP()


// INPUTDLG message handlers
void INPUTDLG::OnBnClickedOk()
{
	// TODO: 在此添加控件通知处理程序代码
	mPath.GetWindowText(theApp.rootPath);

	CDialog::OnOK();
}


void INPUTDLG::OnBnClickedButtonBrowse()
{
	// TODO: 在此添加控件通知处理程序代码
	rootPath = GetDirPath();
	mPath.SetWindowText(rootPath);
}

CString  INPUTDLG::GetDirPath()
{
	CString strPath;
	BROWSEINFO bInfo;
	ZeroMemory(&bInfo, sizeof(bInfo));
	//    bInfo.hwndOwner = this;
	bInfo.lpszTitle = _T("请选择存放数据的文件夹: ");
	bInfo.ulFlags = BIF_RETURNONLYFSDIRS;    

	LPITEMIDLIST lpDlist; //用来保存返回信息的IDList
	lpDlist = SHBrowseForFolder(&bInfo) ; //显示选择对话框
	if(lpDlist != NULL)  //用户按了确定按钮
	{
		TCHAR chPath[255]; //用来存储路径的字符串
		SHGetPathFromIDList(lpDlist, chPath);//把项目标识列表转化成字符串
		strPath = chPath; //将TCHAR类型的字符串转换为CString类型的字符串
	}
	return strPath;
}