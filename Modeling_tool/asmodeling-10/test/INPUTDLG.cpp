// INPUTDLG.cpp : 实现文件
//

#include "stdafx.h"
#include "test.h"
#include "INPUTDLG.h"
#include "afxdialogex.h"


// INPUTDLG 对话框

IMPLEMENT_DYNAMIC(INPUTDLG, CDialogEx)

INPUTDLG::INPUTDLG(CWnd* pParent /*=NULL*/)
	: CDialogEx(INPUTDLG::IDD, pParent)
{

}

INPUTDLG::~INPUTDLG()
{
}

void INPUTDLG::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_EDIT_ROOTPATH, m_path);
}




BEGIN_MESSAGE_MAP(INPUTDLG, CDialogEx)
	ON_BN_CLICKED(IDOK, &INPUTDLG::OnBnClickedOk)
	ON_BN_CLICKED(IDC_BUTTON_BROWSE, &INPUTDLG::OnBnClickedButtonBrowse)
	ON_EN_CHANGE(IDC_EDIT_ROOTPATH, &INPUTDLG::OnEnChangeEditRootpath)
END_MESSAGE_MAP()


// INPUTDLG 消息处理程序


void INPUTDLG::OnBnClickedOk()
{
	// TODO: 在此添加控件通知处理程序代码
	m_path.GetWindowTextW(theApp.rootPath);

	CDialogEx::OnOK();
}


void INPUTDLG::OnBnClickedButtonBrowse()
{
	// TODO: 在此添加控件通知处理程序代码
	rootPath = GetDirPath();
	m_path.SetWindowTextW(rootPath);
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

void INPUTDLG::OnEnChangeEditRootpath()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}
