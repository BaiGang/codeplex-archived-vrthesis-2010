// INPUTDLG.cpp : ʵ���ļ�
//

#include "stdafx.h"
#include "test.h"
#include "INPUTDLG.h"
#include "afxdialogex.h"


// INPUTDLG �Ի���

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


// INPUTDLG ��Ϣ�������


void INPUTDLG::OnBnClickedOk()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	m_path.GetWindowTextW(theApp.rootPath);

	CDialogEx::OnOK();
}


void INPUTDLG::OnBnClickedButtonBrowse()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	rootPath = GetDirPath();
	m_path.SetWindowTextW(rootPath);
}

CString  INPUTDLG::GetDirPath()
{
    CString strPath;
    BROWSEINFO bInfo;
    ZeroMemory(&bInfo, sizeof(bInfo));
//    bInfo.hwndOwner = this;
    bInfo.lpszTitle = _T("��ѡ�������ݵ��ļ���: ");
    bInfo.ulFlags = BIF_RETURNONLYFSDIRS;    

    LPITEMIDLIST lpDlist; //�������淵����Ϣ��IDList
    lpDlist = SHBrowseForFolder(&bInfo) ; //��ʾѡ��Ի���
    if(lpDlist != NULL)  //�û�����ȷ����ť
    {
        TCHAR chPath[255]; //�����洢·�����ַ���
        SHGetPathFromIDList(lpDlist, chPath);//����Ŀ��ʶ�б�ת�����ַ���
        strPath = chPath; //��TCHAR���͵��ַ���ת��ΪCString���͵��ַ���
    }
    return strPath;
}

void INPUTDLG::OnEnChangeEditRootpath()
{
	// TODO:  ����ÿؼ��� RICHEDIT �ؼ���������
	// ���ʹ�֪ͨ��������д CDialogEx::OnInitDialog()
	// ���������� CRichEditCtrl().SetEventMask()��
	// ͬʱ�� ENM_CHANGE ��־�������㵽�����С�

	// TODO:  �ڴ���ӿؼ�֪ͨ����������
}
