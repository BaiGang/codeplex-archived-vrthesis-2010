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
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	mPath.GetWindowText(theApp.rootPath);

	CDialog::OnOK();
}


void INPUTDLG::OnBnClickedButtonBrowse()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	rootPath = GetDirPath();
	mPath.SetWindowText(rootPath);
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