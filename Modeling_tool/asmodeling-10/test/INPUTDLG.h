#pragma once
#include "afxwin.h"


// INPUTDLG �Ի���

class INPUTDLG : public CDialogEx
{
	DECLARE_DYNAMIC(INPUTDLG)

public:
	INPUTDLG(CWnd* pParent = NULL);   // ��׼���캯��
	virtual ~INPUTDLG();

// �Ի�������
	enum { IDD = IDD_INPUT_ROOT };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��
	CString GetDirPath();

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOk();
	CEdit m_path;
	afx_msg void OnBnClickedButtonBrowse();
	CString rootPath;
	afx_msg void OnEnChangeEditRootpath();
};
