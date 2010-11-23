#pragma once
#include "afxwin.h"


// INPUTDLG 对话框

class INPUTDLG : public CDialogEx
{
	DECLARE_DYNAMIC(INPUTDLG)

public:
	INPUTDLG(CWnd* pParent = NULL);   // 标准构造函数
	virtual ~INPUTDLG();

// 对话框数据
	enum { IDD = IDD_INPUT_ROOT };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持
	CString GetDirPath();

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOk();
	CEdit m_path;
	afx_msg void OnBnClickedButtonBrowse();
	CString rootPath;
	afx_msg void OnEnChangeEditRootpath();
};
