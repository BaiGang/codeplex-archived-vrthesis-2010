#pragma once
#include "afxwin.h"


// INPUTDLG dialog

class INPUTDLG : public CDialog
{
	DECLARE_DYNAMIC(INPUTDLG)

public:
	INPUTDLG(CWnd* pParent = NULL);   // standard constructor
	virtual ~INPUTDLG();

// Dialog Data
	enum { IDD = IDD_INPUT_ROOT };


protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	CString GetDirPath();
	DECLARE_MESSAGE_MAP()

public:
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedButtonBrowse();
	CString rootPath;
	afx_msg void OnEnChangeEditRootpath();
	CEdit mPath;
};
