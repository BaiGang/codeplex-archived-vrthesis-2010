#pragma once


// COptionWnd

class COptionWnd : public CDockablePane
{
	DECLARE_DYNAMIC(COptionWnd)

public:
	COptionWnd();
	virtual ~COptionWnd();

protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);

	CButton radioCamera;
	CButton radioLight;
	CButton blankBn;

private:
	CFont m_font;

public:
	void setCamera();
	void setLight();
};


