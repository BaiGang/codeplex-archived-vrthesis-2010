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

// testView.h : CtestView ��Ľӿ�
//

#pragma once

#include "resource.h"


class CtestView : public CFormView
{
protected: // �������л�����
	CtestView();
	DECLARE_DYNCREATE(CtestView)

public:
	enum{ IDD = IDD_TEST_FORM };

// ����
public:
	CtestDoc* GetDocument() const;

// ����
public:

// ��д
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��
	virtual void OnInitialUpdate(); // ������һ�ε���
	CPoint					m_mouseLast;			// Mouse Last point
	enum RBState {CAMERA, LIGHT};
	RBState currentRBState;
// ʵ��
public:
	virtual ~CtestView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// ���ɵ���Ϣӳ�亯��
protected:
	afx_msg void OnFilePrintPreview();
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnContextMenu(CWnd* pWnd, CPoint point);
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg void OnBnClickedRadioCamera();
	afx_msg void OnBnClickedRadioLight();
};

#ifndef _DEBUG  // testView.cpp �еĵ��԰汾
inline CtestDoc* CtestView::GetDocument() const
   { return reinterpret_cast<CtestDoc*>(m_pDocument); }
#endif

