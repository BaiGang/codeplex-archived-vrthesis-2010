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

// test.h : test Ӧ�ó������ͷ�ļ�
//
#pragma once
#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"       // ������

#include "image\Image.h"
#include "image\PFMImage.h"

// CtestApp:
// �йش����ʵ�֣������ test.cpp
//
struct Levels{
	CString Level5;
	CString Level6;
	CString Level7;
	CString Level8;
};

struct PMedia
{
	CString extinction;
	CString scattering;
	CString alaph;
};

struct Render
{
	CString CurrentView;
	CString width;
	CString height;
	struct Levels RenderInterval;
	CString RotAngle;
};

struct Light{
	CString LightType;
	CString LightIntensityR;
	CString LightX;
	CString LightY;
	CString LightZ;
};

struct Volume{
	CString BoxSize;
	CString TransX;
	CString TransY;
	CString TransZ;
	CString VolumeInitialLevel;
	CString VolumeMaxLevel;
	struct Levels VolInterval;
};
struct LBFGSB{
	CString disturb;
	CString EpsG;
	CString EpsF;
	CString EpsX;
	CString MaxIts;
	CString m;
	CString ConstrainType;
	CString LowerBound;
	CString UpperBound;
};
struct  xmlStruct
{
	struct PMedia PMedia;

	struct Render Render;

	struct Light Light;

	struct Volume Volume;

	struct LBFGSB LBFGSB;
};

class CtestApp : public CWinAppEx
{
public:
	CtestApp();


// ��д
public:
	virtual BOOL InitInstance();

// ʵ��
	UINT  m_nAppLook;
	BOOL  m_bHiColorIcons;
	CString rootPath;
	struct xmlStruct currentXML;
	PFMImage* lastPic;
	PFMImage* currentPic;
	PFMImage* nextPic;
	int picIndex;

	void loadPic();
	char* cstring2char (CString cstr);

	virtual void PreLoadState();
	virtual void LoadCustomState();
	virtual void SaveCustomState();
	int ExitInstance();

	afx_msg void OnAppAbout();
	afx_msg void OnSetRoot();
	afx_msg void OnButtonLastPic();
	afx_msg void OnButtonNextPic();

	void loadXml();

	DECLARE_MESSAGE_MAP()
	
};

extern CtestApp theApp;
