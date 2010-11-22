// This MFC Samples source code demonstrates using MFC Microsoft Office Fluent User Interface 
// (the "Fluent UI") and is provided only as referential material to supplement the 
// Microsoft Foundation Classes Reference and related electronic documentation 
// included with the MFC C++ library software.  
// License terms to copy, use or distribute the Fluent UI are available separately.  
// To learn more about our Fluent UI licensing program, please visit 
// http://msdn.microsoft.com/officeui.
//
// Copyright (C) Microsoft Corporation
// All rights reserved.

// asmodeling.h : main header file for the asmodeling application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CasmodelingApp:
// See asmodeling.cpp for the implementation of this class
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

class CasmodelingApp : public CWinAppEx
{
public:
	CasmodelingApp();


// Overrides
public:
	virtual BOOL InitInstance();

// Implementation
	UINT  m_nAppLook;
	BOOL  m_bHiColorIcons;

	CString rootPath;
	struct xmlStruct currentXML;
	

	virtual void PreLoadState();
	virtual void LoadCustomState();
	virtual void SaveCustomState();

	

	afx_msg void OnAppAbout();
	afx_msg void OnSetRoot();
	DECLARE_MESSAGE_MAP()
};

extern CasmodelingApp theApp;
