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

// asmodelingDoc.cpp : implementation of the CasmodelingDoc class
//

#include "stdafx.h"
#include "asmodeling.h"

#include "asmodelingDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CasmodelingDoc

IMPLEMENT_DYNCREATE(CasmodelingDoc, CDocument)

BEGIN_MESSAGE_MAP(CasmodelingDoc, CDocument)
END_MESSAGE_MAP()


// CasmodelingDoc construction/destruction

CasmodelingDoc::CasmodelingDoc()
{
	// TODO: add one-time construction code here

}

CasmodelingDoc::~CasmodelingDoc()
{
}

BOOL CasmodelingDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}




// CasmodelingDoc serialization

void CasmodelingDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add loading code here
	}
}


// CasmodelingDoc diagnostics

#ifdef _DEBUG
void CasmodelingDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CasmodelingDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CasmodelingDoc commands
