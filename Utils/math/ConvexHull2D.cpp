#include "ConvexHull2D.h"
#include <iostream>

ConvexHull2D::ConvexHull2D(void)
{

}
ConvexHull2D::~ConvexHull2D()
{

}
ConvexHull2D::ConvexHull2D(PT2DVEC pts)
{
	this->BuildConvexHull(pts);
}
void ConvexHull2D::BuildConvexHull(PT2DVEC pts)
{
	int i,j,k=0,top=2;
	Point2D tmp;

	int inlen;
	inlen = pts.size();

	Point2D* ch;
	ch = new Point2D[inlen];

	ur.x = pts[0].x;
	ur.y = pts[0].y;
	ll.x = pts[0].x;

	//选取PointSet中y坐标最小的点PointSet[k]，如果这样的点有多个，则取最左边的一个   
	for(i=1;i<inlen;i++)
	{
		if((pts[i].y<pts[k].y)
			||((pts[i].y==pts[k].y)&&(pts[i].x<pts[k].x)))   
			k=i; 

		if(pts[i].x>ur.x)
			ur.x = pts[i].x;
		if(pts[i].y>ur.y)
			ur.y = pts[i].y;
		if(pts[i].x<ll.x)
			ll.x = pts[i].x;
	}

	ll.y=pts[k].y;

	tmp=pts[0];   
	pts[0]=pts[k];   
	pts[k]=tmp;   //现在PointSet中y坐标最小的点在PointSet[0]

	for(i=1;i<inlen-1;i++)	//对顶点按照相对PointSet[0]的极角从小到大进行排序，极角相同的按照距离PointSet[0]从近到远进行排序   
	{       
		k=i;   

		for(j=i+1;j<inlen;j++)   
			if((Multiply(pts[j],pts[k],pts[0])>0)||((Multiply(pts[j],pts[k],pts[0])==0)&&(Distance(pts[0],pts[j])<Distance(pts[0],pts[k]))))
				k=j;   

		tmp=pts[i];
		pts[i]=pts[k];
		pts[k]=tmp;   
	}  

	//   以上的代码将点集按照逆时针排序
	//std::cout << pts.size() << endl;
	//for ( int i = 0; i < pts.size(); i++ )
	//{
	//	std::cout << pts[i].x << " " << pts[i].y << endl;
	//}

	*ch=pts[0];   
	*(ch+1)=pts[1];   
	*(ch+2)=pts[2];
	//for ( int i = 0; i < top; i++ )
	//	std::cout << (ch+i)->x << (ch+i)->y << endl;

	//std::cout << top << endl;

	for(i=3;i<inlen;i++)
	{   
		//std::cout << i << " " << top << " " << Multiply(pts[i],*(ch+top),*(ch+top-1)) << endl;
		while( top > 0 && Multiply(pts[i],*(ch+top),*(ch+top-1)) >= 0 )
		{		
			//if ( i == 3 )
			//	std::cout << Multiply(pts[i],*(ch+top),*(ch+top-1)) << endl;
			//if ( i == 3 && Multiply(pts[i],*(ch+top),*(ch+top-1))>= 0 )
			//	//cout << ">0" << endl;
			//	cout << pts[i].x << " " << pts[i].y << " "
			//	<< (ch+top)->x << " " << (ch+top)->y << " "
			//	<< (ch+top-1)->x << " " << (ch+top-1)->y << endl;
			//	std::cout << "11" << endl;
			top--;
		}

		*(ch+top+1)=pts[i];
		top++;
	}

	//std::cout << top << endl;

	for(i=0;i<=top;i++)
	{
		m_cvPoint.push_back(*(ch+i));
	}
	delete[] ch;
}


int ConvexHull2D::Multiply(Point2D p1,Point2D p2,Point2D p0)
{   
	double p = ((p1.x-p0.x)*(p2.y-p0.y)-(p2.x-p0.x)*(p1.y-p0.y));
	if ( p > 0 )
		return 1;
	else if ( p < 0 )
		return -1;
	//std::cout << p << endl;
	return 0;
}

int ConvexHull2D::Dot(Point2D p1,Point2D p2,Point2D p0)
{
	double p = ((p1.x-p0.x)*(p2.x-p0.x)+(p2.y-p0.y)*(p1.y-p0.y));
	if ( p > 0 )
		return 1;
	else if ( p < 0 )
		return -1;
	//std::cout << p << endl;
	return 0;
}

float ConvexHull2D::Distance(Point2D p1,Point2D p2)   
{   
	return(sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)));
}   

PT2DVEC& ConvexHull2D::GetConvexHullPoints()
{
	return m_cvPoint;
}


bool ConvexHull2D::IfInConvexHull(float x, float y)
{
	bool res = true;

	//想象一个凸多边形，其每一个边都将整个2D屏幕划分成为左右两边，连接每一边的第一个端点和要测试的点得到一个矢量v，
	//将两个2维矢量扩展成3维的，然后将该边与v叉乘，判断结果3维矢量中Z分量的符号是否发生变化，
	//进而推导出点是否处于凸多边形内外。这里要注意的是，多边形顶点究竟是左手序还是右手序，这对具体判断方式有影响。
	float z;
	Point2D pt;
	pt.x = x;
	pt.y = y;
	int ptIndex1, ptIndex2;

	if (m_cvPoint.size() == 0)
	{
		return res;
	}
	else if ( m_cvPoint.size() == 2 )
	{
		int r = Multiply(m_cvPoint[0], pt, m_cvPoint[1]);
		if ( r == 0 && Dot(m_cvPoint[0], m_cvPoint[1], pt ) <= 0 )
		{
			res = true;
		}
		else
			res = false;
		return res;
	}

	for (int i=0; i<m_cvPoint.size(); i++)
	{
		ptIndex1 = i;
		ptIndex2 = i+1;
		if (ptIndex2 == m_cvPoint.size())
		{
			ptIndex2 = 0;
		}
		z = Multiply(m_cvPoint[ptIndex2], pt, m_cvPoint[ptIndex1]);
		if (z < 0)
		{
			res = false;
			break;
		}
	}
	return res;
}


void ConvexHull2D::GetBoundingBox(float& xMin, float& xMax, float& yMin, float& yMax)
{
	xMin = ll.x;
	xMax = ur.x;

	yMin = ll.y;
	yMax = ur.y;
}