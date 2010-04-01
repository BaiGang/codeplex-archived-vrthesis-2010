#include ".\boundingbox.h"
#include <stdlib.h>

BoundingBox::BoundingBox(void)
{
	m_vMin.nullVec();
	m_vMax.nullVec();
}

BoundingBox::~BoundingBox(void)
{
}

BoundingBox::BoundingBox(float xMin, float yMin, float zMin, float xMax, float yMax, float zMax)
{
	//设置左下角点
	m_vMin.x = xMin;
	m_vMin.y = yMin;
	m_vMin.z = zMin;
	
	//设置右上角点
	m_vMax.x = xMax;
	m_vMax.y = yMax;
	m_vMax.z = zMax;
	UpdateCorners();
}

void BoundingBox::ChangeBoxSize(Vector4 vMinPoint, Vector4 vMaxPoint)
{
	m_vMin = vMinPoint;
	m_vMax = vMaxPoint;
	UpdateCorners();
}

void BoundingBox::SetBoundingBox(float xMin, float yMin, float zMin, float xMax, float yMax, float zMax)
{
	m_vMin.setVec(xMin, yMin, zMin);
	m_vMax.setVec(xMax, yMax, zMax);
	UpdateCorners();
}


Vector4 BoundingBox::GetMaxPoint()
{
	return m_vMax;
}

Vector4 BoundingBox::GetMinPoint()
{
	return m_vMin;
}

void BoundingBox::SetMaxPoint(Vector4 vMaxPoint)
{
	m_vMax = vMaxPoint;
	UpdateCorners();
}

void BoundingBox::SetMaxPoint(float xMax, float yMax, float zMax )
{
	m_vMax.x = xMax;
	m_vMax.y = yMax;
	m_vMax.z = zMax;
	UpdateCorners();
}

void BoundingBox::SetMinPoint(Vector4 vMinPoint)
{
	m_vMin = vMinPoint;
	UpdateCorners();
}

void BoundingBox::SetMinPoint(float xMin, float yMin, float zMin)
{
	m_vMin.x = xMin;
	m_vMin.y = yMin;
	m_vMin.z = zMin;
	UpdateCorners();
}

void BoundingBox::MergeBoxWith(BoundingBox box)
{
	Vector4 vMinPoint = box.GetMinPoint();
	Vector4 vMaxPoint = box.GetMaxPoint();

	//比较两个包围盒左下角定点的坐标，取最小的
	
	m_vMin.x = __min(vMinPoint.x, m_vMin.x);

	m_vMin.y = __min(vMinPoint.y, m_vMin.y);

	m_vMin.z = __min(vMinPoint.z, m_vMin.z);

	//比较两个包围盒右上角点坐标，取最大的 
	
	m_vMax.x = __max(vMaxPoint.x, m_vMax.x);

	m_vMax.y = __max(vMaxPoint.y, m_vMax.y);

	m_vMax.z = __max(vMaxPoint.z, m_vMax.z);

	UpdateCorners();


}

BoundingBox BoundingBox::MergeTwoBox(BoundingBox box1, BoundingBox box2)
{
	Vector4 vMinPoint1 = box1.GetMinPoint();
	Vector4 vMaxPoint1 = box1.GetMaxPoint();

	Vector4 vMinPoint2 = box2.GetMinPoint();
	Vector4 vMaxPoint2 = box2.GetMaxPoint();

	BoundingBox boxResult;
	Vector4 vMinPointResult, vMaxPointResult;

	vMinPointResult.x = __min(vMinPoint1.x, vMinPoint2.x);
	vMinPointResult.y = __min(vMinPoint1.y, vMinPoint2.y);
	vMinPointResult.z = __min(vMinPoint1.z, vMinPoint2.z);

	vMaxPointResult.x = __max(vMaxPoint1.x, vMaxPoint2.x);
	vMaxPointResult.y = __max(vMaxPoint1.y, vMaxPoint2.y);
	vMaxPointResult.z = __max(vMaxPoint1.z, vMaxPoint2.z);

	boxResult.SetMinPoint(vMinPointResult);
	boxResult.SetMaxPoint(vMaxPointResult);

	return boxResult;
}

void BoundingBox::UpdateCorners()
{
	/*
	*	获得的包围盒各个定点的顺序如下图所示
       1-----2
	  /|    /|
	 / |   / |
	5-----4  |
	|  0--|--3
	| /   | /
	|/    |/
	6-----7
	 */
	m_vCorners[0] = m_vMin;
	m_vCorners[1].x = m_vMin.x; m_vCorners[1].y = m_vMax.y; m_vCorners[1].z = m_vMin.z;
	m_vCorners[2].x = m_vMax.x; m_vCorners[2].y = m_vMax.y; m_vCorners[2].z = m_vMin.z;
	m_vCorners[3].x = m_vMax.x; m_vCorners[3].y = m_vMin.y; m_vCorners[3].z = m_vMin.z;            

	m_vCorners[4] = m_vMax;
	m_vCorners[5].x = m_vMin.x; m_vCorners[5].y = m_vMax.y; m_vCorners[5].z = m_vMax.z;
	m_vCorners[6].x = m_vMin.x; m_vCorners[6].y = m_vMin.y; m_vCorners[6].z = m_vMax.z;
	m_vCorners[7].x = m_vMax.x; m_vCorners[7].y = m_vMin.y; m_vCorners[7].z = m_vMax.z;            

}

Vector4* BoundingBox::GetCorners()
{
	return m_vCorners;
}


/***********************************************************************************/
/*函数介绍：测试一条射线与包围盒是否相交
 *输入参数：rayOrg    射线的起点
			rayTarget 射线的终点
 *输出参数：无
 *返回值  ：如果相交则返回交点的坐标参数值(0<t<1)，否则返回小于0的数                                                         
/***********************************************************************************/
float BoundingBox::RayIntersece(const Vector4 rayOrg, const Vector4 rayTarget)
{
	return -1;
}










