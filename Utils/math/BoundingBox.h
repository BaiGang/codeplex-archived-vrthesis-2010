#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include "geomath.h"


class Camera;

/*
 *	AABB轴向包围盒由左下角的点和右上角的点来记录
 */
class BoundingBox
{
public:
	BoundingBox(void);
	BoundingBox(float xMin, float yMin, float zMin,
		float xMax, float yMax, float zMax);
	virtual ~BoundingBox(void);
	//数据存取
	Vector4 GetMinPoint();
	Vector4 GetMaxPoint();
	void SetMinPoint(float xMin, float yMin, float zMin);
	void SetMaxPoint(float xMax, float yMax, float zMax);
	void SetMinPoint(Vector4 vMinPoint);
	void SetMaxPoint(Vector4 vMaxPoint);
	void SetBoundingBox(float xMin, float yMin, float zMin, float xMax, float yMax, float zMax);
	//改变包围盒大小
	void ChangeBoxSize(Vector4 vMinPoint, Vector4 vMaxPoint);
	//合并两个包围盒
	void MergeBoxWith(BoundingBox box);
	BoundingBox MergeTwoBox(BoundingBox box1, BoundingBox box2);
	//获取包围盒的八个定点坐标
	Vector4* GetCorners();

	//射线与包围盒的相交检测
	float RayIntersece(const Vector4 rayOrg, const Vector4 rayTarget);

	//判断Camera是否与包围盒碰撞
	bool CollideWithCamera(Camera* pCamera);
private:
	void UpdateCorners();
	Vector4 m_vMin;
	Vector4 m_vMax;
	Vector4 m_vCorners[8];
};

#endif
