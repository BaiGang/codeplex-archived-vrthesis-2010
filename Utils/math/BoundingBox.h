#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include "geomath.h"


class Camera;

/*
 *	AABB�����Χ�������½ǵĵ�����Ͻǵĵ�����¼
 */
class BoundingBox
{
public:
	BoundingBox(void);
	BoundingBox(float xMin, float yMin, float zMin,
		float xMax, float yMax, float zMax);
	virtual ~BoundingBox(void);
	//���ݴ�ȡ
	Vector4 GetMinPoint();
	Vector4 GetMaxPoint();
	void SetMinPoint(float xMin, float yMin, float zMin);
	void SetMaxPoint(float xMax, float yMax, float zMax);
	void SetMinPoint(Vector4 vMinPoint);
	void SetMaxPoint(Vector4 vMaxPoint);
	void SetBoundingBox(float xMin, float yMin, float zMin, float xMax, float yMax, float zMax);
	//�ı��Χ�д�С
	void ChangeBoxSize(Vector4 vMinPoint, Vector4 vMaxPoint);
	//�ϲ�������Χ��
	void MergeBoxWith(BoundingBox box);
	BoundingBox MergeTwoBox(BoundingBox box1, BoundingBox box2);
	//��ȡ��Χ�еİ˸���������
	Vector4* GetCorners();

	//�������Χ�е��ཻ���
	float RayIntersece(const Vector4 rayOrg, const Vector4 rayTarget);

	//�ж�Camera�Ƿ����Χ����ײ
	bool CollideWithCamera(Camera* pCamera);
private:
	void UpdateCorners();
	Vector4 m_vMin;
	Vector4 m_vMax;
	Vector4 m_vCorners[8];
};

#endif
