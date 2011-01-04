#pragma once
#include <vector>
#include <math.h>
using namespace std;
struct Point2D
{
	float x;
	float y;
};
typedef vector<Point2D> PT2DVEC;

class ConvexHull2D
{
public:
	ConvexHull2D(void);
	ConvexHull2D(PT2DVEC pts);
	~ConvexHull2D(void);
	void BuildConvexHull(PT2DVEC pts);
	bool IfInConvexHull(float x, float y);
	PT2DVEC& GetConvexHullPoints();
	void GetBoundingBox(float& xMin, float& xMax, float& yMin, float& yMax);
private:	

	//����(P1-P0)*(P2-P0)�Ĳ���� 
	int Multiply(Point2D p1,Point2D p2,Point2D p0);
	//����(P1-P0)*(P2-P0)�ĵ��
	int Dot(Point2D p1,Point2D p2,Point2D p0);
	//��ƽ��������֮��ľ���   
	float Distance(Point2D p1,Point2D p2);


private:
	PT2DVEC m_cvPoint;

	Point2D ur;
	Point2D ll;



};