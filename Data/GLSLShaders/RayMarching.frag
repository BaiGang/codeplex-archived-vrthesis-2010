varying vec4 nearVertexPosEye;
varying vec4 posProj;
uniform vec3 lightIntensity;
uniform vec4 lightPosWorld;
uniform float absorptionCoefficient;
uniform float scatteringCoefficient;
//uniform vec4 cameraPos;
uniform int steps;
uniform sampler3D volumeTex;
uniform sampler2D backPosTex;
uniform sampler2D gaussLegendreTex;
uniform mat4 modelInv;
uniform mat4 cameraInv;
uniform float boxSize;
const float LOG2E = 1.442695; //1 / log(2)
const float PI = 3.1415926;
float boxNorScale = 1.0/boxSize;
/*
//计算点光源与任意voxel连线与volume的交点
void ComputeNearPoint(vec3 pointpos, vec3 lightpos, vec3 r, out float tnear)
{
	vec3 boxmin = vec3(0.0, 0.0, 0.0);
	vec3 boxmax = vec3(1.0, 1.0, 1.0);
    // compute intersection of ray with all six bbox planes
    vec3 invR = 1.0 / r;
    vec3 tbot = invR * (boxmin.xyz - lightpos);
    vec3 ttop = invR * (boxmax.xyz - lightpos);

    // re-order intersections to find smallest and largest on each axis
    vec3 tmin = min (ttop, tbot);
    vec3 tmax = max (ttop, tbot);

    // find the largest tmin and the smallest tmax
    vec2 t0 = max(tmin.xx, tmin.yz);
    float largest_tmin = max (t0.x, t0.y);
    tnear = largest_tmin;         
}
//计算光源到voxel的transmittance 
float ComputeTransmittance(vec3 pointpos, vec3 lightpos, float step)
{
		float tNear = 0.0;
		vec3 r = (pointpos-lightpos); 
		r = normalize(r);
		ComputeNearPoint(pointpos, lightpos, r, tNear);
		vec3 P = lightpos + r*tNear;
		float stepSize = 1.0/step;
		float density = 0.0;
		float integrationRes = 0.0;
		float extinction = scatteringCoefficient+absorptionCoefficient;
		float dist = distance(pointpos, lightpos)/step;
		for(int i=0; i<128; i++)
		{
			density = texture3D(volumeTex, P).r;
			integrationRes = integrationRes - density*extinction*dist;
			P += i*stepSize;
		}
		return integrationRes;
}*/

float GaussLegendreIntegration(vec3 pointBegin, vec3 pointEnd, vec3 direction)
{
	
	float integrationRes = 0.0;	
	
	//g channel is the weight
	//取出权重值和积分点变量位置
	vec2 glTable00 = texture2D(gaussLegendreTex, vec2(0,0)).rg;
	vec2 glTable01 = texture2D(gaussLegendreTex, vec2(0,1)).rg;
	vec2 glTable10 = texture2D(gaussLegendreTex, vec2(1,0)).rg;
	vec2 glTable11 = texture2D(gaussLegendreTex, vec2(1,1)).rg; 
	
	vec4 weights = vec4(glTable00.g, glTable01.g, glTable10.g, glTable11.g);
	vec4 variables = vec4(glTable00.r, glTable01.r, glTable10.r, glTable11.r);
	
	//转化积分区间，从[0, tEnd]到[-1,+1]
	float tEndHalf = distance(pointEnd, pointBegin)/2.0;
	variables = (variables+1.0)*tEndHalf;//variables*(tEnd/2.0)+tEnd/2.0;
	
	//计算纹理坐标，取出密度值
	vec4 funv;
	vec4 ThreeDTexCoord = vec4(1.0, 1.0, 1.0, 1.0);
	ThreeDTexCoord.xyz = pointBegin + variables.r*direction;
	
	// 转换到纹理坐标空间[0,1]
	ThreeDTexCoord =  modelInv * ThreeDTexCoord;    
	ThreeDTexCoord.xyz = ThreeDTexCoord.xyz*boxNorScale + 0.5; 
	funv.r = texture3D(volumeTex, ThreeDTexCoord.xyz).r;
	
	ThreeDTexCoord.xyz = pointBegin + variables.g*direction;
	// 转换到纹理坐标空间[0,1]
	ThreeDTexCoord =  modelInv * ThreeDTexCoord;    
	ThreeDTexCoord.xyz = ThreeDTexCoord.xyz*boxNorScale + 0.5; 
	funv.g = texture3D(volumeTex, ThreeDTexCoord.xyz).r;
	
	ThreeDTexCoord.xyz = pointBegin + variables.b*direction;
	// 转换到纹理坐标空间[0,1]
	ThreeDTexCoord =  modelInv * ThreeDTexCoord;    
	ThreeDTexCoord.xyz = ThreeDTexCoord.xyz*boxNorScale + 0.5; 
	funv.b = texture3D(volumeTex, ThreeDTexCoord.xyz).r;
	
	ThreeDTexCoord.xyz = pointBegin + variables.a*direction;
	// 转换到纹理坐标空间[0,1]
	ThreeDTexCoord =  modelInv * ThreeDTexCoord;    
	ThreeDTexCoord.xyz = ThreeDTexCoord.xyz*boxNorScale + 0.5; 
	funv.a = texture3D(volumeTex, ThreeDTexCoord.xyz).r;
	
	//计算积分
	integrationRes = dot(funv, weights);
	integrationRes *= tEndHalf;
	//计算消光系数
	float extinction = scatteringCoefficient+absorptionCoefficient;
	integrationRes *=-extinction;
	
	return integrationRes;
}

void main()
{	
	//读取back face 纹理
	vec2 texCoor = (posProj.xy*0.5/posProj.w + vec2(0.5, 0.5));
	vec4 farVertexPosEye = texture2D(backPosTex, texCoor.st);
	//转换到世界坐标系	
	vec4 farVertexPosWorld = cameraInv * farVertexPosEye;
	vec4 nearVertexPosWorld = cameraInv* nearVertexPosEye;
	
	//转换到-1到+1坐标空间
//	vec4 norFarVertexPos =  modelviewInv * farVertexPos;
//	vec4 norNearVertexPos = modelviewInv * nearVertexPos;	
//	vec4 newLightPos = modelInv * lightPos;    
    // 转换到纹理坐标空间[0,1]
//    norNearVertexPos = norNearVertexPos*0.25 + 0.5;
//    norFarVertexPos = norFarVertexPos*0.25 + 0.5; 
//    newLightPos = newLightPos*0.25 + 0.5;  
    
    
      
   //计算ray
   	vec3 raydir = farVertexPosWorld.xyz - nearVertexPosWorld.xyz;	
	raydir = normalize(raydir);	         
   float rayLength = distance(farVertexPosWorld.xyz, nearVertexPosWorld.xyz); 
   float deltaStep = rayLength / float(steps);
   if(rayLength == 0.0)
   {
	  raydir = vec3(0.0,0.0,0.0);
   }
   
     //debug begin
   	//转换到-1到+1坐标空间
//	vec4	pp =  modelInv * nearVertexPosWorld;  
//	vec4 pf = modelInv * farVertexPosWorld;  
	// 转换到纹理坐标空间[0,1]
//	pp = pp*boxNorScale + 0.5;
//	pf = pf*boxNorScale+0.5;
 //   gl_FragData[0].rgb = normalize(pp.xyz-pf.xyz);
    //debug end
    

	float extinction = absorptionCoefficient;
	
 
	// march along ray, accumulating color
    vec4 c = vec4(0.0, 0.0, 0.0, 0.0);	
	float opticalThicknessP2V = 0.0;
	float opticalThicknessL2P = 0.0;
	vec3 lx0 = vec3(1.0, 1.0, 1.0);
		
	 // use front-to-back rendering
	float density = 0.0;
    vec4 P = nearVertexPosWorld;
    vec4 norP;
    vec3 Pstep = raydir * deltaStep;
	float deltaTao = 1.0;	
	float a=0.0;
    for(int i=0; i<steps; i++) 
    {
		//转换到-1到+1坐标空间
		norP =  modelInv * P;    
	    // 转换到纹理坐标空间[0,1]
	    norP = norP*boxNorScale + 0.5;  
    
    
/*		density = texture3D(volumeTex, norP.xyz).r;
		opticalThicknessP2V = GaussLegendreIntegration(P, nearVertexPos.xyz, -raydir);
		opticalThicknessP2V = exp2(opticalThicknessP2V*LOG2E);		
		
		opticalThicknessL2P = GaussLegendreIntegration(newLightPos.xyz, P,normalize(P-newLightPos.xyz));
		opticalThicknessL2P = exp2(opticalThicknessL2P*LOG2E);
		
		lx0 = lightIntensity * opticalThicknessL2P / pow(distance(newLightPos.xyz, P), 2.0);
		
		
		lx0 = lx0*scatteringCoefficient*density/(4.0*PI);
		
        c.rgb += lx0*opticalThicknessP2V;   */


		density = texture3D(volumeTex, norP.xyz).r;
		a = GaussLegendreIntegration(lightPosWorld.xyz, P.xyz, normalize(P.xyz-lightPosWorld.xyz));
		opticalThicknessL2P = 0.0;
//		opticalThicknessL2P = ComputeTransmittance(P.xyz, lightPosWorld.xyz, 1.0);
		opticalThicknessL2P = exp2(opticalThicknessL2P*LOG2E);	
	
		lx0 = lightIntensity * opticalThicknessL2P / pow(distance(lightPosWorld.xyz, P.xyz), 2.0);		
		lx0 = lx0*scatteringCoefficient*density/(4.0*PI);		
		
		deltaTao *= exp2(-density*extinction*deltaStep*LOG2E);   
		
        c.rgb += lx0*deltaTao*deltaStep;
        //c.rgb += lx0*deltaTao;          
        P.xyz += Pstep;   

    }
 
  gl_FragData[1].rgb = vec3(1.0, 1.0, 1.0);
  gl_FragData[0].a = a;
  gl_FragData[0].rgb=c.xyz;
  
}