//////////////////////////////////////////////////////////////////
//    Implemented based on HU Yong's shader code.
//
//  RayMarchingBlendX.frag
//  Ray marching for alpha blending along Y axis.
//
//
//////////////////////////////////////////////////////////////////

varying vec4 vertexPosEye;  // 视坐标系下的顶点坐标 

//uniform vec3  boxTrans;
uniform ivec4 disturbPara;//M,KU,KV,PtSlice： M体素间隔，KU，KV扰动序号,扰动切片号
uniform float fwidth;  //体素分辨率
uniform float disturb;  // 扰动差分， delta x
uniform vec3 lightIntensity; // 光强度
uniform vec4 lightPosWorld; // 世界坐标系下光源位置
uniform float absorptionCoefficient;  // 参与介质 属性
uniform float scatteringCoefficient;  // 参与介质 属性
uniform vec4 cameraPos;  // 相机位置 （世界坐标系下）
uniform float stepSize;  // ray marching 的步长
uniform sampler3D volumeTex;  // 保存体数据的三维纹理的id
uniform mat4 cameraInv;  //

const float LOG2E = 1.442695; //1 / log(2)
const float PI = 3.1415926;

void main()
{	
  //转换到世界坐标系	
  vec4 VertexPosWorld = cameraInv * vertexPosEye;	 
  //VertexPosWorld = VertexPosWorld - vec4(boxTrans.xyz, 0.0);

  //计算ray
  vec3 raydir = cameraPos.xyz - VertexPosWorld.xyz;	
  raydir = normalize(raydir);
  float dotProduct = dot(raydir.xyz, vec3(0.0, 1.0, 0.0));
  //计算当前像素所在视线被平行多边形分割的间距
  float deltaStep = stepSize / abs(dotProduct);

  //计算光源到当前片元的距离衰减
  vec3 lightAttenuation = lightIntensity  / pow(distance(lightPosWorld.xyz, VertexPosWorld.xyz), 2.0);	   

  //获取密度
  float density = texture3D(volumeTex, gl_TexCoord[0].stp).r;

  if(density < disturb)
    discard;

  int u = int(gl_TexCoord[0].s*fwidth);
  int v = int(gl_TexCoord[0].p*fwidth);		
  int w = int(gl_TexCoord[0].t*fwidth);

  u = int(mod(float(u), float(disturbPara.x)));
  v = int(mod(float(v), float(disturbPara.x)));
  // on the slie and u%M==K && v%M==k
  if(w == disturbPara.w && u== disturbPara.y && v == disturbPara.z)
  {
    density += disturb;
  }



  //计算该片元的不透明度
  float extinction = absorptionCoefficient;
  float attenuationTerm = exp2(-density*extinction*deltaStep*LOG2E);  

  //计算当前片元散射的光强   
  vec3 lightScatt = lightAttenuation *scatteringCoefficient*density*attenuationTerm*deltaStep/(4.0*PI);

  gl_FragData[0].rgb = lightScatt.xyz;
  gl_FragData[0].a = attenuationTerm;   


}