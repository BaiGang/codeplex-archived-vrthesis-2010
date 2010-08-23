//////////////////////////////////////////////////////////////////
//    Implemented based on HU Yong's shader code.
//
//  RayMarchingBlendX.frag
//  Ray marching for alpha blending along Y axis.
//
//
//////////////////////////////////////////////////////////////////

varying vec4 vertexPosEye;  // ������ϵ�µĶ������� 

//uniform vec3  boxTrans;
uniform ivec4 disturbPara;//M,KU,KV,PtSlice�� M���ؼ����KU��KV�Ŷ����,�Ŷ���Ƭ��
uniform float fwidth;  //���طֱ���
uniform float disturb;  // �Ŷ���֣� delta x
uniform vec3 lightIntensity; // ��ǿ��
uniform vec4 lightPosWorld; // ��������ϵ�¹�Դλ��
uniform float absorptionCoefficient;  // ������� ����
uniform float scatteringCoefficient;  // ������� ����
uniform vec4 cameraPos;  // ���λ�� ����������ϵ�£�
uniform float stepSize;  // ray marching �Ĳ���
uniform sampler3D volumeTex;  // ���������ݵ���ά�����id
uniform mat4 cameraInv;  //

const float LOG2E = 1.442695; //1 / log(2)
const float PI = 3.1415926;

void main()
{	
  //ת������������ϵ	
  vec4 VertexPosWorld = cameraInv * vertexPosEye;	 
  //VertexPosWorld = VertexPosWorld - vec4(boxTrans.xyz, 0.0);

  //����ray
  vec3 raydir = cameraPos.xyz - VertexPosWorld.xyz;	
  raydir = normalize(raydir);
  float dotProduct = dot(raydir.xyz, vec3(0.0, 1.0, 0.0));
  //���㵱ǰ�����������߱�ƽ�ж���ηָ�ļ��
  float deltaStep = stepSize / abs(dotProduct);

  //�����Դ����ǰƬԪ�ľ���˥��
  vec3 lightAttenuation = lightIntensity  / pow(distance(lightPosWorld.xyz, VertexPosWorld.xyz), 2.0);	   

  //��ȡ�ܶ�
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



  //�����ƬԪ�Ĳ�͸����
  float extinction = absorptionCoefficient;
  float attenuationTerm = exp2(-density*extinction*deltaStep*LOG2E);  

  //���㵱ǰƬԪɢ��Ĺ�ǿ   
  vec3 lightScatt = lightAttenuation *scatteringCoefficient*density*attenuationTerm*deltaStep/(4.0*PI);

  gl_FragData[0].rgb = lightScatt.xyz;
  gl_FragData[0].a = attenuationTerm;   


}