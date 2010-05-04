varying vec4 vertexPosEye;  // ������ϵ�µĶ������� 

uniform ivec4 disturbPara;//M,KU,KV,slice, M���ؼ����KU��KV�Ŷ����,slice��ǰ�����Ƿ��Ŷ�,
uniform float fwidth;  //���طֱ���
uniform float disturb;  // �Ŷ���֣� delta x
uniform vec3 lightIntensity; // ��ǿ��
uniform vec4 lightPosWorld; // ��������ϵ�¹�Դλ��
uniform float absorptionCoefficient;  // ������� ����
uniform float scatteringCoefficient;  // ������� ����
uniform vec4 cameraPos;  // ���λ�� ����������ϵ�£�
uniform float stepSize;  // ray marching �Ĳ���
uniform sampler3D volumeTex;  // ���������ݵ���ά������id
uniform vec3 principalAxis;   // ����
uniform mat4 cameraInv;  // what the heck?

const float LOG2E = 1.442695; //1 / log(2)
const float PI = 3.1415926;

void main()
{	
  //ת������������ϵ	
  vec4 VertexPosWorld = cameraInv * vertexPosEye;	 

  //����ray
  vec3 raydir = cameraPos.xyz - VertexPosWorld.xyz;	
  raydir = normalize(raydir);
  float dotProduct = dot(raydir.xyz, principalAxis.xyz);
  //���㵱ǰ�����������߱�ƽ�ж���ηָ�ļ��
  float deltaStep = stepSize / abs(dotProduct);

  //�����Դ����ǰƬԪ�ľ���˥��
  vec3 lightAttenuation = lightIntensity  / pow(distance(lightPosWorld.xyz, VertexPosWorld.xyz), 2.0);	   

  //��ȡ�ܶ�
  float density = texture3D(volumeTex, gl_TexCoord[0].stp).r;

  int u=0;
  int v=0;
  /*	int w=0;
  u = int(gl_TexCoord[0].t*fwidth);
  v = int(gl_TexCoord[0].p*fwidth);	
  w = int(gl_TexCoord[0].s*fwidth);*/


  if(disturbPara.w == 1)
  {		
    u = int(gl_TexCoord[0].t*fwidth);
    v = int(gl_TexCoord[0].p*fwidth);	
    u = int(mod(float(u), float(disturbPara.x)));
    v = int(mod(float(v), float(disturbPara.x)));
    //u%M==K && v%M==k
    if(u== disturbPara.y && v == disturbPara.z)
    {
      density += disturb;
    }

  }
  if(disturbPara.w == 2)
  {
    u = int(gl_TexCoord[0].s*fwidth);
    v = int(gl_TexCoord[0].p*fwidth);		

    u = int(mod(float(u), float(disturbPara.x)));
    v = int(mod(float(v), float(disturbPara.x)));
    //u%M==K && v%M==k
    if(u== disturbPara.y && v == disturbPara.z)
    {
      density += disturb;
    }
  }
  if(disturbPara.w == 3)
  {
    u = int(gl_TexCoord[0].s*fwidth);
    v = int(gl_TexCoord[0].t*fwidth);		
    u = int(mod(float(u), float(disturbPara.x)));
    v = int(mod(float(v), float(disturbPara.x)));
    //u%M==K && v%M==k
    if(u== disturbPara.y && v == disturbPara.z)
    {
      density += disturb;
    }
  }

  if(density < 0.000001)
    discard;

  //�����ƬԪ�Ĳ�͸����
  float extinction = absorptionCoefficient;
  float attenuationTerm = exp2(-density*extinction*deltaStep*LOG2E);  

  //���㵱ǰƬԪɢ��Ĺ�ǿ   
  vec3 lightScatt = lightAttenuation *scatteringCoefficient*density*attenuationTerm*deltaStep/(4.0*PI);

  gl_FragData[0].rgb = lightScatt.xyz;
  gl_FragData[0].a = attenuationTerm;   


}