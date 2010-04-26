
varying vec4 posProj;
varying vec4 nearVertexPosEye;
void main()
{
	
	nearVertexPosEye = (gl_ModelViewMatrix * gl_Vertex);
//	rayDir = nearVertexPos.xyz - cameraPos.xyz;
	
//	rayDir = normalize(rayDir);
	posProj = gl_ModelViewProjectionMatrix  * gl_Vertex;
	gl_Position = ftransform();
}


