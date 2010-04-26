varying vec4 vertexPosEye;
void main()
{	
	vertexPosEye = gl_ModelViewMatrix * gl_Vertex;
	gl_TexCoord[0] = gl_MultiTexCoord0;	
	gl_Position = ftransform();
}


