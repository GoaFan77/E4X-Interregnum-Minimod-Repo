#define RIPPLE_SCALE 25.0
#define RIPPLE_SPEED 10.0
#define RIPPLE_REFINEMENT_ITERATION 3 //complexity of the ripple, smaller number reduces complexity
#define RIPPLE_DERIVATIVE_WIDTH 50.0

#define USE_TURBULENCE // Turbulence effect can be turned off for less complex shield if performance is bad
	#define TURBULENCE_INTENSITY 1.0
	#define TURBULENCE_SPEED 0.5

#define IMPACTFLARE_SCALE 1.0
#define IMPACTFLARE_ATTENUATION 100.0
#define IMPACTFLARE_INTENSITY 1000.0
#define IMPACTFLARE_PULSE_SPEED 10.0

//math constants, don't touch :P
#define PI 		3.1415926535897932384626433832795
#define INVPI 	0.3183098861837906715377675267450
#define TWOPI	6.283185307179586476925286766559

//gravity wave function
#define W(x,k,c) A*sin(k*(X=x-c*t))*exp(-X*X)

float4x4 g_World : World;
float4x4 g_WorldViewProjection	: WorldViewProjection;

float g_Time;
float3 g_ImpactPosition;
float g_ShieldPercent;

texture g_TextureNoise3D;
texture	g_TextureEnvironmentCube : Environment;
texture g_TextureDiffuse0;

float g_InternalHitRadius;
float g_ExternalHitRadius;

float4 g_GlowColor;

samplerCUBE TextureEnvironmentCubeSampler = sampler_state{
    Texture = <g_TextureEnvironmentCube>;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = LINEAR;
    AddressU = CLAMP;
    AddressV = CLAMP;
    AddressW = CLAMP;
};

sampler NoiseSampler 
= sampler_state 
{
    texture = < g_TextureNoise3D >;
    AddressU  = WRAP;        
    AddressV  = WRAP;
	AddressW  = WRAP;
    MIPFILTER = LINEAR;
    MINFILTER = LINEAR;
    MAGFILTER = LINEAR;
};

float Square(float X)
{
	return X * X;
}

float Pow3(float X)
{
	return Square(X) * X;
}

float Pow4(float X)
{
	return Square(X) * Square(X);
}

float Pow5(float X)
{
	return Pow4(X) * X;
}	

float3 Square(float3 X)
{
	return X * X;
}
float4 Square(float4 X)
{
	return X * X;
}	
	

void
RenderSceneVS_WithoutPixelShader( 
	float3 iPosition					: POSITION, 
	float2 iTexCoord0					: TEXCOORD0,
	float3 iNormal						: NORMAL,
	float4 iColor						: COLOR0,
	out float4 oPosition				: POSITION,
    out float4 oColor0					: COLOR0,
    out float2 oTexCoord0				: TEXCOORD0 )
{
	float dist = distance( iPosition, g_ImpactPosition );
	
	oColor0 = g_GlowColor;
	oColor0.a = lerp( 0.1f, 0.f, dist/g_ExternalHitRadius);

	oColor0.a *= g_ShieldPercent;

	oPosition = mul( float4( iPosition, 1.f ), g_WorldViewProjection );
   
    oTexCoord0 = iTexCoord0; 
}

void
RenderSceneVS_WithPixelShader( 
	float3 iPosition					: POSITION, 
	float2 iTexCoord0					: TEXCOORD0,
	float3 iNormal						: NORMAL,
	float3 iTangent						: TANGENT,
	float4 iColor						: COLOR0,
	out float4 oPosition				: POSITION,
    out float4 oColor0					: COLOR0,
    out float2 oTexCoord0				: TEXCOORD0,
	out float oDist						: TEXCOORD1,
	out float3 oPos						: TEXCOORD2,
	out float3 oView					: TEXCOORD3,
	out float3 oNormWS					: TEXCOORD4,
	out float3 oPosWS					: TEXCOORD5)
{
	oColor0 = g_GlowColor;

	oDist = distance( iPosition, g_ImpactPosition );
	oPos = iPosition;
	
	//Vertex Position
	float3 position = iPosition;
	
	oPosition = mul( float4( position, 1.f ), g_WorldViewProjection );
   
    oTexCoord0 = iTexCoord0; 
    
	float3 tangentInWorldSpace = normalize(mul(iTangent, (float3x3)g_World));
	float3 normalInWorldSpace = normalize(mul(iNormal, (float3x3)g_World));
	oNormWS = normalInWorldSpace;
	float3 biTangentInWorldSpace = cross(normalInWorldSpace, tangentInWorldSpace);
    float3x3 tangentMatrix = transpose(float3x3(tangentInWorldSpace, biTangentInWorldSpace, normalInWorldSpace));
    
	float3 positionInWorldSpace = mul(float4(position, 1.f), g_World).xyz;
	oPosWS = positionInWorldSpace;
	float3 positionInTangentSpace = mul(positionInWorldSpace, tangentMatrix);
	oView = normalize(-positionInTangentSpace);
}

float4 GetPixelColor( float4 iColor, float2 iTexCoord, float iDist, float3 iPos, float3 iView, float3 iNormWS, float3 iPosWS)
{

	
	float3	ImpactPos		= iPos - g_ImpactPosition;
	float	ImpactSqr		= dot(ImpactPos, ImpactPos);
	if(ImpactSqr > g_InternalHitRadius * g_InternalHitRadius)
		return 0;
	const float inverseScale = rcp(g_InternalHitRadius);

	float ImpactDist = sqrt(ImpactSqr) * inverseScale;//linear distance
	float LinearFade = (1 - ImpactDist);
	
	float A			= .8;
	float4 X; 
	float4 y; 
	float t			= g_Time * RIPPLE_SPEED;
	
	float3 ImpactPosX = ImpactPos - float3(RIPPLE_DERIVATIVE_WIDTH, 0.0, 0.0);
	float3 ImpactPosY = ImpactPos - float3(0.0, RIPPLE_DERIVATIVE_WIDTH, 0.0);
	float3 ImpactPosZ = ImpactPos - float3(0.0, 0.0, RIPPLE_DERIVATIVE_WIDTH);
	
	float4 g	= float4(float3(length(ImpactPosX), length(ImpactPosY), length(ImpactPosZ)) * inverseScale, ImpactDist)  * RIPPLE_SCALE; 
	
	float4 r;
	
	#ifdef USE_TURBULENCE
		r.x			= tex3Dlod(NoiseSampler, float4(ImpactPosX.xyz * 0.005234 + g_Time * TURBULENCE_SPEED, 0)).r;
		r.y			= tex3Dlod(NoiseSampler, float4(ImpactPosY.yzx * 0.004973 - g_Time * TURBULENCE_SPEED, 0)).r;
		r.z			= tex3Dlod(NoiseSampler, float4(ImpactPosZ.zxy * 0.005011 + g_Time * TURBULENCE_SPEED, 0)).r;
		r.a			= tex3Dlod(NoiseSampler, float4(ImpactPos.zyx  * 0.004948 - g_Time * TURBULENCE_SPEED, 0)).r;
		r 			= float4(r.a - r.xyz, (rcp(1 + Square((r.x - r.y) * 20.0 * LinearFade)) * rcp(1 + Square((r.z - r.w) * 25.0 * LinearFade))) * 4 * LinearFade);
	#else
		r 			= 0;
	#endif
	
	float4 Y	= 0.; 
    for(float k = 1.; k < RIPPLE_REFINEMENT_ITERATION; k++)
	{
		
     //   Y += y = W(abs(uv.x), k, sqrt(k))/k;   // dispertion for capillary waves
		Y += y = W(g, k, rsqrt(k)) / k;// dispertion for gravity waves
    }
	Y *= IMPACTFLARE_SCALE;

	float3 Nws			= normalize(iNormWS + (Y.aaa - Y.xyz) + r.xyz * TURBULENCE_INTENSITY);
//	return float4(r.www, 1);
	float3 Vws 			= normalize(-iPosWS);
	float NoV 			= dot(Vws, Nws);

	float3 Rws = -(Vws - 2.0 * Nws * NoV);	
	float3 color = texCUBElod(TextureEnvironmentCubeSampler, float4(Rws, 0)).rgb;	//sure it'll oversample a bit but the mips don't work for grad sampling.
	
//	return float4(color, /*saturate(LinearFade * (1-Pow5(1-saturate(abs(dot(Vws, iNormWS))))) * */rcp(1.0 + Square(g_Time * 2.0)));
	

	float lambda = .35f;
	float3 shieldColor = iColor.rgb;
	shieldColor = normalize(shieldColor.rgb);
	shieldColor *= length(color);
	color = color * lambda + (1.f - lambda) * shieldColor;
	
	//can this line be reduced?
	color *= 1.0 + r.a + max(0.0, (rcp(1 + Square(ImpactDist * IMPACTFLARE_ATTENUATION)) - rcp(1.0 + Square(IMPACTFLARE_ATTENUATION))) * IMPACTFLARE_INTENSITY * rcp(0.1 + Square(g_Time * IMPACTFLARE_PULSE_SPEED)));
	
	return float4(color, LinearFade * rcp(1.0 + Square(g_Time * 2.0)) * 0.85);

}

void
RenderScenePS( 
	float4 iColor					: COLOR0,
	float2 iTexCoord0				: TEXCOORD0,
	float iDist						: TEXCOORD1,
	float3 iPos						: TEXCOORD2,
	float3 iView					: TEXCOORD3,
	float3 iNormWS					: TEXCOORD4,
	float3 iPosWS					: TEXCOORD5,
	out float4 oColor0				: COLOR0 )
{ 
	oColor0 = GetPixelColor( iColor, iTexCoord0, iDist, iPos, iView, iNormWS, iPosWS);
}

technique RenderWithoutVertexShader
{
    pass Pass0
    {   	        
        VertexShader		= NULL;
		PixelShader			= NULL;
		
		AlphaBlendEnable	= true;
		SrcBlend			= srcalpha;
		DestBlend			= invsrcalpha;
		
		ZEnable				= true;
		ZWriteEnable		= false;		
    }
}

technique RenderWithoutPixelShader
{
    pass Pass0
    {   	        
        VertexShader		= compile vs_1_1 RenderSceneVS_WithoutPixelShader();
		PixelShader			= NULL;
		
		AlphaTestEnable		= false;
		
		ZEnable				= true;
		ZWriteEnable		= false;		
    }
}

technique RenderWithPixelShader
{
    pass Pass0
    {          
        VertexShader		= compile vs_1_1 RenderSceneVS_WithPixelShader();
        PixelShader			= compile ps_3_0 RenderScenePS();
        
        AlphaTestEnable		= false;
//		DestBlend			= ONE;
		ZEnable				= true;
		ZWriteEnable		= false;
		CullMode 			= None;
     }
}
