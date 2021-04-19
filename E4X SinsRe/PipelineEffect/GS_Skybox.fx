//math constants, don't touch :P
#define PI 		3.1415926535897932384626433832795
#define INVPI 	0.3183098861837906715377675267450
#define TWOPI	6.283185307179586476925286766559

//Settings
//#define USE_MACRO_DISTORTION
#define MACRO_DISTORTION_SCALE 0.5
#define MACRO_DISTORTION_INTENSITY 0.5
#define MACRO_DISTORTION_WEIGHT 0.25
#define MACRO_OCTAVES 4 //less is cheaper

//#define USE_MICRO_DISTORTION
#define MICRO_DISTORTION_SCALE 4.0
#define MICRO_DISTORTION_INTENSITY 0.005
#define MICRO_OCTAVES 4 //runds *2!!, less is cheaper

//#define USE_COLOR_REFINEMENT
#define COLOR_REFINEMENT_INTENSITY 0.15

//#define USE_BRIGHTPASS
#define BRIGHTPASS_FILTERWIDTH 16384.0
#define BRIGHTPASS_INTENSITY 1.0

//#define USE_AUTOLEVEL

shared float4x4	g_ViewProjection : ViewProjection;
float4x4 g_World;

texture	g_TextureDiffuse0 : Diffuse;
texture	g_TextureEnvironmentCube : Environment;
texture g_EnvironmentIllumination : Environment;

float colorMultiplier = 1.f;

sampler TextureDiffuse0Sampler = 
sampler_state
{
    Texture		= < g_TextureDiffuse0 >;    
    MipFilter	= LINEAR;
    MinFilter	= LINEAR;
    MagFilter	= LINEAR;
};

samplerCUBE TextureEnvironmentCubeSampler = sampler_state{
    Texture = <g_TextureEnvironmentCube>;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = LINEAR;
    AddressU = CLAMP;
    AddressV = CLAMP;
    AddressW = CLAMP;
	 
};

samplerCUBE EnvironmentIlluminationCubeSampler = sampler_state{
    Texture = <g_EnvironmentIllumination>;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = LINEAR;
    AddressU = CLAMP;
    AddressV = CLAMP;
    AddressW = CLAMP;

};

float4 SRGBToLinear(float4 color)
{	
	#ifdef CHEAPLINEARCOLOR
		return float4(color.rgb * color.rgb, color.a);
	#else
		return float4(color.rgb * (color.rgb * (color.rgb * 0.305306011f + 0.682171111f) + 0.012522878f), color.a);
	#endif	
}

float4 LinearToSRGB(float4 color)
{
	float3 S1 = sqrt(color.rgb);
	#ifdef CHEAPLINEARCOLOR
		return float4(S1, color.a);
	#else		
		float3 S2 = sqrt(S1);
		float3 S3 = sqrt(S2);
		return float4(0.662002687 * S1 + 0.684122060 * S2 - 0.323583601 * S3 - 0.225411470 * color.rgb, color.a);
	#endif		
}

// random stable 3D noise
inline float3 hash33(float3 p3)
{
	p3 = frac(p3 * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz + 19.19);
    return -1.0 + 2.0 * frac(float3((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y, (p3.y + p3.z) * p3.x));
}

inline float simplex_noise(float3 p)
{
    float K1 = 0.333333333;
    float K2 = 0.166666667;
    
    float3 i = floor(p + (p.x + p.y + p.z) * K1);
    float3 d0 = p - (i - (i.x + i.y + i.z) * K2);
    
    float3 e = step((float3)0.0, d0 - d0.yzx);
	float3 i1 = e * (1.0 - e.zxy);
	float3 i2 = 1.0 - e.zxy * (1.0 - e);
    
    float3 d1 = d0 - (i1 - 1.0 * K2);
    float3 d2 = d0 - (i2 - 2.0 * K2);
    float3 d3 = d0 - (1.0 - 3.0 * K2);
    
    float4 h = max(0.6 - float4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.0);
    float4 n = h * h * h * h * float4(dot(d0, hash33(i)), dot(d1, hash33(i + i1)), dot(d2, hash33(i + i2)), dot(d3, hash33(i + 1.0)));
    
    return dot((float4)31.316, n);
}

inline float simplex_refine(float3 n, int iterations, float amplitude) {
	float total = 0.0;
	for (int i = 0; i < iterations; i++) {
		total += simplex_noise(n) * amplitude;
		n += n;
		amplitude *= 0.701;
	}
	return total;
}

float Square(float x)
{
	return x*x;
}
float selfDotInv(float3 x)
{
	return rcp(1.0 + dot(x , x));
}

float2 rotate(float2 rotation, float rate)
{
	return float2(dot(rotation,  float2(cos(rate),  -sin(rate))), dot(rotation,  float2(sin(rate),  cos(rate))));
}

void RenderSceneVS( 
	float3 iPosition : POSITION, 
	float3 iNormal : NORMAL,
	float2 iTexCoord0 : TEXCOORD0,
	out float4 oPosition : POSITION,
	out float3 oNormal	 : NORMAL,	
    out float4 oColor0 : COLOR0,
    out float2 oTexCoord0 : TEXCOORD0 )
{
	oPosition = mul(float4(iPosition, 1), g_World);
	oPosition = mul(oPosition, g_ViewProjection);
	oNormal = iPosition.xyz * 0.1;
    oColor0 = float4(1, 1, 1, 1) * colorMultiplier;
    oTexCoord0 = iTexCoord0;
}

void
RenderScenePS(
	float3 iNormal : NORMAL,
	float4 iColor : COLOR,
	float2 iTexCoord0 : TEXCOORD0,
	out float4 oColor0 : COLOR0 ) 
{ 
	oColor0 = tex2D(TextureDiffuse0Sampler, iTexCoord0);
	
	float distortWeight = Square(1.0 - dot((float3)rcp(3.0), oColor0.rgb));
	
	#ifdef USE_MACRO_DISTORTION
	//	float2 uv = float2(atan2(iNormal.x, iNormal.z) * INVPI * 0.5 + 0.5, iNormal.y * 0.5 + 0.5);
		float distortionBase = distortWeight - simplex_refine(iNormal * MACRO_DISTORTION_SCALE, MACRO_OCTAVES, 0.5) * MACRO_DISTORTION_INTENSITY;
		float3 distortion_vector = float3(cos(distortionBase * TWOPI), 2.0 * distortionBase, sin(distortionBase * TWOPI)) * MACRO_DISTORTION_WEIGHT;
	#else
		float3 distortion_vector = 0;
	#endif
	#ifdef USE_MICRO_DISTORTION
		float2 simplex = (float2(simplex_refine(iNormal * MICRO_DISTORTION_SCALE + distortion_vector, MICRO_OCTAVES, 0.5), simplex_refine(iNormal * MICRO_DISTORTION_SCALE + 8.0 + distortion_vector, MICRO_OCTAVES, 0.5)) - 0.5);
		iTexCoord0 += simplex * (distortWeight * MICRO_DISTORTION_INTENSITY);
		float2 duvx = ddx(iTexCoord0);
		float2 duvy = ddy(iTexCoord0);
		oColor0 = tex2Dgrad(TextureDiffuse0Sampler, iTexCoord0, duvx, duvy);
	#endif
	#ifdef USE_BRIGHTPASS
		#ifndef USE_MICRO_DISTORTION
			float2 duvx = ddx(iTexCoord0);
			float2 duvy = ddy(iTexCoord0);
		#endif

		float FilterWidth = (1.0 / BRIGHTPASS_FILTERWIDTH);
		float4 EdgePass = tex2Dgrad(TextureDiffuse0Sampler, iTexCoord0 - FilterWidth, duvx, duvy) - tex2Dgrad(TextureDiffuse0Sampler, iTexCoord0 + FilterWidth, duvx, duvy);
		oColor0.rgb += EdgePass.rgb * BRIGHTPASS_INTENSITY * dot(oColor0.rgb, oColor0.rgb);
	#endif
	#ifdef USE_COLOR_REFINEMENT
		oColor0.rgb *= 1.0 + (simplex.x * simplex.x - simplex.y * simplex.y) * distortWeight * COLOR_REFINEMENT_INTENSITY;
	#endif
	
	#ifdef USE_AUTOLEVEL
		oColor0.rgb = 1-exp(-oColor0.rgb);
	#endif
	oColor0.a = oColor0.a;
}

technique RenderWithoutPixelShader
{
    pass Pass0
    {   	        
        VertexShader = compile vs_3_0 RenderSceneVS();
        PixelShader = NULL;
        ZEnable = FALSE;
        ZWriteEnable = FALSE;
		AlphaTestEnable = TRUE;
		AlphaBlendEnable = TRUE;
		SrcBlend = ONE;
		DestBlend = INVSRCALPHA;            
		Texture[0] = < g_TextureDiffuse0 >;        
    }
}

technique RenderWithPixelShader
{
    pass Pass0
    {   	        
        VertexShader = compile vs_3_0 RenderSceneVS();
        PixelShader = compile ps_3_0 RenderScenePS();
        ZEnable = FALSE;
        ZWriteEnable = FALSE;
		AlphaTestEnable = FALSE;
		AlphaBlendEnable = TRUE;
		SrcBlend = ONE;
		DestBlend = INVSRCALPHA;            
    }
}

