//math constants, don't touch :P
#define PI 		3.1415926535897932384626433832795
#define INVPI 	0.3183098861837906715377675267450
#define TWOPI	6.283185307179586476925286766559

//Settings
#define MACRO_DISTORTION_SCALE 0.1
#define MACRO_DISTORTION_INTENSITY 0.025
#define MACRO_DISTORTION_WEIGHT 8.0
#define MICRO_DISTORTION_SCALE 4.0
#define MICRO_DISTORTION_INTENSITY 2.0
#define COLOR_REFINEMENT_INTENSITY 2.5

shared float4x4	g_ViewProjection : ViewProjection;
float4x4 g_World;

texture	g_TextureDiffuse0 : Diffuse;
texture	g_TextureSelfIllumination;
texture	g_TextureEnvironmentCube : Environment;
texture g_EnvironmentIllumination : Environment;

float4 g_MaterialAmbient:Ambient;
float4 g_MaterialSpecular:Specular;
float4 g_MaterialEmissive:Emissive;
float4 g_MaterialDiffuse:Diffuse;

//texture g_TextureNoise3D;//doesn't work, not available

float colorMultiplier = 1.f;

sampler TextureDiffuse0Sampler = 
sampler_state
{
    Texture		= < g_TextureDiffuse0 >;    
    MipFilter	= LINEAR;
    MinFilter	= LINEAR;
    MagFilter	= LINEAR;
};

sampler TextureDataSampler = sampler_state
{
    Texture = <g_TextureSelfIllumination>;
    AddressU = WRAP;        
    AddressV = WRAP;
    Filter = LINEAR;
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
/*
sampler NoiseSampler = sampler_state 
{
    texture = <g_TextureNoise3D>;
    AddressU = WRAP;        
    AddressV = WRAP;
	AddressW = WRAP;
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
};
*/

inline float4 SRGBToLinear(float4 color)
{	
	#ifdef CHEAPLINEARCOLOR
		return float4(color.rgb * color.rgb, color.a);
	#else
		return float4(color.rgb * (color.rgb * (color.rgb * 0.305306011f + 0.682171111f) + 0.012522878f), color.a);
	#endif	
}

inline float4 LinearToSRGB(float4 color)
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

inline float Square(float x)
{
	return x*x;
}
inline float selfDotInv(float3 x)
{
	return rcp(1.0 + dot(x , x));
}

inline float2 rotate(float2 rotation, float rate)
{
	return float2(dot(rotation,  float2(cos(rate),  -sin(rate))), dot(rotation,  float2(sin(rate),  cos(rate))));
}

void UpdateUVs(in float3 P, inout float2 UV0, inout float2 UV1)	
{
	float PSqrXY = dot(P.xz, P.xz);
	float PSqrZnorth = Square((1.0 + P.y) * 2.0);
	float PSqrZsouth = Square((1.0 - P.y) * 2.0);
	UV0 = P.xz / sqrt(PSqrZnorth + PSqrXY);
	UV1 = P.xz / sqrt(PSqrZsouth + PSqrXY);
}

inline float4 GetNoise(float4 UVs, float scale, float grad)
{
	return lerp(tex2Dlod(TextureDataSampler, float4((UVs.zw - 0.43167) * scale, 0, 0)), tex2Dlod(TextureDataSampler, float4((UVs.xy + 0.3147) * scale, 0, 0)), grad) - 0.5;	
}

inline float4 GetNoiseRefine(float4 UVs, float scale, float grad, int iterations, float amplitude)
{
	float2x2 m = float2x2(1.6,  1.2, -1.2,  1.6);
	float4 total = 0.0;
	float base = 1.0;
	for (int i = 0; i < iterations; i++) {
		total += GetNoise(UVs, scale, grad) * base;
		UVs = float4(mul(UVs.xy, m), mul(UVs.zw, m));
		base *= amplitude;
	}
	return total;
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
	oColor0 = tex2Dlod(TextureDiffuse0Sampler, float4(iTexCoord0, 0, 0));
	
//	[branch]
	if(round(g_MaterialDiffuse.a) == 0)
	{
			
		
		float blend = saturate(iNormal.y * 4 + 0.5);
		float3 Original = oColor0.rgb;
		float distortWeight = dot((float3)rcp(3.0), oColor0.rgb) * MACRO_DISTORTION_WEIGHT;
		float4 UVs = 0;
		UpdateUVs(iNormal, UVs.xy, UVs.zw);

		float4 distortionBase = GetNoiseRefine(UVs, MACRO_DISTORTION_SCALE, blend, 3, 0.701);;
		float rotationBase = ((distortionBase.x - distortionBase.w) + (distortionBase.y - distortionBase.z)) * distortWeight;

		float3 distortion_vector = float3(cos(rotationBase * TWOPI), 2.0 * rotationBase, sin(rotationBase * TWOPI)) * MACRO_DISTORTION_INTENSITY;
		UpdateUVs(iNormal + distortion_vector, UVs.xy, UVs.zw);
		float4 simplex = GetNoiseRefine(UVs, MICRO_DISTORTION_SCALE, blend, 4, 0.701);

		float2 pixelSize = rcp(float2(4096.0, 2048.0)) * MICRO_DISTORTION_INTENSITY;
		oColor0 = tex2Dlod(TextureDiffuse0Sampler, float4(iTexCoord0 + pixelSize * (simplex.xy - simplex.zw), 0, 0));
		
		oColor0.rgb += abs(oColor0.rgb - Original) * (1.0 + dot(simplex, (float4)1.0)) * (1.0 - oColor0.a) * COLOR_REFINEMENT_INTENSITY;

		oColor0.rgb = 1-exp(-oColor0.rgb);
	}
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

