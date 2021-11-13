//math constants
#define SF (1.0/float(0xffffffffU))
#define HALFPI 1.5707963267948966192313216916398
#define PI 		3.1415926535897932384626433832795
#define INVPI 	0.3183098861837906715377675267450
#define TWOPI 	6.2831853071795864769252867665590
#define INVTWOPI 	0.1591549430918953357688837633725

float4x4 g_World : World;
float4x4 g_WorldViewProjection : WorldViewProjection;

texture	g_TextureDiffuse0 : Diffuse;
texture	g_TextureSelfIllumination;
texture g_TextureTeamColor: Displacement;
float colorMultiplier = 4.f;

float3 g_Light0_Position: Position = float3( 0.f, 0.f, 0.f );

float4 g_MaterialAmbient:Ambient;
float4 g_MaterialSpecular:Specular;
float4 g_MaterialEmissive:Emissive;
float4 g_MaterialDiffuse:Diffuse;

float g_Time;

sampler TextureColorSampler = sampler_state{
    Texture = <g_TextureDiffuse0>;    
    AddressU = WRAP;        
    AddressV = WRAP;
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
};

sampler TextureDataSampler = sampler_state{
    Texture = <g_TextureSelfIllumination>;    
    AddressU = WRAP;        
    AddressV = WRAP;
	MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
};
sampler TextureDisplacementSampler = sampler_state
{
    Texture	= <g_TextureTeamColor>;
    AddressU = WRAP;        
    AddressV = WRAP;
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
};
/*
sampler CloudLayerSampler = sampler_state
{
    Texture	= <g_TextureDiffuse2>;
    AddressU = WRAP;        
    AddressV = WRAP;
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;	
};
*/

struct VsOutput
{
	float4 position	: POSITION;
	float2 texCoord : TEXCOORD0;
	float3 normal : TEXCOORD1;
	float3 lightDir : TEXCOORD2;
	float3 normalObj : TEXCOORD3;
};

VsOutput RenderSceneVS( 
	float3 position : POSITION, 
	float3 normal : NORMAL,
	float3 tangent : TANGENT,
	float2 texCoord : TEXCOORD0 )
{
	VsOutput output;  
	output.position = mul(float4(position, 1.0f), g_WorldViewProjection);
    output.texCoord = texCoord; 
    output.normal = mul(normal, (float3x3)g_World);
	output.normalObj = normal;
    output.lightDir = normalize(g_Light0_Position.xyz - output.position.xyz);
    return output;
}

float Square(float X)
{
	return X * X;
}
float2 Square(float2 X)
{
	return X * X;
}
float3 Square(float3 X)
{
	return X * X;
}
float4 Square(float4 X)
{
	return X * X;
}	
/*	
inline float MapClouds(float c)
{
	return (1.0 - exp(-Square(c) * g_MaterialAmbient.a * 5.0));
}

inline float CloudSim(float2 uv, float2 duvx, float2 duvy, float speed)
{
	float4 baseSample = tex2Dgrad(CloudLayerSampler, uv, duvx, duvy);
	float2 flowDir = (baseSample.yw - 0.5) * float2(1.0, -1.0) * 0.125;
	
	float swap_a = frac(speed);
	float swap_b = frac(speed + 0.5);
	
	float2 offset_a = (flowDir * swap_a);
	float cloud_detail_a = (tex2Dgrad(CloudLayerSampler, (uv - offset_a * 1.5) * 3, duvx * 2, duvy * 2).r + tex2Dgrad(CloudLayerSampler, (uv - offset_a * 2.0) * 4 + 0.5, duvx * 3, duvy * 3).r);
	float cloud_a = tex2Dgrad(CloudLayerSampler, uv - offset_a, duvx, duvy).b;
	cloud_a += cloud_detail_a * 0.25;
	
	float2 offset_b = (flowDir * swap_b);
	float cloud_detail_b = (tex2Dgrad(CloudLayerSampler, (uv - offset_b * 1.5) * 3, duvx * 2, duvy * 2).r + tex2Dgrad(CloudLayerSampler, (uv - offset_b * 2.0) * 4 + 0.5, duvx * 3, duvy * 3).r);
	float cloud_b = tex2Dgrad(CloudLayerSampler, uv - offset_b, duvx, duvy).b;
	cloud_b += cloud_detail_b * 0.25;
	
	return 1.0 - rcp(1.0 + Square(lerp(cloud_a, cloud_b, abs(swap_a * 2.0 - 1.0))));
}
*/

float2 GetEquirectangularUV(float3 normalObj, inout float2 duvx, inout float2 duvy)
{
	float2 uv;
	
	float4 uvBasis;
	uvBasis.xy = atan2(float2(normalObj.x, normalObj.x), float2(normalObj.z, -normalObj.z)) * INVTWOPI;
	uvBasis.x = -uvBasis.x;
	uvBasis.z = acos(normalObj.y) * INVPI;
	uvBasis.w = -normalObj.y * INVPI;
	
	uv = uvBasis.xz;
	
	// uses a small bias to prefer the first 'UV set'
	uv.x = (fwidth(uv.x) - 0.001 < fwidth(frac(uv.x)) ? uv.x : frac(uv.x)) + 0.5;

	duvx = float2(	normalObj.z > 0.0 ? ddx(uvBasis.x) : ddx(uvBasis.y), 
					abs(normalObj.y) > 0.025 ? ddx(uvBasis.z) : ddx(uvBasis.w));
	duvy = float2(	normalObj.z > 0.0 ? ddy(uvBasis.x) : ddy(uvBasis.y),
						abs(normalObj.y) > 0.025 ? ddy(uvBasis.z) : ddy(uvBasis.w));
	return uv;
}

float4 RenderScenePS(VsOutput input) : COLOR0
{
	const bool OceanMode = int(g_MaterialSpecular.a) == 1;//planets that got oceans, support citylights too
	const bool VolcanoMode = int(g_MaterialSpecular.a * 4) == 1;//planet that are lave based, no citylight
	const bool GasGiantMode = int(g_MaterialSpecular.a * 4) == 2;//planet that are lave based, no citylight

	float2 duvx;
	float2 duvy;
	float2 uv								= GetEquirectangularUV(normalize(input.normalObj), duvx, duvy);		
	
/*
	float2 uv 				= input.texCoord;
	float2 duvx				= ddx(uv);
	float2 duvy				= ddy(uv);
*/	
	float cloudsSpeed		= g_Time * 0.05;
	
	float volcanoSmokeMask	= 1;
	if(VolcanoMode)
		volcanoSmokeMask		-= tex2D(TextureDisplacementSampler, uv).g;
	
	float CloudAbsorption		= 1;
//	if(!GasGiantMode)
//	{
//		float cloudSample 		= MapClouds(CloudSim(/*UVcloud*/ uv, duvx, duvy, cloudsSpeed));
//		CloudAbsorption 		*= 1.0 - cloudSample * cloudSample;
//	}
//

//	float4 lightSideSample = tex2D(TextureColorSampler, input.texCoord);
//	float4 darkSideSample = tex2D(TextureDataSampler, input.texCoord);
	float3 emissive = tex2D(TextureDataSampler, uv).rgb;
	emissive *= emissive * emissive * volcanoSmokeMask * CloudAbsorption;
//	float dotLight = max(dot(input.lightDir, input.normal), 0.f);
	float NightMask = smoothstep(0.1, -0.25, dot(input.lightDir, input.normal));
	return float4(emissive * NightMask * colorMultiplier, 1);
//	float4 finalColor = 0.f;
	
//	finalColor += lightSideSample * NightMask;
//	finalColor += darkSideSample * (1.f - NightMask);
	
//	return finalColor * darkSideSample.a * colorMultiplier;
}

technique RenderWithPixelShader
{
    pass Pass0
    {          
        VertexShader = compile vs_1_1 RenderSceneVS();
        PixelShader = compile ps_3_0 RenderScenePS();
		AlphaTestEnable = FALSE;
        AlphaBlendEnable = TRUE;
		SrcBlend = ONE;
		DestBlend = ZERO;
		ZEnable = TRUE;
		ZWriteEnable = TRUE;			   
    }
}
