//math constants, don't touch :P
#define PI 		3.1415926535897932384626433832795
#define INVPI 	0.3183098861837906715377675267450
#define TWOPI	6.283185307179586476925286766559

float4x4		g_WorldViewProjection	: WorldViewProjection;

texture			g_TextureDiffuse0		: Diffuse;
texture			g_TextureNoise3D;

float			g_Time;

float postProcessColorMultiplier = 1.3f;

sampler TextureDiffuse0Sampler = 
sampler_state
{
    Texture		= < g_TextureDiffuse0 >;    
    MipFilter	= LINEAR;
    MinFilter	= LINEAR;
    MagFilter	= LINEAR;
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

void
RenderSceneVS( 
	float3 iPosition			: POSITION, 
	float3 iNormal				: NORMAL,
	float2 iTexCoord0			: TEXCOORD0,
	
	out float4 oPosition		: POSITION,
	out float4 oColor0			: COLOR0,
	out float2 oTexCoord0		: TEXCOORD0,
	out float3 oPosObj 			: TEXCOORD1,
	out float3 oNormalObj		: TEXCOORD2,
	out float3 oViewObj			: TEXCOORD3)
{
	oPosition = mul( float4( iPosition, 1.0f ), g_WorldViewProjection );
    oColor0	= float4( 0, 0, 0, 0 );
    
    oTexCoord0 = iTexCoord0;
	oPosObj = iPosition;
	oNormalObj = iNormal;
	oViewObj = transpose(g_WorldViewProjection)._m30_m31_m32;//StarDock dammit, just define all the matrices as globals in the future so we don't needs this kind of gymnastics :P
}

// random stable 3D noise
float3 hash33(float3 p3)
{
	p3 = frac(p3 * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz + 19.19);
    return -1.0 + 2.0 * frac(float3((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y, (p3.y + p3.z) * p3.x));
}

float simplex_noise(float3 p)
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

float simplex_refine(float3 n, int iterations, float amplitude) {
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
	return x * x;
}
float selfDotInv(float3 x)
{
	return rcp(1.0 + dot(x , x));
}

void UpdateUVs(in float3 P, inout float2 UV0, inout float2 UV1)	
{
	float PSqrXY = dot(P.xz, P.xz);
	float PSqrZnorth = Square((1.0 + P.y) * 2.0);
	float PSqrZsouth = Square((1.0 - P.y) * 2.0);
	UV0 = P.xz / sqrt(PSqrZnorth + PSqrXY);
	UV1 = P.xz / sqrt(PSqrZsouth + PSqrXY);
}
	
float4 GetPixelColor( float4 iColor, float2 iTexCoord, float3 iPosObj, float3 iNormalObj, float3 iViewObj )
{
	float3 view = normalize(-iViewObj);
	float3 normal = normalize(iPosObj);

	float2 UV0 = 0;
	float2 UV1 = 0;
	UpdateUVs(iNormalObj, UV0, UV1);
	float4 diffuse = lerp(tex2D(TextureDiffuse0Sampler, UV0), tex2D(TextureDiffuse0Sampler, 0.5-UV1), saturate(iTexCoord.y * 11 - 5));
	
	float distortWeight = Square(1.0 - dot((float3)rcp(3.0), diffuse.rgb));
	//return float4(iNormalObj * 0.5 + 0.5, 1);
	//float variation = tex3Dlod(NoiseSampler, float4(iNormalObj * 8 + frac(g_Time * float3(0.01, 0.0134, 0.0976)), 0)).x - tex3Dlod(NoiseSampler, float4(iNormalObj * 8 - frac(g_Time * float3(0.0124, 0.0114, 0.0973)), 0)).x;
	float distortionBase = simplex_refine(iNormalObj * 6 + g_Time * float3(0.01, 0.0134, 0.00976), 3, 0.5) - simplex_refine(iNormalObj * 6 - g_Time * float3(0.0124, 0.0114, 0.00973), 3, 0.5);
	float3 distortion_vector = float3(cos(distortionBase * TWOPI), 2.0 * distortionBase, sin(distortionBase * TWOPI)) * 0.05;
	
	float starBase = selfDotInv(simplex_refine(iNormalObj * 21 + g_Time * float3(0.02, 0.0234, 0.0276) * 3 + distortion_vector, 5, 0.701) - simplex_refine(iNormalObj * 19 - g_Time * float3(0.0224, 0.0214, 0.0273) * 3 + distortion_vector, 5, 0.701));
	float NoV = abs(dot(view, normal));
//	float4 LUTCord = float4(saturate(float2(NoV, starBase * 0.5 + distortionBase * 0.25 + 0.5)) * 0.96875 + 0.015625, 0, 0);
//	float3 starSurfaceColor = tex2Dlod(TextureDiffuse0Sampler, LUTCord).rgb;
	

	UpdateUVs(iNormalObj + distortion_vector * 0.05, UV0, UV1);
	diffuse = lerp(tex2D(TextureDiffuse0Sampler, UV0), tex2D(TextureDiffuse0Sampler, 0.5-UV1), saturate(iTexCoord.y * 11 - 5));
	diffuse.rgb = pow(diffuse.rgb, starBase * NoV) * (1.0 + (1-NoV));
	diffuse.a *= NoV;

	
//	return float4(starSurfaceColor, 1);

/*		
	float temperatureStart = 
		(0.3 * diffuse.r) + 
		(0.59 * diffuse.g) + 
		(0.14 * diffuse.b);

	float maxTime = 6.0f;
	float minTime = 5.9f;
	float timeScale = lerp( maxTime, minTime, temperatureStart );
	
	//keep time from getting too big
	float time = g_Time % maxTime + 750;
		
	float noiseTime = time / timeScale;
	float3 noiseIndex = float3(iTexCoord.x, iTexCoord.y, noiseTime);
	float domainPhaseShift = tex3D(NoiseSampler, noiseIndex);
	
	float2 shiftedCoord;
	shiftedCoord.x = iTexCoord.x + (cos(domainPhaseShift * 6.28) / 100.f);
	shiftedCoord.y = iTexCoord.y + (sin(domainPhaseShift * 6.28) / 100.f);
	diffuse = tex2D(TextureDiffuse0Sampler, shiftedCoord + starBase * (1.0/256.0));
*/

	return diffuse;

}

void
RenderScenePS_Color( 
	float4 iColor				: COLOR,
	float2 iTexCoord0			: TEXCOORD0,
	float3 iPosObj				: TEXCOORD1,
	float3 iNormalObj			: TEXCOORD2,
	float3 iViewObj				: TEXCOORD3,
	out float4 oColor0			: COLOR0 ) 
{ 
	oColor0 = GetPixelColor( iColor, iTexCoord0, iPosObj, iNormalObj, iViewObj);
//	oColor0 *= postProcessColorMultiplier;
}

technique RenderWithoutPixelShader
{
    pass Pass0
    {   	        
        VertexShader = compile vs_1_1 RenderSceneVS();
        PixelShader = NULL;
		Texture[ 0 ] = < g_TextureDiffuse0 >;
		ZEnable = true;
		ZWriteEnable = true;
		AlphaTestEnable = FALSE;
        AlphaBlendEnable = TRUE;
		SrcBlend = ONE;
		DestBlend = ZERO;
    }
}

technique RenderWithPixelShader
{
    pass Pass0
    {          
        VertexShader = compile vs_1_1 RenderSceneVS();
        PixelShader = compile ps_3_0 RenderScenePS_Color();
		ZEnable = true;
		ZWriteEnable = true;
		AlphaTestEnable = FALSE;
        AlphaBlendEnable = TRUE;
		SrcBlend = ONE;
		DestBlend = ZERO;
    }
}
