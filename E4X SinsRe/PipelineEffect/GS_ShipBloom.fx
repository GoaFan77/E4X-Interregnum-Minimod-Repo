#define PBR 
#define PI 		3.1415926535897932384626433832795
#define INVPI 	0.3183098861837906715377675267450
#define FRESNEL_CLAMP 0.0001 //variable that avoids caps a potenital near 0 division leading to infinitely bright fresnel effect
#define LIGHTINTENSITY_FAR 0.8			// light intensity far from stars
#define LIGHTINTENSITY_NEAR	8.0			// light intensity close to stars
#define LIGHTINTENSITY_POST 0.8			//light intensity after remapping to 0-1 ranges

float4x4 g_World : World;
float4x4 g_WorldViewProjection : WorldViewProjection;

texture	g_TextureDiffuse0 : Diffuse;
texture g_TextureTeamColor;
texture	g_TextureSelfIllumination;
float4 g_TeamColor;
float colorMultiplier = 1000.f;
float3 g_Light0_Position: Position = float3( 0.f, 0.f, 0.f );
float4 g_Light0_DiffuseLite: Diffuse;// = float4( 1.f, 1.f, 1.f, 1.f );

sampler TextureColorSampler = 
sampler_state
{
    Texture = < g_TextureDiffuse0 >;    
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
};

sampler TextureDataSampler = 
sampler_state
{
    Texture = < g_TextureSelfIllumination >;    
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
};

sampler TexturetTeamSampler = sampler_state{
    Texture = <g_TextureTeamColor>;    
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
};

struct VsOutput
{
	float4 Position	: POSITION;
	float2 TexCoord0 : TEXCOORD0;
	float3 Normal : TEXCOORD1;
	float3 Pos : TEXCOORD2;
};

VsOutput
RenderSceneVS( 
	float3 iPosition : POSITION, 
	float3 iNormal : NORMAL,
	float3 iTangent : TANGENT,
	float2 iTexCoord0 : TEXCOORD	)
{
	VsOutput o;  
	
	//Final Position
	o.Position = mul( float4( iPosition, 1.0f ), g_WorldViewProjection );
	
	//Texture Coordinates
    o.TexCoord0 = iTexCoord0; 
    o.Normal = mul( iNormal, (float3x3)g_World);
	o.Pos = mul(float4(iPosition, 1.f), g_World).xyz;

    return o;
}
struct PBRProperties
{
	float3 SpecularColor;
	float3 DiffuseColor;
	float Roughness;
};

struct Dots
{
	float NoV;
	float NoL;
	float VoL;
	float NoH;
	float VoH;
};

void GetDots(inout Dots Dot, float3 normal, float3 view, float3 light)
{
	Dot.NoL = dot(normal, light);
	Dot.NoV = dot(normal, view);
	Dot.VoL = dot(view, light);
	float distInvHalfVec = rsqrt(2.0 + 2.0 * Dot.VoL);
	Dot.NoH = ((Dot.NoL + Dot.NoV) * distInvHalfVec);
	Dot.VoH = (distInvHalfVec + distInvHalfVec * Dot.VoL);
}

inline float Square(float X)
{
	return X * X;
}
inline float2 Square(float2 X)
{
	return X * X;
}
inline float3 Square(float3 X)
{
	return X * X;
}
inline float4 Square(float4 X)
{
	return X * X;
}

inline float Pow3(float X)
{
	return Square(X) * X;
}
inline float2 Pow3(float2 X)
{
	return Square(X) * X;
}
inline float3 Pow3(float3 X)
{
	return Square(X) * X;
}
inline float4 Pow3(float4 X)
{
	return Square(X) * X;
}

inline float Pow4(float X)
{
	return Square(Square(X));
}
inline float2 Pow4(float2 X)
{
	return Square(Square(X));
}
inline float3 Pow4(float3 X)
{
	return Square(Square(X));
}
inline float4 Pow4(float4 X)
{
	return Square(Square(X));
}

inline float Pow5(float X)
{
	return Pow4(X) * X;
}	
inline float2 Pow5(float2 X)
{
	return Pow4(X) * X;
}	
inline float3 Pow5(float3 X)
{
	return Pow4(X) * X;
}	
inline float4 Pow5(float4 X)
{
	return Pow4(X) * X;
}	

inline float ToLinear(float aGamma)
{
	return pow(aGamma, 2.2);
}
inline float3 ToLinear(float3 aGamma)
{
	return pow(aGamma, (float3)2.2);
}

void GetSphereNoH(inout Dots Dot, float sphereSinAlpha)
{
	if(sphereSinAlpha > 0)
	{
		float sphereCosAlpha = sqrt(1.0 - Square(sphereSinAlpha));
	
		float RoL = 2.0 * Dot.NoL * Dot.NoV - Dot.VoL;
		if(RoL >= sphereCosAlpha)
		{
			Dot.NoH = 1;
			Dot.VoH = abs(Dot.NoV);
		}
		else
		{
			float distInvTR = sphereSinAlpha * rsqrt(1.0 - Square(RoL));
			float NoTr = distInvTR * (Dot.NoV - RoL * Dot.NoL);

			float VoTr = distInvTR * (2.0 * Square(Dot.NoV) - 1.0 - RoL * Dot.VoL);

			float NxLoV = sqrt(saturate(1.0 - Square(Dot.NoL) - Square(Dot.NoV) - Square(Dot.VoL) + 2.0 * Dot.NoL * Dot.NoV * Dot.VoL));

			float NoBr = distInvTR * NxLoV;
			float VoBr = distInvTR * NxLoV * 2.0 * Dot.NoV;

			float NoLVTr = Dot.NoL * sphereCosAlpha + Dot.NoV + NoTr;
			float VoLVTr = Dot.VoL * sphereCosAlpha + 1.0 + 	VoTr;

			float p = NoBr   * VoLVTr;
			float q = NoLVTr * VoLVTr;
			float s = VoBr   * NoLVTr;

			float xNum = q * (-0.5 * p + 0.25 * VoBr * NoLVTr);
			float xDenom = Square(p) + s * (s - 2.0 * p) + NoLVTr * ((Dot.NoL * sphereCosAlpha + Dot.NoV) * Square(VoLVTr) + q * (-0.5 * (VoLVTr + Dot.VoL * sphereCosAlpha) - 0.5));
			float TwoX1 = 2.0 * xNum / (Square(xDenom) + Square(xNum));
			float SinTheta = TwoX1 * xDenom;
			float CosTheta = 1.0 - TwoX1 * xNum;
			NoTr = CosTheta * NoTr + SinTheta * NoBr;
			VoTr = CosTheta * VoTr + SinTheta * VoBr;

			Dot.NoL = Dot.NoL * sphereCosAlpha + NoTr;
			Dot.VoL = Dot.VoL * sphereCosAlpha + VoTr;

			float distInvHalfVec = rsqrt(2.0 + 2.0 * Dot.VoL);
			Dot.NoH = saturate((Dot.NoL + Dot.NoV) * distInvHalfVec);
			Dot.VoH = saturate(distInvHalfVec + distInvHalfVec * Dot.VoL);
		}
	}
}
void SphereLight(PBRProperties Properties, float3 normal, float3 view, float3 lightPos, float lightRad, float4 lightColorIntensity, inout float3 specular, inout float3 diffuse, inout float attenuation, inout float NoL)
{

	float lightSqr				= dot(lightPos, lightPos);
	float lightDist				= sqrt(lightSqr);
	float lightFalloff			= rcp(1.0 + lightSqr);//regular pointlight
	
	float3 light				= lightPos / lightDist;
	NoL							= dot(normal, light);
	float sphereSinAlphaSqr		= saturate(Square(lightRad) * lightFalloff);
	//Spherical attenuation - but since we are stars and low HDR, we probably don't really get anything from it.
//	attenuation 				= ((abs(NoL) + lightRad) / (lightSqr / (1.0 + lightRad))) * lightColorIntensity.a;
//	attenuation 				= rcp(1.0 + (max(0.0, lightDist - lightRad))) * lightColorIntensity.a;
	#if 1
		#if 1 // Analytical solution above horizon
			
			// Patch to Sphere frontal equation ( Quilez version )
			float lightRadSqr = Square(lightRad);
			// Do not allow object to penetrate the light ( max )
			// Form factor equation include a (1 / FB_PI ) that need to be cancel
			// thus the " FB_PI *"
			float illuminance = PI * (lightRadSqr / (max(lightRadSqr, lightSqr))) * saturate(NoL);
			
		# else // Analytical solution with horizon
			
			// Tilted patch to sphere equation
			float beta = acos(NoL);
			float h = lightDist / lightRad;
			float x = sqrt(Square(h) - 1.0) ;
			float y = -x * rcp(tan(beta));
			
			float illuminance = 0;
			if (h * cos(beta) > 1)
				illuminance = cos(beta) / Square(h);
			else
			{
				illuminance = rcp(PI * Square(h)) *
				(cos(beta) * acos(y) - x * sin(beta) * sqrt(1.0 - Square(y))) +
				INVPI * atan(sin(beta) * sqrt(1.0 - Square(y)) / x);
			}
			
			illuminance *= PI;
			
		# endif
	attenuation *= lerp(LIGHTINTENSITY_FAR, LIGHTINTENSITY_NEAR, illuminance);
	# endif

	float sphereSinAlpha		= sqrt(sphereSinAlphaSqr);
	
#if 1
	if(NoL < sphereSinAlpha)
	{
		NoL						= max(NoL, -sphereSinAlpha);
#if 0
		float sphereCosBeta 	= NoL;
		float sphereSinBeta 	= sqrt(1.0 - Square(sphereCosBeta));
		float sphereTanBeta 	= sphereSinBeta / sphereCosBeta;

		float x 				= sqrt(rcp(sphereSinAlphaSqr) - 1.0);
		float y 				= -x / sphereTanBeta;
		float z 				= sphereSinBeta * sqrt(1 - Square(y));

		NoL 					= NoL * acos(y) - x * z + atan(z / x) / sphereSinAlphaSqr;
		NoL 					*= INVPI;
#else
		NoL 					= Square(sphereSinAlpha + NoL) / (4.0 * sphereSinAlpha);
#endif
	}
#else
	NoL 						= saturate((NoL + sphereSinAlphaSqr) / (1.0 + sphereSinAlphaSqr));
#endif
	[branch]
	if(NoL > 0)
	{
	//	Properties.Roughness 		= Square(Properties.Roughness * 0.5 + 0.5);
		float roughness2 			= Square(Properties.Roughness);
		sphereSinAlpha 				= saturate((lightRad / lightDist) * (1.0 - roughness2));
		
		Dots Dot;	
		GetDots(Dot, normal, view, light);
		GetSphereNoH(Dot, sphereSinAlpha);

		Dot.NoV 					= saturate(abs(Dot.NoV) + 0.00001);
		Dot.NoL 					= saturate(Dot.NoL);
		
		float roughness4 			= Square(roughness2);

		float sphereRoughness4 		= roughness4;
		float specularEnergy 		= 1.0;
		if(sphereSinAlpha > 0.0)
		{
			sphereRoughness4 		= roughness4 + 0.25 * sphereSinAlpha * (3.0 * roughness2 + sphereSinAlpha) / (Dot.VoH + 0.001);
			specularEnergy 			= roughness4 / sphereRoughness4;
		}
		float Fc 					= Pow5(1.0 - Dot.VoH);		
		float3 fresnel				= saturate(50.0 * Properties.SpecularColor.g ) * Fc + (1.0 - Fc) * Properties.SpecularColor;
		float distribution			= roughness4 / (PI * Square((Dot.NoH * roughness4 - Dot.NoH) * Dot.NoH + 1.0));
		float geometricShadowing	= 0.5 / (Dot.NoL * (Dot.NoV * (1.0 - roughness2) + roughness2) + Dot.NoV * (Dot.NoL * (1.0 - roughness2) + roughness2));
		specular 					+= (fresnel * lightColorIntensity.rgb) * (attenuation * NoL * distribution * geometricShadowing * specularEnergy);

	#if 0
		//Lambert
		diffuse 					+= INVPI * (Properties.DiffuseColor * lightColorIntensity.rgb);
	#else
		//Burley
		float FD90 					= 0.5 + 2 * Square(Dot.VoH) * Properties.Roughness;
		float FdV 					= 1.0 + (FD90 - 1.0) * Pow5(1.0 - Dot.NoV);
		float FdL 					= 1.0 + (FD90 - 1.0) * Pow5(1.0 - Dot.NoL);
		diffuse						+= (Properties.DiffuseColor * lightColorIntensity.rgb) * (attenuation * NoL * INVPI * FdV * FdL);
	#endif
	}
}


float4 GetPixelColor( float2 iTexCoord, float3 iNormal, float3 iPos)
{
	float4 illuminationSample = tex2D(TexturetTeamSampler, iTexCoord);
	float4 colorSample = tex2D(TextureColorSampler, iTexCoord);
	illuminationSample = colorSample * Square(illuminationSample);
	float3 baseColor = ToLinear(colorSample.rgb);
    float4 dataSample = tex2D(TextureDataSampler, iTexCoord);
    
    //Team Color
    float4 teamColorScalar = (colorSample.a * g_TeamColor.a); 
	colorSample *= (1.f - teamColorScalar);
	colorSample += (g_TeamColor * teamColorScalar);

	//Self Illumination
	float selfIlluminationScalar = dataSample.b;

#ifdef PBR	
	//Bloom
	float bloomScalar = dataSample.b;
#else
	float bloomScalar = dataSample.g;
#endif
	colorSample = Square(colorSample) * bloomScalar;
	colorSample.rgb = (colorSample + illuminationSample);
	float4 oColor = colorSample * colorMultiplier;
//	oColor.rgb = 1-exp(-oColor.rgb);
	float3 normal		= normalize(iNormal);
	float3 view			= normalize(-iPos);
	float3 lightPos		= g_Light0_Position - iPos;
	float3 lightColor	= ToLinear(g_Light0_DiffuseLite.rgb);
	float NoL			= dot(normal, normalize(lightPos));
	
	float lightRad			= 25000;
	float attenuation		= 1.0;
	
	float3 specular = 0;
	float3 diffuse = 0;
	
	PBRProperties Properties;
	Properties.DiffuseColor		= baseColor * (1.0 - dataSample.r);
	Properties.SpecularColor	= lerp((float3)0.04, baseColor, dataSample.r);
	Properties.Roughness		= dataSample.a;
	
	SphereLight(Properties, normal, view, lightPos, lightRad, float4(lightColor.rgb, 1.0), specular, diffuse, attenuation, NoL);

	//NOTE we are deliberately NOT outputting the lighting in gamma space!
	return float4(max(oColor.rgb, (1.0 - exp(-(specular + diffuse))) * LIGHTINTENSITY_POST), oColor.a);
}

void
RenderScenePS( 
	VsOutput i,
	out float4 oColor0 : COLOR0 ) 
{ 
	oColor0 = GetPixelColor( i.TexCoord0, i.Normal, i.Pos);
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
