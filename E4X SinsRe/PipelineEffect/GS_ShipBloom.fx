#define PBR 
#define PI 		3.1415926535897932384626433832795
#define INVPI 	0.3183098861837906715377675267450
#define FRESNEL_CLAMP 0.0001 //variable that avoids caps a potenital near 0 division leading to infinitely bright fresnel effect
#define ONETHIRD 0.333333333333333333333333333333

#define LIGHTINTENSITY_FAR 0.8				// light intensity far from stars
#define LIGHTINTENSITY_NEAR	8.0				// light intensity close to stars
#define LIGHTINTENSITY_POST 1.0				// light intensity after remapping to 0-1 ranges
//nice if you play with bloom on!
#define AMBIENT_BOOST 10.0
#define EMISSIVE_BOOST 1.0					// emissive intensity
#define EMISSIVE_SELFILLUMINATION_BOOST 1.0	// emissive selfillumination intensity - will only do something if there is a lightmap
#define EMISSIVE_BOOST_POST 1.0				// emissive intensity after remapping to 0-1 ranges
#define IRIDESCENCE
#define SELFILLUMINATON
#define MULTISCATTER_IBL
#define LINEARCOLOR

float4x4 g_World : World;
float4x4 g_WorldViewProjection : WorldViewProjection;

texture	g_TextureDiffuse0 : Diffuse;
texture g_TextureTeamColor;
texture	g_TextureSelfIllumination;

texture	g_TextureEnvironmentCube : Environment;
texture g_EnvironmentIllumination : Environment;

float4 g_MaterialSpecular:Specular;
float4 g_MaterialAmbient:Ambient;
float4 g_MaterialEmissive:Emissive;

float4 g_TeamColor;


float colorMultiplier = 1000.f;
float3 g_Light0_Position: Position = float3( 0.f, 0.f, 0.f );
float4 g_Light0_DiffuseLite: Diffuse;// = float4( 1.f, 1.f, 1.f, 1.f );

sampler TextureColorSampler = sampler_state{
    Texture = <g_TextureDiffuse0>;
#ifndef Anisotropy
    Filter = LINEAR;
#else
	Filter = ANISOTROPIC;
	MaxAnisotropy = AnisotropyLevel;
#endif
};

sampler TextureDataSampler = sampler_state{
    Texture = <g_TextureSelfIllumination>;    
#ifndef Anisotropy
    Filter = LINEAR;
#else
	Filter = ANISOTROPIC;
	MaxAnisotropy = AnisotropyLevel;
#endif
};

sampler TexturetTeamSampler = sampler_state{
    Texture = <g_TextureTeamColor>;    
#ifndef Anisotropy
    Filter = LINEAR;
#else
	Filter = ANISOTROPIC;
	MaxAnisotropy = AnisotropyLevel;
#endif
};

samplerCUBE TextureEnvironmentCubeSampler = sampler_state{
    Texture = <g_TextureEnvironmentCube>;
    MinFilter = ANISOTROPIC;
    MagFilter = ANISOTROPIC;
    MipFilter = ANISOTROPIC;
    AddressU = CLAMP;
    AddressV = CLAMP;
    AddressW = CLAMP;
	 
};

samplerCUBE EnvironmentIlluminationCubeSampler = sampler_state{
    Texture = <g_EnvironmentIllumination>;
    MinFilter = ANISOTROPIC;
    MagFilter = ANISOTROPIC;
    MipFilter = ANISOTROPIC;
    AddressU = CLAMP;
    AddressV = CLAMP;
    AddressW = CLAMP;

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

float GetMipRoughness(float Roughness, float MipCount)
{
	//return MipCount - 1 - (3 - 1.15 * log2(Roughness));
	return sqrt(Roughness) * MipCount;
}
	
struct PBRProperties
{
	float3 SpecularColor;
	float3 DiffuseColor;
	float Roughness;
	float RoughnessMip;
	float Metal;
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

float Luminance(float3 X)
{
	return dot(float3(0.299, 0.587, 0.114), X);
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

float Pow3(float X)
{
	return Square(X) * X;
}
float2 Pow3(float2 X)
{
	return Square(X) * X;
}
float3 Pow3(float3 X)
{
	return Square(X) * X;
}
float4 Pow3(float4 X)
{
	return Square(X) * X;
}

float Pow4(float X)
{
	return Square(Square(X));
}
float2 Pow4(float2 X)
{
	return Square(Square(X));
}
float3 Pow4(float3 X)
{
	return Square(Square(X));
}
float4 Pow4(float4 X)
{
	return Square(Square(X));
}

float Pow5(float X)
{
	return Pow4(X) * X;
}	
float2 Pow5(float2 X)
{
	return Pow4(X) * X;
}	
float3 Pow5(float3 X)
{
	return Pow4(X) * X;
}	
float4 Pow5(float4 X)
{
	return Pow4(X) * X;
}	

float3 SRGBToLinear(float3 color)
{
	#ifdef LINEARCOLOR	
		//When external colors and the data texture are redone this can be reenabled.
		return pow(abs(color), 2.2);
	#else
		return color;
	#endif
	
}

float4 SRGBToLinear(float4 color)
{
	#ifdef LINEARCOLOR
		//When external colors and the data texture are redone this can be reenabled.
		return float4(SRGBToLinear(color.rgb), color.a);
	#else
		return color;
	#endif
}

float3 LinearToSRGB(float3 color)
{
	#ifdef LINEARCOLOR
		return pow(color, (float3)(1.0 / 2.2));
	#else
		return color;
	#endif
}

float4 LinearToSRGB(float4 color)
{
	#ifdef LINEARCOLOR
		return float4(LinearToSRGB(color.rgb), color.a);
	#else
		return color;
	#endif
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

// ratio: 1/3 = neon, 1/4 = refracted, 1/5+ = approximate white
float3 PhysHueToRGB(float Hue, float Ratio) {
	//return smoothstep((float3)0.0,(float3)1.0, abs(frac(Hue + float3(0.0,1.0,2.0) * Ratio) * 2.0 - 1.0));
	return smoothstep((float3)0.0,(float3)1.0, abs(frac(Hue + float3(0.0,1.0,2.0) * Ratio) * 2.0 - 1.0));
	//return cos((Hue + float3(0.0,1.0,2.0) * Ratio) * TWOPI) * 0.5 + 0.5;
}

	// based on http://home.hiroshima-u.ac.jp/kin/publications/TVC01/examples.pdf
	//and https://home.hiroshima-u.ac.jp/~kin/publications/EG00/rough_surface.pdf		
	void Iridescence(inout PBRProperties Properties, float NoV)
	{					
		float IridescenceThickness = mad(Luminance(Properties.SpecularColor), g_MaterialSpecular.g, g_MaterialSpecular.r) * PI;
		float Phase = 1.0 / 2.8;
		
		float3 IridescenceWeight = (Properties.SpecularColor * (1-Properties.SpecularColor));
		IridescenceWeight *= (2 * (1-saturate(lerp(g_MaterialSpecular.b, g_MaterialSpecular.a, Properties.Metal))));
		
		Properties.SpecularColor += PhysHueToRGB(-0.5 * NoV + IridescenceThickness * 2.4, IridescenceThickness * Phase) * IridescenceWeight;
	}

// Brian Karis(Epic's) optimized unified term derived from Call of Duty metallic/dielectric term improved with https://bruop.github.io/ibl/
void AmbientBRDF(inout float3 diffuse, inout float3 specular, float NoV, PBRProperties Properties, const float3 radiance = 1.0, const float3 irradiance = 1.0, const bool multiscatter = true)
{

	float4 r = Properties.Roughness * float4(-1.0, -0.0275, -0.572, 0.022) + float4(1.0, 0.0425, 1.04, -0.04);
	float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
	float2 AB = float2(-1.04, 1.04) * a004 + r.zw;

	AB.y *= (1.0 - 1.0 / (1.0 + max(0.0, 50.0))) * 3.0;
	
	if(multiscatter)
	{

	    // Roughness dependent fresnel, from Fdez-Aguera
//		float3 Fr = max((float3)(1.0 - Properties.Roughness), NoV) - NoV;
//		float3 k_S = Properties.SpecularColor * (NoV + Fr * Pow5(1.0 - NoV));
	
		float3 FssEss = Properties.SpecularColor * AB.x + AB.y;
	
		// Multiple scattering, from Fdez-Aguera
		float Ems = (1.0 - (AB.x + AB.y));
		float3 F_avg = NoV + (1.0 - NoV) * rcp(21.0);
		float3 FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
		float3 k_D = Properties.DiffuseColor * (1.0 - FssEss - FmsEms);
		
		diffuse 	+= (FmsEms + k_D) * irradiance;
		specular 	+= FssEss * radiance;
	}
	else
	{
		diffuse 	+= Properties.DiffuseColor * irradiance;
		specular 	+= (Properties.SpecularColor * AB.x + AB.y) * radiance;
	}
	
}


// Fresnel term
float3 F_GGX(PBRProperties Properties, float VoH)
{
	float Fc = Pow5(1.0 - VoH);

	return saturate(50.0 * Properties.SpecularColor.g) * Fc + (1.0 - Fc) * Properties.SpecularColor;
}

// Specular lobe
float D_GGX(float NoH, float Roughness)
{
	float a2 = Pow4(Roughness);	
	float d = Square((NoH * a2 - NoH) * NoH + 1.0);
	return a2 / (PI * d);	
//		return a2 / d;	
}
	
// Tweaked for Selfillumination
float V_GGX_Selfillumination(float Roughness, float NoV, float NoL)
{
	float k = Square(Roughness) * 0.5f;
	float G1V = NoV * (1.0 - k) + k;
	float G1L = NoL * (1.0 - k) + k;
	return 0.25f / (G1V * G1L);
}
	
float4 GetPixelColor( float2 iTexCoord, float3 iNormal, float3 iPos)
{
	float4 illuminationSample = tex2D(TexturetTeamSampler, iTexCoord);
	float4 colorSample = tex2D(TextureColorSampler, iTexCoord);
	illuminationSample = colorSample * Square(illuminationSample);
	
	float3 baseColor = SRGBToLinear(colorSample.rgb);
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
	float3 emissive = (SRGBToLinear(colorSample) * bloomScalar * EMISSIVE_BOOST + illuminationSample * EMISSIVE_SELFILLUMINATION_BOOST);

	float3 normal		= normalize(iNormal);
	float3 view			= normalize(-iPos);
	float3 lightPos		= g_Light0_Position - iPos;
	float3 lightColor	= SRGBToLinear(g_Light0_DiffuseLite.rgb);
	float NoL			= dot(normal, normalize(lightPos));
	
	float lightRad			= 25000;
	float attenuation		= 1.0;
	
	float3 specular = 0;
	float3 diffuse = 0;
	
	PBRProperties Properties;
	Properties.DiffuseColor		= baseColor * (1.0 - dataSample.r);
	Properties.SpecularColor	= lerp((float3)0.04, baseColor, dataSample.r);
	Properties.Roughness		= dataSample.a;
	Properties.RoughnessMip 	= GetMipRoughness(dataSample.a, 5.0);
	Properties.Metal			= dataSample.r;
	
	float NoV							= dot(normal, view);
	float3 reflection			= -(view - 2.0 * normal * NoV);
	NoV							= saturate(abs(NoV) * (1.0-FRESNEL_CLAMP) + FRESNEL_CLAMP);

	#ifdef IRIDESCENCE
		Iridescence(Properties, NoV);
	#endif
	
//	return float4(Properties.SpecularColor, 1);

	float3 ReflectionSample 	= SRGBToLinear(texCUBElod(TextureEnvironmentCubeSampler, float4(reflection, Properties.RoughnessMip))).rgb;// + max(0.04, dataSample.r) * lightmap;
//	#ifdef PURE_ROUGNESSBIASED_REFLECTION
//		return 					LinearToSRGB(float4(ReflectionSample, 0.0));
//	#endif
	float3 DiffuseSample		= SRGBToLinear(texCUBE(EnvironmentIlluminationCubeSampler, normal)).rgb;
//	#ifdef PUREREFLECTION
//		return 					float4(texCUBElod(TextureEnvironmentCubeSampler, float4(reflection, 0.0)).rgb, 0.0);
//	#endif
	#ifdef AMBIENT_BOOST
		ReflectionSample  		= 1-exp(-(ReflectionSample  + Square(ReflectionSample)	* AMBIENT_BOOST));
		DiffuseSample			= 1-exp(-(DiffuseSample		+ Square(DiffuseSample)		* AMBIENT_BOOST));
	#endif

	#ifdef MULTISCATTER_IBL
		AmbientBRDF(diffuse, specular, NoV, Properties, ReflectionSample, DiffuseSample, true);
	#else
		AmbientBRDF(diffuse, specular, NoV, Properties, ReflectionSample, DiffuseSample, false);
	#endif
	
	#ifdef SELFILLUMINATON 
		float3 SampleSelfIllumination = SRGBToLinear(tex2D(TexturetTeamSampler, iTexCoord)).rgb;
		float3 halfVec = normalize(reflection + view);
		float VoH = max(0.0, dot(view, halfVec));
		float3 F = F_GGX(Properties, VoH);
		float3 A = (1.0 - F) * Properties.DiffuseColor;
	
		float D = max(0.0, D_GGX(pow(saturate(abs(dot(normalize(reflection + view), view)) * 2.0 - 1.0), 0.125), Properties.Roughness));

		float3 V = saturate(V_GGX_Selfillumination(Properties.Roughness, NoV, dot(SampleSelfIllumination, SampleSelfIllumination)));
		V = max((float3)0.0, V) * D;
				
		diffuse += (SampleSelfIllumination * Properties.DiffuseColor * (1.0 - F));
		specular += SampleSelfIllumination * F * V;
	#endif
	
//	SphereLight(Properties, normal, view, lightPos, lightRad, float4(lightColor.rgb, 1.0), specular, diffuse, attenuation, NoL);

	//NOTE we are deliberately NOT outputting the lighting in gamma space!
	return float4((1.0 - exp(-(specular + diffuse))) * LIGHTINTENSITY_POST + (1.0 - exp(-emissive)) * EMISSIVE_BOOST_POST, 1);
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
        VertexShader = compile vs_3_0 RenderSceneVS();
        PixelShader = compile ps_3_0 RenderScenePS();
		AlphaTestEnable = FALSE;
        AlphaBlendEnable = TRUE;
		SrcBlend = ONE;
		DestBlend = ZERO;
		ZEnable = TRUE;
		ZWriteEnable = TRUE;			   
    }
}
