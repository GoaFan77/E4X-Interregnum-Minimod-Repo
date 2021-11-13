


//math constants
#define SF (1.0/float(0xffffffffU))
#define HALFPI 1.5707963267948966192313216916398
#define PI 		3.1415926535897932384626433832795
#define INVPI 	0.3183098861837906715377675267450
#define TWOPI 	6.2831853071795864769252867665590
#define INVTWOPI 	0.1591549430918953357688837633725

//options, you can comment them out to tweak performance if you have performance problems
#define SUPPORT_PARALLAX_DISPLACEMENT
#define SUPPORT_PARALLAX_SHADOWS        //only works if SUPPORT_PARALLAX_DISPLACEMENT is also on!
#define SUPPORT_SUBSURFACE
#define LIGHTINTENSITY_FAR 0.8			// light intensity far from stars
#define LIGHTINTENSITY_NEAR	8.0			// light intensity close to stars

//turns the planets in sphere probes to debug cubemaps
//#define DEBUG_MAKE_PLANET_SPHEREPROBE
#define DEBUG_SPHEREPROBE_METAL
#define DEBUG_SPHEREPROBE_MULTISCATTER true
#define DEBUG_SPHEREPROBE_WATER

float4x4		g_World					: World;
float4x4		g_WorldViewProjection	: WorldViewProjection;

texture	g_TextureDiffuse0 : Diffuse;
texture	g_TextureSelfIllumination;
texture g_TextureDiffuse2 : Diffuse;
texture g_TextureNormal : Normal;
texture g_TextureTeamColor: Displacement;
texture g_TextureNoise3D;
texture	g_TextureEnvironmentCube : Environment;
texture g_EnvironmentIllumination : Environment;


float4 g_Light_Emissive: Emissive;
float4 g_Light_AmbientLite: Ambient;
float4 g_Light_AmbientDark;

float3 g_Light0_Position: Position = float3( 0.f, 0.f, 0.f );
float4 g_Light0_DiffuseLite: Diffuse;// = float4( 1.f, 1.f, 1.f, 1.f );
float4 g_Light0_DiffuseDark;
float4 g_Light0_Specular;

float4 g_MaterialAmbient:Ambient;
float4 g_MaterialSpecular:Specular;
float4 g_MaterialEmissive:Emissive;
float4 g_MaterialDiffuse:Diffuse;
float g_MaterialGlossiness: Glossiness;
float4 g_GlowColor:Emissive;
float4 g_CloudColor:cloudColor;

float g_Time;
float g_Radius;

const float3 Ozone = float3(0, 0.475, 1);
#ifdef IsDebug
float DiffuseScalar = 1;
float AmbientScalar = 1;
float SelfIllumScalar = 1;
float SpecularScalar = 1;
float EnvironmentScalar = 1;
float TeamColorScalar = 1;
#endif

sampler TextureColorSampler = sampler_state
{
    Texture	= <g_TextureDiffuse0>;
    AddressU = WRAP;        
    AddressV = WRAP;
#ifndef Anisotropy
    Filter = LINEAR;
#else
	Filter = ANISOTROPIC;
	MaxAnisotropy = AnisotropyLevel;	
#endif
};

sampler TextureDataSampler = sampler_state
{
    Texture = <g_TextureSelfIllumination>;
    AddressU = WRAP;        
    AddressV = WRAP;
#ifndef Anisotropy
    Filter = LINEAR;
#else
	Filter = ANISOTROPIC;
	MaxAnisotropy = AnisotropyLevel;
#endif
};

sampler TextureNormalSampler = sampler_state
{
    Texture	= <g_TextureNormal>;
    AddressU = WRAP;        
    AddressV = WRAP;	
#ifndef Anisotropy
    Filter = LINEAR;
#else
	Filter = ANISOTROPIC;
	MaxAnisotropy = AnisotropyLevel;
#endif
};

sampler TextureDisplacementSampler = sampler_state
{
    Texture	= <g_TextureTeamColor>;
    AddressU = WRAP;        
    AddressV = WRAP;	
#ifndef Anisotropy
    Filter = LINEAR;
#else
	Filter = ANISOTROPIC;
	MaxAnisotropy = AnisotropyLevel;
#endif
};

sampler CloudLayerSampler = sampler_state
{
    Texture	= <g_TextureDiffuse2>;    
    AddressU = WRAP;        
    AddressV = WRAP;
#ifndef Anisotropy
    Filter = LINEAR;
#else
	Filter = ANISOTROPIC;
	MaxAnisotropy = AnisotropyLevel;
#endif		
};

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
void RenderSceneVSSimple(	float3 iPosition : POSITION, 
							float3 iNormal : NORMAL,
							float2 iTexCoord0 : TEXCOORD0,
							out float4 oPosition : POSITION,
							out float4 oColor : COLOR0,
							out float2 oTexCoord : TEXCOORD0)
{
	oPosition = mul(float4(iPosition, 1.f), g_WorldViewProjection);
	
	float3 lightPositionLocal = mul(g_Light0_Position, transpose((float3x3)g_World));
	float3 vToLightLocal = normalize(lightPositionLocal - iPosition);

    float4 diffuse = g_MaterialDiffuse * g_Light0_DiffuseLite * max(dot(vToLightLocal, iNormal), 0.f);
	float4 ambient = g_MaterialAmbient * g_Light_AmbientLite;
	oColor = diffuse + ambient;
		
	oTexCoord = iTexCoord0;
}

struct VsSceneOutput
{
	float4 position	: POSITION;
	float2 texCoord : TEXCOORD0;
	float3 normal : TEXCOORD1;
	float3 lightPos : TEXCOORD2;
//	float3 viewDir : TEXCOORD3;
	float3 pos : TEXCOORD3;
	float3 posObj : TEXCOORD4;
	float3 normalObj : TEXCOORD5;
	float3 planetPos : TEXCOORD6;
};
//TODO clean up unused interpolaters later, since this is probably not something the compiler will do!
VsSceneOutput RenderSceneVS(
	float3 position : POSITION, 
	float3 normal : NORMAL,
	float2 texCoord : TEXCOORD0)	
{
	VsSceneOutput output;  
	
	output.position = mul(float4(position/* * float3(1000.0, 0.001, 1000.0)*/, 1.0f), g_WorldViewProjection);
    output.texCoord = texCoord; 
	output.normal = mul(normal, (float3x3)g_World);
	//output.lightDir = normalize(g_Light0_Position.xyz - output.position.xyz);
    //output.lightPos = normalize(g_Light0_Position.xyz - output.position.xyz);
	float3 positionInWorldSpace = mul(float4(position/* * float3(1000.0, 0.001, 1000.0)*/, 1.f), g_World).xyz;

	output.lightPos = g_Light0_Position.xyz - positionInWorldSpace;
	
	output.planetPos = (positionInWorldSpace - mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz);
//    output.viewDir = normalize(-positionInWorldSpace);
    output.pos = positionInWorldSpace;
	output.posObj = position/* * float3(1000.0, 0.001, 1000.0)*/;
	output.normalObj = normal;
    return output;
}

inline float4 SRGBToLinear(float4 color)
{
	//return color;

	//When external colors and the data texture are redone this can be reenabled.
	return float4(color.rgb * (color.rgb * (color.rgb * 0.305306011f + 0.682171111f) + 0.012522878f), color.a);
}

inline float3 SRGBToLinear(float3 color)
{
	//return color;

	//When external colors and the data texture are redone this can be reenabled.
	return float3(color * (color * (color * 0.305306011f + 0.682171111f) + 0.012522878f));
}

inline float4 LinearToSRGB(float4 color)
{
	//return color;

	//When external colors and the data texture are redone this can be reenabled.
	float3 S1 = sqrt(color.rgb);
	float3 S2 = sqrt(S1);
	float3 S3 = sqrt(S2);
	return float4(0.662002687 * S1 + 0.684122060 * S2 - 0.323583601 * S3 - 0.225411470 * color.rgb, color.a);
}
float Luminance(float3 X)
{
	return dot(float3(0.3, 0.59, 0.11), X);
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

float3 Hash(float3 p, float2x2 rot)
{
	p = float3( dot(p,float3(127.1, 311.7, 74.7)),
				dot(p,float3(269.5, 183.3, 246.1)),
				dot(p,float3(113.5, 271.9, 124.6)));
	p = -1.0 + 2.0 * frac(sin(p) * 43758.5453123);
	p.xz = mul(p.xz, rot);
	return p;
}

float Snoise(float3 p, float speed)
{
    float3 i = floor(p);
    float3 f = frac(p);
	
	float3 u = f * f * (3.0 - 2.0 * f);
	float2x2 rot = float2x2(float2(cos(speed), -sin(speed)), float2(sin(speed), cos(speed)));

    return lerp(lerp(lerp(dot(Hash(i + float3(0.0, 0.0, 0.0), rot), f - float3(0.0, 0.0, 0.0)), 
                          dot(Hash(i + float3(1.0, 0.0, 0.0), rot), f - float3(1.0, 0.0, 0.0)), u.x),
                     lerp(dot(Hash(i + float3(0.0, 1.0, 0.0), rot), f - float3(0.0, 1.0, 0.0)), 
                          dot(Hash(i + float3(1.0, 1.0, 0.0), rot), f - float3(1.0, 1.0, 0.0)), u.x), u.y),
                lerp(lerp(dot(Hash(i + float3(0.0, 0.0, 1.0), rot), f - float3(0.0, 0.0, 1.0)), 
                          dot(Hash(i + float3(1.0, 0.0, 1.0), rot), f - float3(1.0, 0.0, 1.0)), u.x),
                     lerp(dot(Hash(i + float3(0.0, 1.0, 1.0), rot), f - float3(0.0, 1.0, 1.0)), 
                          dot(Hash(i + float3(1.0, 1.0, 1.0), rot), f - float3(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}


// return value noise (in x) and its derivatives (in yzw)
float4 Nnoise(float3 p, float speed)
{
    // grid
    float3 i = floor(p);
    float3 w = frac(p);
    
    #if 1
    // quintic interpolant
    float3 u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
    float3 du = 30.0 * w * w * (w * (w - 2.0) + 1.0);
    #else
    // cubic interpolant
    float3 u = w * w * (3.0 - 2.0 * w);
    float3 du = 6.0 * w * (1.0 - w);
    #endif    
    
	float2x2 rot = float2x2(float2(cos(speed), -sin(speed)), float2(sin(speed), cos(speed)));
    // gradients
    float3 ga = Hash(i + float3(0.0, 0.0, 0.0), rot);
    float3 gb = Hash(i + float3(1.0, 0.0, 0.0), rot);
    float3 gc = Hash(i + float3(0.0, 1.0, 0.0), rot);
    float3 gd = Hash(i + float3(1.0, 1.0, 0.0), rot);
    float3 ge = Hash(i + float3(0.0, 0.0, 1.0), rot);
	float3 gf = Hash(i + float3(1.0, 0.0, 1.0), rot);
    float3 gg = Hash(i + float3(0.0, 1.0, 1.0), rot);
    float3 gh = Hash(i + float3(1.0, 1.0, 1.0), rot);
    
    // projections
    float va = dot(ga, w - float3(0.0,0.0,0.0));
    float vb = dot(gb, w - float3(1.0,0.0,0.0));
    float vc = dot(gc, w - float3(0.0,1.0,0.0));
    float vd = dot(gd, w - float3(1.0,1.0,0.0));
    float ve = dot(ge, w - float3(0.0,0.0,1.0));
    float vf = dot(gf, w - float3(1.0,0.0,1.0));
    float vg = dot(gg, w - float3(0.0,1.0,1.0));
    float vh = dot(gh, w - float3(1.0,1.0,1.0));
	
    // interpolations
    return float4(	ga + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga) + u.x * u.y * (ga - gb - gc + gd) + u.y * u.z * (ga - gc - ge + gg) + u.z * u.x * (ga - gb - ge + gf) + (-ga + gb + gc - gd + ge - gf - gg + gh) * u.x * u.y * u.z +   
					du * (float3(vb, vc, ve) - va + u.yzx * float3(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf) + u.zxy * float3(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg) + u.yzx * u.zxy * (-va + vb + vc - vd + ve - vf - vg + vh)),// derivatives
					va + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va) + u.x * u.y * (va - vb - vc + vd) + u.y * u.z * (va - vc - ve + vg) + u.z * u.x * (va - vb - ve + vf) + (-va + vb + vc - vd + ve - vf - vg + vh) * u.x * u.y * u.z    );// value
}
		
struct PBRProperties
{
	float3 SpecularColor;
	float3 DiffuseColor;
	float4 EmissiveColor;
	float Roughness;
	float AO;
	float3 SubsurfaceColor;
	float SubsurfaceOpacity;
};

// Frostbite presentation (moving frostbite to pbr)
inline float3 GetSpecularDominantDir(float3 vN, float3 vR, PBRProperties Properties)
{
	float InvRoughness = 1.0 - Properties.Roughness;
	float lerpFactor = saturate(InvRoughness * (sqrt(InvRoughness) + Properties.Roughness));

	return lerp(vN, vR, lerpFactor);
}

// Brian Karis(Epic's) optimized unified term derived from Call of Duty metallic/dielectric term
inline float3 AmbientBRDF(float NoV, PBRProperties Properties)
{
	float4 r = Properties.Roughness * float4(-1.0, -0.0275, -0.572, 0.022) + float4(1.0, 0.0425, 1.04, -0.04);
	float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
	float2 AB = float2(-1.04, 1.04) * a004 + r.zw;

	AB.y *= (1.0 - 1.0 / (1.0 + max(0.0, 50.0 * Properties.SpecularColor.g))) * 3.0;

	return Properties.SpecularColor * AB.x + AB.y;
}

inline float AmbientDielectricBRDF(float Roughness, float NoV)
{
	// Same as EnvBRDFApprox( 0.04, Roughness, NoV )
	const float2 c0 = float2(-1	, -0.0275);
	const float2 c1 = float2(1	, 0.0425);
	float2 r = Roughness * c0 + c1;
	return min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
}

inline float3 AmbientBRDF(float NoV, float Roughness, float3 SpecularColor)
{
	float4 r = Roughness * float4(-1.0, -0.0275, -0.572, 0.022) + float4(1.0, 0.0425, 1.04, -0.04);
	float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
	float2 AB = float2(-1.04, 1.04) * a004 + r.zw;

	AB.y *= (1.0 - 1.0 / (1.0 + max(0.0, 50.0 * SpecularColor.g))) * 3.0;

	return SpecularColor * AB.x + AB.y;
}
float GetMipRoughness(float Roughness, float MipCount)
{
	//return MipCount - 1 - (3 - 1.15 * log2(Roughness));
	return sqrt(Roughness) * MipCount;
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

void AmbientBRDF(inout float3 diffuse, inout float3 specular, float NoV, float roughness, float3 specularColor, float3 diffuseColor, const float3 radiance = 1.0, const float3 irradiance = 1.0, const bool multiScatter = true)
{

	float4 r = roughness * float4(-1.0, -0.0275, -0.572, 0.022) + float4(1.0, 0.0425, 1.04, -0.04);
	float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
	float2 AB = float2(-1.04, 1.04) * a004 + r.zw;

	AB.y *= (1.0 - 1.0 / (1.0 + max(0.0, 50.0))) * 3.0;
	
	if(multiScatter)
	{

		// Roughness dependent fresnel, from Fdez-Aguera
//		float3 Fr = max((float3)(1.0 - Properties.Roughness), NoV) - NoV;
//		float3 k_S = Properties.SpecularColor * (NoV + Fr * Pow5(1.0 - NoV));
	
		float3 FssEss = specularColor * AB.x + AB.y;
	
		// Multiple scattering, from Fdez-Aguera
		float Ems = (1.0 - (AB.x + AB.y));
		float3 Favg = NoV + (1.0 - NoV) * rcp(21.0);
		float3 FmsEms = Ems * FssEss * Favg / (1.0 - Favg * Ems);
		float3 kD = diffuseColor * (1.0 - FssEss - FmsEms);
		
		diffuse 	+= (FmsEms + kD) * irradiance;
		specular 	+= FssEss * radiance;
	}
	else
	{
		diffuse 	+= diffuseColor * irradiance;
		specular 	+= (specularColor * AB.x + AB.y) * radiance;
	}
	
}

// Frostbite presentation (moving frostbite to pbr)
inline float3 GetDiffuseDominantDir(float3 normal, float3 view, float NoV, PBRProperties Properties)
{
	float a = 1.02341 * Properties.Roughness - 1.51174;
	float b = -0.511705 * Properties.Roughness + 0.755868;
	// The result is not normalized as we fetch in a cubemap
	return lerp(normal , view , saturate((NoV * a + b) * Properties.Roughness));
}

struct PBRDots
{
	float NoV;
	float NoL;
	float VoL;
	float NoH;
	float VoH;
};

PBRDots GetDots(float3 normal, float3 view, float3 light)
{	
	PBRDots Ouput;
	Ouput.NoL = dot(normal, light);
	Ouput.NoV = dot(normal, view);
	Ouput.VoL = dot(view, light);
	float invLenH = rcp(sqrt( 2.0 + 2.0 * Ouput.VoL));
	Ouput.NoH = saturate((Ouput.NoL + Ouput.NoV) * invLenH);
	Ouput.VoH = saturate(invLenH + invLenH * Ouput.VoL);
	Ouput.NoL = saturate(Ouput.NoL);
	Ouput.NoV = saturate(abs(Ouput.NoV * 0.9999 + 0.0001));
	return Ouput;
}

// Diffuse term
inline float3 DiffuseBurley(PBRProperties Properties, PBRDots Dots)
{
	float FD90 = 0.5 + 2.0 * Square(Dots.VoH) * Properties.Roughness;
	float FdV = 1.0 + (FD90 - 1.0) * Pow5( 1.0 - Dots.NoV );
	float FdL = 1.0 + (FD90 - 1.0) * Pow5( 1.0 - Dots.NoL );
	return Properties.DiffuseColor * INVPI * FdV * FdL; // 0.31831 = 1/pi
}

// Specular lobe
inline float Dggx(PBRProperties Properties, PBRDots Dots)
{
	float a2 = Pow4(Properties.Roughness);	
	float d = Square((Dots.NoH * a2 - Dots.NoH) * Dots.NoH + 1.0);
	return a2 / (PI * d);				
}

// Geometric attenuation
inline float Vggx(PBRProperties Properties, PBRDots Dots)
{
	float a = Square(Properties.Roughness);
	float visSmithV = Dots.NoL * (Dots.NoV * (1.0 - a) + a);
	float visSmithL = Dots.NoV * (Dots.NoL * (1.0 - a) + a);
	return 0.5 * rcp(visSmithV + visSmithL);
}

// Fresnel term
inline float3 Fggx(PBRProperties Properties, PBRDots Dots)
{
	float Fc = Pow5(1.0 - Dots.VoH);
	return saturate(50.0 * Properties.SpecularColor.g) * Fc + (1.0 - Fc) * Properties.SpecularColor;
}
	
// Note the specular peak will be high dynamic range on low roughness surfaces, especially metals. We tonemap in the end, to avoid whiteouts
inline float3 SpecularGGX(PBRProperties Properties, PBRDots Dots)
{
	return Fggx(Properties, Dots) * (Dggx(Properties, Dots) * Vggx(Properties, Dots));
}

// recreate blue normal map channel
inline float DeriveZ(float2 normalXY)
{	

	float normalZ = sqrt(abs(1.0 - Square(Square(normalXY.x) - Square(normalXY.y))));
	
	return normalZ;
}
		
inline float2 ToNormalSpace(float2 normalXY)
{	
	return normalXY * 2.0 - 1.0;
}	
	
inline float3 GetNormalDXT5(float4 normalSample)
{
	float2 normalXY = normalSample.wy;
	normalXY = ToNormalSpace(normalXY);
	// play safe and normalize
	return normalize(float3(normalXY, DeriveZ(normalXY)));
}

inline float MapClouds(float c)
{
	return (1.0 - exp(-Square(c) * g_MaterialAmbient.a * 5.0));
}

inline float RayleighPhaseFunction(float VoL)
{
	return
			3. * (1. + VoL * VoL)
	/ //------------------------
				(16. * PI);
}

inline float HenyeyGreensteinPhaseFunction(float VoL, float g)
{
	return
						(1. - g * g)
	/ //---------------------------------------------
		((4. + PI) * pow(abs(1. + g * g - 2. * g * VoL) + 0.0001, 1.5));
}

// Schlick Phase Function factor
// Pharr and  Humphreys [2004] equivalence to g above
inline float SchlickPhaseFunction(float VoL, float g)
{
const float k = 1.55 * g - 0.55 * (g * g * g);				
	return
					(1. - k * k)
	/ //-------------------------------------------
		(4. * PI * (1. + k * VoL) * (1. + k * VoL));
}
	
// Hash without Sine
// MIT License...
// Copyright (c)2014 David Hoskins.
// https://www.shadertoy.com/view/4djSRW
inline float Hash13(float3 p3)
{
	p3  = frac(p3 * .1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return frac((p3.x + p3.y) * p3.z);
}

inline float Noise13(	float3 pos,
						const int detail = 1, 
						const float dimension = 0.0, 
						const float lacunarity = 2.0,
						//const float3 Phase	= 0.0,
						const bool smooth = true)
{
	float sum = 0;
		
	for(int j = 0; j < detail; j++)
	{
		//P += Phase;
		float3 i = floor(pos);
		float3 f = frac(pos);
		if(smooth)
			f *= f * (3.0 - 2.0 * f);

		sum += mad(lerp(lerp(lerp(	Hash13(i + float3(0, 0, 0)),
									Hash13(i + float3(1, 0, 0)), f.x),
							lerp(	Hash13(i + float3(0, 1, 0)),
									Hash13(i + float3(1, 1, 0)), f.x), f.y),
						lerp(lerp(	Hash13(i + float3(0, 0, 1)),
									Hash13(i + float3(1, 0, 1)), f.x),
							lerp(	Hash13(i + float3(0, 1, 1)),
									Hash13(i + float3(1, 1, 1)), f.x), f.y), f.z), 2, -1) * (1 - dimension);
		pos = mul(pos * lacunarity, float3x3(0.00,  0.80,  0.60, 
											-0.80,  0.36, -0.48, 
											-0.60, -0.48,  0.64));
	}
	return sum * rcp(detail);	
}
inline float2 Rotate(float2 pos, float rot)
{
	float s, c;
	sincos(rot, s, c);
	return mul(float2x2(c, -s, s, c), pos);
}

inline float2 VolcanoSmoke(float2 uv, float2 duvx, float2 duvy, float smokeDensity, float smokePuffyness)	
{
	float2 smokeSample = tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).gb;
	float smokeSpeed = (g_Time * 0.25 + (smokeSample.r * 4 + smokeSample.g));
	float smokeStep = floor(smokeSpeed) * 0.125;
	float smokeBlend = frac(smokeSpeed);
	float smokeAnimation = lerp(tex2Dgrad(TextureDisplacementSampler, float2(uv.x + frac(smokeStep - g_Time * 0.001), uv.y), duvx, duvy).b, 
								tex2Dgrad(TextureDisplacementSampler, float2(uv.x + frac(smokeStep + 0.125 - g_Time * 0.001), uv.y), duvx, duvy).b, smokeBlend)
								* smokePuffyness;
	return float2(max(0, Square(smokeSample.r) * smokeDensity + smokeSample.r * smokeAnimation), smokeAnimation);
}

inline float3 GetFlow(float timer)
{
	return float3(frac(float2(timer, timer + 0.5)), abs(frac(timer) * 2 - 1));
}

inline float3 GetLavaFlow(float2 uv, float2 duvx, float2 duvy)
{
	float2 flowSampleRaw = tex2Dlod(TextureNormalSampler, float4(uv, 0, 2)).yw * 2.0 - 1.0;
	float lavaMask = tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).r;

	if(dot(flowSampleRaw.yx, flowSampleRaw.yx) > 0.0001)//could lead to a NaN
	{
		flowSampleRaw = normalize(flowSampleRaw.yx) * rcp(float2(1024, 512)) * lavaMask;
	}
	else
	{
		flowSampleRaw = 0;	
	}
	
	float3 flowBase = GetFlow(g_Time * 0.25);

	return lerp(tex2Dgrad(TextureDataSampler, uv - flowSampleRaw * flowBase.x, duvx, duvy).rgb,
				tex2Dgrad(TextureDataSampler, uv - flowSampleRaw * flowBase.y, duvx, duvy).rgb, flowBase.z);
}

inline float4 MapGasGiant(float4 map)
{
	return lerp(float4(map.r, map.g, 1.0, 1.0 - map.a), map * map * (3.0 - 2.0 * map), g_MaterialSpecular.r);
}
//compiler failed to choose the code moved to where it was called instead
inline float4 MapGasGiant(float map)
{
	return lerp(0.5, map, g_MaterialSpecular.r);
}

inline float CloudSim(float2 uv, float2 duvx, float2 duvy, float speed)
{
	float4 baseSample = tex2Dgrad(CloudLayerSampler, uv, duvx, duvy);
	float2 flowDir = (baseSample.yw - 0.5) * float2(1.0, -1.0) * 0.125;
	
	float swapA = frac(speed);
	float swapB = frac(speed + 0.5);
	
	float2 offsetA = (flowDir * swapA);
	float cloudDetailA = (tex2Dgrad(CloudLayerSampler, (uv - offsetA * 1.5) * 3, duvx * 2, duvy * 2).r + tex2Dgrad(CloudLayerSampler, (uv - offsetA * 2.0) * 4 + 0.5, duvx * 3, duvy * 3).r);
	float cloudA = tex2Dgrad(CloudLayerSampler, uv - offsetA, duvx, duvy).b;
	cloudA += cloudDetailA * 0.25;
	
	float2 offsetB = (flowDir * swapB);
	float cloudDetailB = (tex2Dgrad(CloudLayerSampler, (uv - offsetB * 1.5) * 3, duvx * 2, duvy * 2).r + tex2Dgrad(CloudLayerSampler, (uv - offsetB * 2.0) * 4 + 0.5, duvx * 3, duvy * 3).r);
	float cloudB = tex2Dgrad(CloudLayerSampler, uv - offsetB, duvx, duvy).b;
	cloudB += cloudDetailB * 0.25;
	
	return 1.0 - rcp(1.0 + Square(lerp(cloudA, cloudB, abs(swapA * 2.0 - 1.0))));
}

inline float3 ToTangentSpace(float3x3 trans, float3 vec)
{
    return mul(trans, vec);
}

inline float3 ToWorldSpace(float3x3 trans, float3 vec)
{
    return mul(vec, trans);
}

inline float2 ParallaxVector(float3 viewTS, float parallaxScale)
{
#if 1 
	float  parallaxLength    	= sqrt(dot(viewTS.xy, viewTS.xy) / (dot(viewTS.z, viewTS.z)));
    return -normalize(viewTS.xy) * (parallaxScale * parallaxLength);
#else
	return (-viewTS.xy / viewTS.z) * parallaxScale;
#endif	
}

//smooth version of step
inline float AAStep(float compValue, float gradient)
{
  float halfChange = fwidth(gradient) * 0.5;
  //base the range of the inverse lerp on the change over one pixel
  float lowerEdge = compValue - halfChange;
  float upperEdge = compValue + halfChange;
  //do the inverse interpolation
  float stepped = (gradient - lowerEdge) / (upperEdge - lowerEdge);
  return saturate(stepped);
}


#if 1

struct Properties
{
	float Roughness;
	float3 SpecularColor;
	float3 DiffuseColor;
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

float3 SubsurfaceLight(float3 subsurfaceColor, float subsurfaceOpacity, float3 light, float3 normal, float3 view, float ao, float falloff)
{
	float3 halfVec = normalize(view + light);

	float inScatter = pow(saturate(dot(light, -view)), 12) * lerp(3, .1f, subsurfaceOpacity);
	float normalContribution = saturate(dot(normal, halfVec) * subsurfaceOpacity + 1.0 - subsurfaceOpacity);
	float backScatter = ao * normalContribution * INVTWOPI;

	return (falloff * lerp(backScatter, 1, inScatter)) * subsurfaceColor;
}

void SphereLight(PBRProperties Properties, float3 normal, float3 view, float3 lightPos, float lightRad, float4 lightColorIntensity, float3 atmosphereAbsorption, inout float3 specular, inout float3 diffuse, inout float attenuation, inout float NoL)
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
		specular 					+= (fresnel * lightColorIntensity.rgb * atmosphereAbsorption) * (attenuation * NoL * distribution * geometricShadowing * specularEnergy);

		float3 transmission = SubsurfaceLight(Properties.SubsurfaceColor, 1-Properties.SubsurfaceOpacity, light, normal, view, Properties.AO, attenuation);

	#if 1
		//Lambert
		diffuse 					+= lerp(((attenuation * NoL * INVPI) * Properties.DiffuseColor), transmission, Properties.SubsurfaceOpacity) * lightColorIntensity.rgb * atmosphereAbsorption;
	#else
		//Burley
		float FD90 					= 0.5 + 2 * Square(Dot.VoH) * Properties.Roughness;
		float FdV 					= 1.0 + (FD90 - 1.0) * Pow5(1.0 - Dot.NoV);
		float FdL 					= 1.0 + (FD90 - 1.0) * Pow5(1.0 - Dot.NoL);
		diffuse						+= lerp((Properties.DiffuseColor * (attenuation * NoL * INVPI * FdV * FdL)), transmission, Properties.SubsurfaceOpacity) * lightColorIntensity.rgb * atmosphereAbsorption;
	#endif
	}
}

#endif

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

#if 0
// return RaySphere(sphereCenter, sphereRadius, rayOrigin, rayDir)
float3 RaySphere(float3 sphereCenter, float sphereRadius, float3 rayOrigin, float3 rayDir, const float depthOffset = 0.0)
{
	float3 offset = rayOrigin - sphereCenter;
	float a = 1;
	float b = 2 * dot(offset, rayDir);
	float c = dot(offset, offset) - sphereRadius * sphereRadius;
	float d = b*b - 4 * a * c;
	float s = 0;
	if(d > 0)
	{
		s = sqrt(d);
		float distToSphereNear = max(0, (-b - s) / (2 * a));
		float distToSphereFar = (-b + s) / (2 * a);
		
		if(distToSphereFar >= 0)
		{
			return float3(distToSphereNear, distToSphereFar - distToSphereNear, s);
		}
	}
	return float3(MF, 0.0, 0.0);
}
#endif

//return SampleCityLightsBiasLoop(uv, uvp, duvx, duvy, bias);
float3 SampleCityLightsBiasLoop(float2 uv, float2 uvp, float2 duvx, float2 duvy, float Bias)
{
	
	
	float3 Sum = tex2Dgrad(TextureDataSampler, uvp, duvx, duvy).rgb;
	float steps = 1;
	Bias = max(1, Bias);
	//mip bias sampling limit is technically max is 15.999, but the lowest mips are too wide, just makes the planet look like smog
	for(float b = 1; b < 16; b += Bias)
	{
		steps +=1;
		Sum += tex2Dgrad(TextureDataSampler, lerp(uvp, uv, Square(saturate(b * 0.0625))), duvx * Square(b), duvy * Square(b)).rgb;
	}
	Sum *= rcp(steps + 1);
	return Sum * Sum * (3.0 - 2.0 * Sum);
}

float4 RenderScenePS(VsSceneOutput input) : COLOR0
{		
	const bool oceanMode 					= g_MaterialSpecular.a 					== 1.0;//planets that got oceans, support citylights too
	const bool iceMode 						= round(g_MaterialSpecular.a * 4.0) 	== 0.0;//planets that are lave based, no citylight
	const bool volcanoMode 					= round(g_MaterialSpecular.a * 4.0) 	== 1.0;//planets that are lave based, no citylight
	const bool gasGiantMode 				= round(g_MaterialSpecular.a * 4.0) 	== 2.0;//planets that are lave based, no citylight
	const bool fogMode 						= round(g_MaterialSpecular.a * 4.0) 	== 3.0;//planets that are lave based, no citylight
	
//	if(iceMode)
//		return float4(0.0, 1.0, 1.0, 1.0);
	float3 Reyleigh 						= g_MaterialDiffuse.rgb;
//	g_MaterialDiffuse.rgb //Reyleigh

	float3 normal 							= normalize(input.normal);
	float3 normalSphere 					= normal;
	
    
//	float3 positionInWorldSpace				= mul(float4(position, 1.f), g_World).xyz;
	
//	input.planetPos	

	
	float3 pos 								= input.pos;
	float3 lightPos							= g_Light0_Position - pos;
	float3 view 							= normalize(-pos);
	
	float lightSqr							= dot(input.lightPos, input.lightPos);
	float lightDist							= sqrt(lightSqr);
	float lightInverseDistance				= rcp(1+lightDist);
	float3 light 							= input.lightPos / lightDist;	

//	return float4(frac(lightDist.r / 1000), saturate(dot(normal, light)), 0, 1);
				
//	float2 uv								= input.texCoord;
//	float2 duvx								= ddx(uv);
//	float2 duvy								= ddy(uv);

	float2 duvx;
	float2 duvy;
	float2 uv								= GetEquirectangularUV(normalize(input.normalObj), duvx, duvy);
	

	float2 uvFlat							= uv;
//	return float4(duvx, 0.0, 1.0);
/*	
	//This only works because the sphere unwrap matches the color coordinates of the normal
	float3 tangent 							= normalize(cross(normal, float3(0.0, 1.0, 0.0)));
	float3 cotangent 						= normalize(cross(normal, tangent));
*/	

	float3 dpx				= ddx(pos);
	float3 dpy				= ddy(pos);
//	return float4(normalize(cross(dpx, dpy)), 1);
	float3 dpxp				= cross(normalSphere, dpx);
	float3 dpyp				= cross(dpy, normalSphere);
	
	float3 tangent 			= dpyp * duvx.x + dpxp * duvy.x;
	float3 cotangent 		= dpyp * duvx.y + dpxp * duvy.y;
	float tcNormalize		= pow(max(dot(tangent, tangent), dot(cotangent, cotangent)), -0.5);
	tangent					*= tcNormalize;
	cotangent 				*= tcNormalize;

	
	float  parallaxScale					= 0.025;
				
	float2 parallaxTS						= ParallaxVector(ToTangentSpace(float3x3(tangent, cotangent, normal), view), parallaxScale);	
			
	float3 lightTS							= ToTangentSpace(float3x3(tangent, cotangent, normal), light);
	

// * lightDist / g_Radius)
//	return float4(lightTS.zzz, 1);
	lightTS									= normalize(float3(lightTS.xy, (1.0 + lightTS.z) * rcp(parallaxScale)));//bending the vector avoids the singularity around 0
	
	float3 lightColor 						= SRGBToLinear(g_Light0_DiffuseLite.rgb) * PI;
	lightColor 								= PI;
	
	float NoV								= abs(dot(view, normal));
	float NoL 								= dot(normal, light);
	
	
	///////////////////////////////////////////////////////////////////////////////////////////////// GAS GIANT START /////////////////////////////////////////////////////////////////////////////////////////////////
	[branch]
	if(gasGiantMode)
	{		
		float poleDensity					= Pow5(abs(uv.y * 2.0 - 1.0));
		float mipBias						= poleDensity * 9.0 + 1.0;
		//force the mips down near the poles!
		duvx								*= mipBias;
		duvy								*= mipBias;
		
		float jacobiFalloff					= saturate(Square(normal.y)) * 0.499 + 0.5;
			
		float speed							= g_Time * g_MaterialSpecular.g;
		int iterations						= 4;
		float2 uvFlowA						= uv;
		float2 uvFlowB						= uv;
		
		const float2 density				= rcp(float2(32.0 * (1.0 - abs(uv.y - 0.5)) + 1.0, 32.0));
			
		float forceFalloff					= 1.0;
			
		float swapA							= frac(speed);
		float swapB							= frac(speed + 0.5);
		
		[unroll]
		for(int i = 0; i < iterations; ++i)
		{
			uvFlowA 						-= (tex2Dgrad(TextureDisplacementSampler, uvFlowA, duvx, duvy).xy - 0.5) * (density * forceFalloff * swapA);
			uvFlowB 						-= (tex2Dgrad(TextureDisplacementSampler, uvFlowB, duvx, duvy).xy - 0.5) * (density * forceFalloff * swapB);
			forceFalloff 					*= jacobiFalloff;
		}

		float gasBlend						= abs(swapA * 2.0 - 1.0);
		
		float4 gasSample					= MapGasGiant(lerp(tex2Dgrad(TextureColorSampler, uvFlowA, duvx, duvy), tex2Dgrad(TextureColorSampler, uvFlowB, duvx, duvy), gasBlend));
		
		uvFlowA								+= parallaxTS * (gasSample.g - 0.5);
		uvFlowB								+= parallaxTS * (gasSample.g - 0.5);
		
		gasSample 							= MapGasGiant(lerp(tex2Dgrad(TextureColorSampler, uvFlowA, duvx, duvy), tex2Dgrad(TextureColorSampler, uvFlowB, duvx, duvy), gasBlend));

		float ao							= Square(gasSample.b) * 0.95 + 0.05;
		
		float2 uvColor						= float2(frac(speed * 0.01), saturate((uv.y - 0.5) + (gasSample.a - 0.5) * 0.2 + 0.5));
		
		//abs shouldn't be needed, but double checking all warnings
		float3 gasColor						= SRGBToLinear(pow(abs(tex2Dlod(TextureNormalSampler, float4(uvColor, 0.0, 0.0)).rgb), (float3)(1.5 - gasSample.r)));

		const float offset					= rcp(512.0);
		float4 blendedNormals;
		blendedNormals						= float4(	tex2Dgrad(TextureColorSampler, uvFlowA - float2(offset, 0.0), duvx, duvy).g,
														tex2Dgrad(TextureColorSampler, uvFlowA + float2(offset, 0.0), duvx, duvy).g,
														tex2Dgrad(TextureColorSampler, uvFlowA + float2(0.0, offset), duvx, duvy).g,
														tex2Dgrad(TextureColorSampler, uvFlowA - float2(0.0, offset), duvx, duvy).g) * (1.0 - gasBlend);
					
		blendedNormals						+= float4(	tex2Dgrad(TextureColorSampler, uvFlowB + float2(offset, 0.0), duvx, duvy).g,
														tex2Dgrad(TextureColorSampler, uvFlowB - float2(offset, 0.0), duvx, duvy).g,
														tex2Dgrad(TextureColorSampler, uvFlowB + float2(0.0, offset), duvx, duvy).g,
														tex2Dgrad(TextureColorSampler, uvFlowB - float2(0.0, offset), duvx, duvy).g) * gasBlend;
											
		blendedNormals						= lerp((float4)0.5, blendedNormals, g_MaterialSpecular.r);							
		//lerp(0.5, map, g_MaterialSpecular.r);
		
		
		blendedNormals.xy					= blendedNormals.xz - blendedNormals.yw;
		blendedNormals.z					= DeriveZ(blendedNormals.xy);
					
		normal								= normalize(ToWorldSpace(float3x3(tangent, cotangent, normal), blendedNormals.xyz));
					
		float3 diffuse						= gasColor * SRGBToLinear(texCUBElod(EnvironmentIlluminationCubeSampler, float4(normal, 0.0))).rgb * ao;
		float3 lightSum 					= diffuse;
					
		float cloudAbsorption				= gasSample.g;
		float steps							= 32 - 31 * saturate(NoL);
		float shadowBlend					= (1.0 - saturate(NoL)) * steps;
		float gasDensity					= 2.0;
		
		[branch]
		if(NoL > -0.1)
		{
			float stepsInv					= 1.0 / steps;	
			float2 shadowStep				= lightTS.xy * stepsInv;		
			
			float NoLinv					= rcp(1.0 + NoL * steps);
			
			//steps = (steps - (steps - 1) * NoL);
			
			[loop]
			for(int g = 0; g < shadowBlend; g++)
			{
				uvFlowA						+= shadowStep;
				uvFlowB						+= shadowStep;
				cloudAbsorption				-= lerp(0.5, lerp(tex2Dgrad(TextureColorSampler, uvFlowA, duvx, duvy).g, tex2Dgrad(TextureColorSampler, uvFlowB, duvx, duvy).g, gasBlend), g_MaterialSpecular.r) * saturate(shadowBlend - g) * gasDensity;
			}			
			cloudAbsorption					= exp(min(0.0, cloudAbsorption * stepsInv));
		}			
		else			
		{			
			cloudAbsorption					= 0.0;
		}			
		float3 absorptionColor				= lerp(0.075 + 0.025 * gasSample.g, Square(saturate(dot(-light, normal) + 0.25)), Square(cloudAbsorption)) * 200.0 * g_MaterialDiffuse.rgb;
		absorptionColor						= saturate(exp(-max((float3)0.0, float3(RayleighPhaseFunction(absorptionColor.r), RayleighPhaseFunction(absorptionColor.g), RayleighPhaseFunction(absorptionColor.b)))));
			
		NoL 								= saturate(dot(light, normal));

		lightSum							+= (gasColor * absorptionColor * lightColor) * (cloudAbsorption * NoL * ao);
		
		return LinearToSRGB(float4(1.0 - exp(-lightSum), 1.0));
	}
	///////////////////////////////////////////////////////////////////////////////////////////////// GAS GIANT END /////////////////////////////////////////////////////////////////////////////////////////////////

	float height;
	float2 offset							= 0;
					
	const float steps 						= 32.0;//((32.0 - 24.0 * NoV));
	const float stepsInv					= rcp(steps);
	float rayHeight							= 1.0;
	float oldRay							= 1.0;
				
	float oldTex							= 1;
	float texAtRay							= 0;
	float yIntersect;				
	uv										-= parallaxTS * 0.5 * NoV;
	float2 offsetStep						= parallaxTS * stepsInv * NoV;
	
	float fogDistance 						= 0;
	[loop]
	for (int i = 0; i < steps; ++i)
	{		
		float texAtRay 						= tex2Dgrad(TextureDisplacementSampler, uv + offset, duvx, duvy).a;
		
		if (rayHeight < texAtRay)
		{
			float xIntersect				= (oldRay - oldTex) + (texAtRay - rayHeight);
			xIntersect						= (texAtRay - rayHeight) / xIntersect;
			yIntersect						= (oldRay * xIntersect) + (rayHeight * (1.0 - xIntersect));
			offset							-= (xIntersect * offsetStep);
			break;			
		}			
		fogDistance							+= texAtRay - 1;
		oldRay								= rayHeight;
		rayHeight							-= stepsInv;
		offset 								+= offsetStep;
		oldTex 								= texAtRay;
	}
	uv 										+= offset;
	height									= tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).a;


	
	float4 sampleA;
	float4 sampleB;
	float4 sampleC;
	float3 sampleS = 0;
	
	float fog = 0;
	float fogScatter = 1.0;
	if(fogMode)
	{
		fog = 1-exp(fogDistance * (Pow3(1.0 - NoV) * 0.75 + 0.25));
		fogScatter += (16.0 - 8.0 * NoV) * fog;
	}
	

	float stepsLight			= 32.0 - 31.0 * saturate(NoL);
	NoL							= min(saturate(dot(normal, light)), saturate(NoL * 10.0 + 2.0));
	
	float shadow				= 0.0;
	
	//bias can't branch and grads derivative multiplication is mipping very uneven:(
	float3 sampleBlur = 0;
	float2 rayS = parallaxTS  * 0.01;
	for(int j = 0; j < 5; j++)
	{	
		sampleBlur += tex2Dbias(TextureColorSampler, float4(uv - rayS * j, 0, 2.5 + j)).rgb;		
	}
	if(iceMode)
	{
		sampleS = SRGBToLinear(sampleBlur * 0.2);
	}	
	
	float waterLimit = 0;
	float depthBlur = 1;
//	return float4(tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).rgb, 1);
	///////////////////////////////////////////////////////////////////////////////////////////////// OCEAN START /////////////////////////////////////////////////////////////////////////////////////////////////
	[branch]
	if(oceanMode)
	{
		waterLimit					= g_MaterialSpecular.r;
		float waterGrad 			= height - g_MaterialSpecular.r;
		float waterMask 			= saturate(-waterGrad * 256.0);
		float waterDepth			= 1 - saturate(exp(waterGrad * g_MaterialSpecular.g * 256.0));
		depthBlur 					+= Square(waterDepth) * 16.0;
		
		sampleC						= tex2Dgrad(TextureNormalSampler, uv, duvx * depthBlur, duvy * depthBlur);
		normal 						= normalize(ToWorldSpace(float3x3(tangent, cotangent, normalSphere), GetNormalDXT5(sampleC)));
		
		float2 waterRefraction 		= 0;
		[branch]
		if(waterMask > 0.0)
		{
			
			float waterScale 		= 1.0 / 10.0;
			float waterSpeed 		= 10.0;
			float waterIntensity 	= 0.05;
			
			//return float4(Square(1.0 - waterDepth).rrr, 1);
			//float4 waterNgrad 		= Nnoise(input.posObj * waterScale + length(input.posObj * waterScale) - (1.0 - Square(waterDepth)) * normal * 2 + g_Time * waterSpeed * waterScale) * waterIntensity;
			
		//	float4 waterNgrad 		= Nnoise(input.posObj * waterScale, g_Time * waterSpeed * waterScale);
		//	waterNgrad.xyz 			= mul(normalize(input.normalObj - waterNgrad.xyz * waterIntensity), (float3x3)g_World);
//			waterNgrad.xyz 			= mul(normalize(waterNgrad.xyz), (float3x3)g_World);
		//	waterRefraction 		= ToTangentSpace(float3x3(tangent, cotangent, normal), refract(view, waterNgrad.xyz, 0.75187969924812030075187969924812)).xy;
		//	normal 					= lerp(normal, lerp(waterNgrad.xyz, normalSphere, Square(waterGrad)), waterMask);
			normal 					= lerp(normal, normalSphere, waterMask);
//			return float4(normal, 1);
			sampleC.b 				= 0;//no metal water
		}
		waterRefraction 		*= float2(0.00125, 0.000675);
		uv						+= waterRefraction;
		sampleA					= tex2Dgrad(TextureColorSampler	, uv, duvx * depthBlur, duvy * depthBlur);
		sampleB					= tex2Dgrad(TextureDataSampler	, uv, duvx * depthBlur, duvy * depthBlur);

		sampleA.rgb 			= SRGBToLinear(sampleA.rgb);

		sampleB.w 				= lerp(sampleB.w, (0.5 - 0.45 * rcp(1+length(-pos) * 0.00005)), waterMask);
		
		float3 oceanColor		= SRGBToLinear(g_MaterialAmbient.rgb);
		
		sampleS					= lerp(sampleS, oceanColor, waterMask);
		sampleA					= lerp(sampleA, float4(lerp(pow(oceanColor, waterDepth * float3(1.0, 0.4, 0.45)) * sampleA.rgb, pow(oceanColor, 1 + NoV * 0.5), saturate(waterDepth * (1.5 - NoV))), 0.23), waterMask);
		sampleC.a				= lerp(sampleC.a, waterDepth * 0.1 * (Pow5(1-NoV) + 0.5), waterMask);

//		return LinearToSRGB(float4(sampleA.rgb, 1.0));
	}
	else
	{			
		sampleA					= tex2Dgrad(TextureColorSampler	, uv, duvx * fogScatter, duvy * fogScatter);
		sampleB					= tex2Dgrad(TextureDataSampler	, uv, duvx * fogScatter, duvy * fogScatter);
		sampleC					= tex2Dgrad(TextureNormalSampler, uv, duvx * fogScatter, duvy * fogScatter);
		normal 					= normalize(ToWorldSpace(float3x3(tangent, cotangent, normalSphere), GetNormalDXT5(sampleC)));
		sampleA.rgb 			= SRGBToLinear(sampleA.rgb);
	}
	

//	return LinearToSRGB(float4(sampleS, 1));
	///////////////////////////////////////////////////////////////////////////////////////////////// OCEAN E /////////////////////////////////////////////////////////////////////////////////////////////////

	[branch]
	if(NoL > 0)
	{
		shadow 					= 1.0;
	
		offset 					= 0;
		
		texAtRay 				= saturate(tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).a - waterLimit) + waterLimit + 0.01;
		
		rayHeight				= texAtRay;

		float stepsLightInv		= rcp(stepsLight) * 0.5;//0.5 to match lightTS.z modification
		float dist				= 0;				
		float shadowPenumbra	= 1.25;
		
		for(int j = 0; j < stepsLight; j++)
		{
			if(rayHeight < texAtRay)
			{
				shadow 				= 0;
				break;
			}
			else
			{
				shadow				= min(shadow, (rayHeight - texAtRay) * shadowPenumbra / dist);
			}

			oldRay					= rayHeight;
			rayHeight				+= lightTS.z * stepsLightInv;
			
			offset					+= lightTS.xy * stepsLightInv;
			oldTex					= texAtRay;
			
			texAtRay 				= saturate(tex2Dgrad(TextureDisplacementSampler, uv + offset, duvx, duvy).a - waterLimit) + waterLimit;
	
			dist					+= stepsLightInv;
		}		
	}
	
	
	shadow					= Pow5(shadow);
	float lightRad			= 25000;
	float3 specular = 0;
	float3 diffuse = 0;
	float3 emissive = 0;
	
//	return float4((attenuation).rrr, 1.0);
	#if 0
//	return LinearToSRGB(float4((shadow * NoL).rrr, 1));
//1.5 - saturate(tex2Dlod(TextureDisplacementSampler, float4(uvFlat, 0.0, 4.0)).a - waterLimit);
	float windLevels = 1.5 - saturate(tex2Dlod(TextureDisplacementSampler, float4(uvFlat, 0.0, 2.0)).a - waterLimit);
//	return float4(tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2(-0.00781250, 0) * 1, 0.0, 3.0 * windLevels)).aaa, 1);
	float3 windForce = 			float3( float2( tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2(-0.00781250, 0) * 1, 0.0, 3.0 * windLevels)).a -
												tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2( 0.00781250, 0) * 1, 0.0, 3.0 * windLevels)).a,
												tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2(0,  0.00390625) * 1, 0.0, 3.0 * windLevels)).a -
												tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2(0, -0.00390625) * 1, 0.0, 3.0 * windLevels)).a)/* +
										float2( tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2(-0.01562500, 0) * 1, 0.0, 4.0 * windLevels)).a -
												tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2( 0.01562500, 0) * 1, 0.0, 4.0 * windLevels)).a,
												tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2(0,  0.00781250) * 1, 0.0, 4.0 * windLevels)).a -
												tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2(0, -0.00781250) * 1, 0.0, 4.0 * windLevels)).a) * 0.5 +
										float2( tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2(-0.03125000, 0) * 1, 0.0, 6.0 * windLevels)).a -
												tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2( 0.03125000, 0) * 1, 0.0, 6.0 * windLevels)).a,
												tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2(0,  0.01562500) * 1, 0.0, 6.0 * windLevels)).a -
												tex2Dlod(TextureDisplacementSampler, float4(uvFlat + float2(0, -0.01562500) * 1, 0.0, 6.0 * windLevels)).a) * 0.25*/,
												0);

	windForce.x *= 0.5;

	windForce = ToWorldSpace(float3x3(tangent, cotangent, normal), windForce);
	windForce = input.posObj * 0.001 + mul((float3x3)g_World, windForce) * 4;
	windForce.xz = Rotate(windForce.xz, g_Time * 0.05);
//	return float4(windForce, 1);

	const float3x3 m = float3x3( 0.00,  0.80,  0.60,
								-0.80,  0.36, -0.48,
								-0.60, -0.48,  0.64 );

    float cloudMathTest 	= 0.7500	* Snoise(windForce, g_Time * 0.2); windForce = mul(windForce * 2.01, m);
    cloudMathTest 			+= 0.500	* Snoise(windForce, g_Time * 0.4); windForce = mul(windForce * 3.02, m);
    cloudMathTest 			+= 0.250	* Snoise(windForce, g_Time * 0.6); windForce = mul(windForce * 4.03, m);
    cloudMathTest 			+= 0.125	* Snoise(windForce, g_Time * 0.8);// windForce = mul(windForce * 4.01, m);
	cloudMathTest			= saturate(cloudMathTest);
	return float4(cloudMathTest.rrr, 1);
	
	#endif
	
//	return float4(Snoise(windForce, g_Time * 0.1).rrr, 1);
	PBRProperties Properties;
	
	Properties.Roughness = sampleB.w;
	Properties.EmissiveColor = 0;
	Properties.SpecularColor = sampleA.a * 0.08;
	Properties.DiffuseColor = sampleA.rgb;	
	Properties.AO = 1;
	Properties.SubsurfaceColor = sampleS;
	Properties.SubsurfaceOpacity = sampleC.a;
//	normal	= normalSphere;

	float smoke = 0;
	float shadowSmoke = 1;
	
#if 1	
	float shadowScalar 	= dot(light, input.normal);

	[branch]
	if(volcanoMode)
	{
//		return float4(0.0, 1.0, 0.0, 1.0);
		const float smokeBaseDensity = 2.0;
		const float smokePuffDensity = 16.0;
		float3 SmokeColor = 0;

		smoke				= 1.0 - exp(-VolcanoSmoke(uvFlat, duvx, duvy, smokeBaseDensity, smokePuffDensity).x);
		float2 uvSmoke		= uvFlat - parallaxTS * (smoke * 0.25) * NoV;
		
		float2 SmokeBase	= VolcanoSmoke(uvSmoke, duvx, duvy, smokeBaseDensity, smokePuffDensity);
		smoke				= SmokeBase.x;
		

		shadowSmoke = smoke;
		float smokeSteps	= 15;

		float smokeShadowBlend = (1 - saturate(shadowScalar)) * smokeSteps + 1;
		float smokeStepInv 	= rcp(smokeSteps);
		float SmokeShadowIntensity = saturate(shadowScalar * 0.25 + 0.1);

		float2 shadowStep 		= lightTS.xy * smokeStepInv;
		float2 uvSmokeShadow 	= uvSmoke - shadowStep * 0.5;
		smoke 					= 1 - exp(-smoke);		
		Reyleigh *= 1.0 - smoke;
		Reyleigh += float3(0.53, 0.72, 0.95) * smoke;
		
		if(shadowScalar > -0.25)
		{
			shadow				= max(smoke, shadow);

			for(int k = 0; k < smokeShadowBlend; k++)
			{				
				uvSmokeShadow += shadowStep;
				shadowSmoke -= VolcanoSmoke(uvSmokeShadow, duvx, duvy, smokeBaseDensity, smokePuffDensity).x * saturate(smokeShadowBlend - k) * SmokeShadowIntensity;// * (1+k)* 0.01;			
			}
			shadowSmoke = exp(min(0, shadowSmoke * smokeStepInv));
			//Reyleigh = lerp()shadow *= saturate(lerp(float3(0.118, 0.071, 0.031), float3(0.631, 0.541, 0.447), shadowSmoke), smoke) * shadowSmoke;
			shadow *= shadowSmoke;
		}
		else
		{
			shadowSmoke = 0;
		}
		SmokeColor			= pow(float3(0.631, 0.541, 0.447) * smoke, 2.2);

		Properties.AO		= Square(saturate(exp(-VolcanoSmoke(uv + parallaxTS * NoV * 0.25, duvx, duvy, smokeBaseDensity, smokePuffDensity).x) + smoke));// * 0.75 + 0.25;
		float SmokeRimGlow = Square(1 - (1 - Square(smoke) * (1 - smoke))) * 2;
		
		float SmokeScatter = rcp(1.0 + smoke * 10.0);
		Properties.EmissiveColor = float4(Square(GetLavaFlow(uv, duvx * SmokeScatter, duvy * SmokeScatter)) * 10.0 +	
														(	Square(tex2Dlod(TextureDataSampler, float4(uvSmoke, 0, 2.5)).rgb) + 
															Square(tex2Dlod(TextureDataSampler, float4(uvSmoke, 0, 3.5)).rgb) +
															Square(tex2Dlod(TextureDataSampler, float4(uvSmoke, 0, 4.5)).rgb) + 
															Square(tex2Dlod(TextureDataSampler, float4(uvSmoke, 0, 5.5)).rgb)) * 0.25, 1);
//		return float4(Properties.EmissiveColor.rgb, 1);
		smoke = saturate(smoke);
		Properties.EmissiveColor 	*= Square(1-smoke); 
		Properties.DiffuseColor 	*= (1-smoke);
		Properties.DiffuseColor 	+= SmokeColor * smoke;
		Properties.Roughness 		= max(Properties.Roughness, smoke);
		Properties.SpecularColor	= max(smoke, (float4)0.08);
		normal						= normalize(normal * (1 - smoke) + normalSphere * smoke);
	}

#endif
	float attenuation		= shadow * NoL;//Square(Pow5(shadow));///pow(shadow, shadowContrast);
	
#if 1
	if(volcanoMode == false)
	{
		float dayMask = saturate(-4 * attenuation + 1.0);
		if(dayMask > 0)
			Properties.EmissiveColor.rgb	+= pow(SampleCityLightsBiasLoop(uvFlat, uv, duvx, duvy, 2.0), 2.2) * dayMask * PI;	
	}
#endif
//	return float4(normal, 1);

/*
	Properties.DiffuseColor 	*= (1-Clouds);
	Properties.DiffuseColor 	+= g_CloudColor * Clouds;
	Properties.Roughness 		= max(Properties.Roughness, Clouds);
	Properties.SpecularColor	= max(Clouds, (float4)0.08);
	normal						= normalize(normal * (1 - Clouds) + normalSphere * Clouds);
*/
	// placeholder
	float cloudSample 			= 0;
	float clouds				= 0;
	float cloudShadow			= 1;
	
	NoV							= dot(normal, view);

	float3 diffuseSample		= SRGBToLinear(texCUBE(EnvironmentIlluminationCubeSampler, normal)).rgb;
	Properties.DiffuseColor     += cloudSample;
	cloudShadow					= Square(saturate(cloudShadow + cloudSample));		
	
	float3 reflection			= -(view - 2.0 * normal * NoV);
	float3 reflectionSample 	= SRGBToLinear(texCUBElod(TextureEnvironmentCubeSampler, float4(reflection, GetMipRoughness(Properties.Roughness, 5.0)/*max(cloudShadow * 6.0, Properties.RoughnessMip)*/))).rgb;
//	return LinearToSRGB(float4(ReflectionSample, 1));
	
	AmbientBRDF(diffuse, specular, saturate(abs(NoV)), Properties, reflectionSample, diffuseSample, true);
	
	diffuse 					*= cloudShadow;
	specular 					*= cloudShadow;

	float3 atmosphereAbsorption	= lerp(0.075 + 0.025 * smoke - clouds * 0.05, Square(saturate(dot(-light, normalSphere) + 0.25)), max(shadowSmoke * pow(cloudShadow, 0.25), saturate(pow(clouds, 0.25)))) * 200 * Reyleigh;
//	float3 atmosphereAbsorption	= dot(-light, normalSphere);
	atmosphereAbsorption = saturate(exp(-max((float3)0, float3(	RayleighPhaseFunction(atmosphereAbsorption.r),	RayleighPhaseFunction(atmosphereAbsorption.g),	RayleighPhaseFunction(atmosphereAbsorption.b)))));
//	return float4(atmosphereAbsorption,1);
//	shadow *= atmosphereAbsorption;


	SphereLight(Properties, normal, view, lightPos, lightRad, float4(lightColor.rgb, 0.1), atmosphereAbsorption, specular, diffuse, attenuation, NoL);
//	specular = pow(specular, 1.2 - exp(-g_MaterialDiffuse.rgb) * 0.4);

	emissive += Properties.EmissiveColor;
//	return LinearToSRGB(float4(specular + diffuse, 1.0));

//	return LinearToSRGB(float4(1.0 - exp(-((specular + diffuse) + emissive)), 1.0));
	
	return LinearToSRGB(float4(1.0 - exp(-lerp(((specular + diffuse) + emissive), saturate(dot(normalSphere, light)) + SRGBToLinear(texCUBE(EnvironmentIlluminationCubeSampler, normalSphere)).rgb, fog)), 1.0));
}


///////////////////////////////////////////////atmosphere

	#define NR_SAMPLE_STEPS 8.0f
	#define INV_NR_SAMPLE_STEPS (1.0f / NR_SAMPLE_STEPS)

	#define NR_SUN_STEPS 8.0f
	#define INV_NR_SUN_STEPS (1.0f / NR_SUN_STEPS)

	#define DISTANCE_MULT_VAL 1000.0f


	struct Ray {
		float3 origin;
		float3 direction;
	};

	struct Earth {
		float3 center;
		float earth_radius;
		float atmosphere_radius;
	};

	struct Sky_PBR {
		float hR;
		float hM;

		float inv_hR;
		float inv_hM;

		float g;

		Earth earth;

		float3 transmittance;

		float3 optical_depthR;
		float3 optical_depthM;

		float3 sumR;
		float3 sumM;

		float3 space_sumR;
		float3 space_sumM;

		float3 betaR;
		float3 betaM;

		float phaseR;
		float phaseM;

		float VL;

		float3 sunDir;
	};

	Ray make_ray(float3 origin, float3 direction){
		Ray r;
		r.origin = origin;
		r.direction = direction;
		return r;
	}
	
	bool isect_sphere(Ray ray, float3 sphere_center, float sphere_radius, inout float t0, inout float t1)
	{
		float3 rc = sphere_center - ray.origin;
		float radius2 = sphere_radius * sphere_radius;
		float tca = dot(rc, ray.direction);

		float d2 = dot(rc, rc) - tca * tca;

		float thc = sqrt(radius2 - d2);
		t0 = tca - thc;
		t1 = tca + thc;


		if (d2 > radius2) return false;
/*
		float thc = sqrt(radius2 - d2);
		t0 = tca - thc;
		t1 = tca + thc;
*/
		return true;
	}

	Earth make_earth(float3 center, float earth_radius, float atmosphere_radius){
		Earth s;
		s.center = center;
		s.earth_radius = earth_radius;
		s.atmosphere_radius = atmosphere_radius;
		return s;
	}

	void get_sun_light_space(
		in Sky_PBR sky,
		in Ray ray,
		in float blue_noise,
		inout float optical_depthR,
		inout float optical_depthM)
	{

		float inner_sphere0;
		float inner_sphere1;
		float outer_sphere0;
		float outer_sphere1;
		isect_sphere(ray, sky.earth.center, sky.earth.earth_radius, inner_sphere0, inner_sphere1);
		isect_sphere(ray, sky.earth.center, sky.earth.atmosphere_radius + 0.0001, outer_sphere0, outer_sphere1);

		float march_step = outer_sphere1;
		if(inner_sphere0 > 0){
			march_step = min(inner_sphere0 + 500, outer_sphere1);
		}

		march_step *= INV_NR_SUN_STEPS;

		float3 s = ray.origin + ray.direction * march_step * (0.5 + (blue_noise - 0.5));

		float march_step_multed = march_step * DISTANCE_MULT_VAL;

		for (int i = 0; i < NR_SUN_STEPS; i++) {

			float height = length(s - sky.earth.center) - sky.earth.earth_radius;

			if(height < 0){
				height *= 0.05;
			}

			optical_depthR += exp(-height * sky.inv_hR) * march_step_multed;
			optical_depthM += exp(-height * sky.inv_hM) * march_step_multed;

			s += ray.direction * march_step;

		}

		return;
	}

	void get_incident_light_space(inout Sky_PBR sky, in Ray ray, float step_dist, float atmopshere_thickness, float blue_noise)
	{

		float height = length(ray.origin - sky.earth.center) - sky.earth.earth_radius;

		[branch]
		if(height > atmopshere_thickness){
			return;
		}

		// integrate the height scale
		float hr = exp(-height * sky.inv_hR) * step_dist;
		float hm = exp(-height * sky.inv_hM) * step_dist;

		sky.optical_depthR += hr;
		sky.optical_depthM += hm;

		// gather the sunlight
		Ray light_ray = make_ray(ray.origin, sky.sunDir);
		float optical_depth_lightR = 0;
		float optical_depth_lightM = 0;
		get_sun_light_space(sky,
							light_ray,
							blue_noise,
							optical_depth_lightR,
							optical_depth_lightM);


		float3 tau =	sky.betaR * (sky.optical_depthR + optical_depth_lightR) +
						sky.betaM * (sky.optical_depthM + optical_depth_lightM);

		sky.transmittance = exp(-(sky.betaR * sky.optical_depthR + sky.betaM * sky.optical_depthM));

		float3 attenuation = exp(-tau);

		float shadow_term = 1;

		sky.sumR += hr * attenuation * shadow_term;
		sky.sumM += hm * attenuation * shadow_term;

		tau =			sky.betaR * (sky.optical_depthR) +
						sky.betaM * (sky.optical_depthM);

		attenuation = exp(-tau);

		sky.space_sumR += hr * attenuation;
		sky.space_sumM += hm * attenuation;
	}
	
	
	void GetAtmosphere(	inout float4 atmosphere, 
						float3 sunDir, 
						float3 sunColor, 
						float3 view, 
						float planetRad, 
						float atmoRad, 
						float3 planetPos, 
						float3 rayleighBeta, 
						float3 mieBeta,
						float scatterRayleight,
						float scatterMie)
	{
		Sky_PBR sky;

		sky.hR = (8000.0 * 0.00002) * scatterRayleight;
		sky.hM = (1200.0 * 0.00002) * scatterMie;

		sky.inv_hR = 1.0 / sky.hR;
		sky.inv_hM = 1.0 / sky.hM;

		sky.g = 0.8;

		sky.earth = make_earth(planetPos, planetRad, atmoRad);

		sky.transmittance = 1;
		sky.optical_depthR = 0;
		sky.optical_depthM = 0;

		sky.sumR = 0;
		sky.sumM = 0;
		sky.betaR = rayleighBeta * 1e-6;
		sky.betaM = (float3)(mieBeta * 1e-6);

		sky.sunDir = sunDir;

		sky.space_sumR = 0;
		sky.space_sumM = 0;

		sky.VL = dot(view, sky.sunDir);

		sky.phaseR = RayleighPhaseFunction(sky.VL);
		sky.phaseM = HenyeyGreensteinPhaseFunction(sky.VL, sky.g);

		Ray view_ray = make_ray((float3)0.0, view);

		float atmopshere_thickness = sky.earth.atmosphere_radius - sky.earth.earth_radius;

		float t0 = 0;
		float t1 = 0; 

		if(isect_sphere(view_ray, sky.earth.center, sky.earth.atmosphere_radius, t0, t1))
		{

			float inner_sphere0 = 0;
			float inner_sphere1 = 0;
			isect_sphere(view_ray, sky.earth.center, sky.earth.earth_radius, inner_sphere0, inner_sphere1);

			float start_dist = t0;

			float end_dist = min(inner_sphere0, t1);

			float step_lengths = INV_NR_SAMPLE_STEPS * (end_dist - start_dist);

			float blue_noise = 0;
			
			float dist = max(0.01f, start_dist - step_lengths * (0.175 + blue_noise * 0.1));
			float prev_dist = dist;

			float3 avg_space_light = 0;

			for(int i = 0; i < NR_SAMPLE_STEPS; i++){

				dist += step_lengths;

				float step_dist = (dist - prev_dist) * DISTANCE_MULT_VAL;

				float3 wp = view * dist;

				Ray ray = make_ray(wp, view);
				get_incident_light_space(sky, ray, step_dist, atmopshere_thickness, blue_noise);

				avg_space_light += texCUBE(EnvironmentIlluminationCubeSampler, float4(normalize(wp - sky.earth.center), 0)).rgb;

				prev_dist = dist;
			}

			avg_space_light *= INV_NR_SAMPLE_STEPS;

			float3 rayleigh_color = (sky.sumR * sky.phaseR * sky.betaR);
			float3 mie_value = (sky.sumM * sky.phaseM * sky.betaM);

			float3 ambient_colorR = (sky.space_sumR * sky.betaR);
			float3 ambient_colorM = (sky.space_sumM * sky.betaM);
			
			//NOTE lazy Ozone tinting, proper could be done like this: https://publications.lib.chalmers.se/records/fulltext/203057/203057.pdf
			atmosphere.a = saturate(Luminance(1 - sky.transmittance));
			//atmosphere.a = saturate(1 - dot(sky.transmittance, (float3)1));
			//atmosphere.a = exp(-sky.transmittance);
			atmosphere.rgb = ((rayleigh_color + mie_value) * sunColor + (ambient_colorR + ambient_colorM) * avg_space_light);			
			atmosphere.rgb /= (atmosphere.rgb + 1.0);
		}
	}




struct VsCloudsOutput
{
	float4 Position: POSITION;
	float2 TexCoord0: TEXCOORD0;
	float3 Light: TEXCOORD1;
	float3 Normal: TEXCOORD2;
	float3 View: TEXCOORD3;
	float3 Pos: TEXCOORD4;
	float3 PlanetPos: TEXCOORD5;
	float PercentHeight : COLOR0;
};

VsCloudsOutput 
RenderCloudVertex(float thicknessModifier, float3 iPosition, float3 iNormal, float2 iTexCoord)
{
//	const bool GasGiantMode = round(g_MaterialSpecular.a * 4) == 2;//planet that are lave based, no citylight
	
	VsCloudsOutput o;
    //Position
    float3 positionInWorldSpace = -mul(float4(iPosition, 1.f), g_World).xyz;
	float inflateScale = distance((mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz) - positionInWorldSpace, positionInWorldSpace) * (1.0 / 131072.0) + 1;
	float atmosphereThickness = 1.0 + frac(g_MaterialGlossiness) * inflateScale;
	//Final Position
	
	o.Position = mul(float4(iPosition * atmosphereThickness, 1.0f), g_WorldViewProjection);
//	if(GasGiantMode)
//		o.Position /= 1-float(GasGiantMode);//NaN cull, compiler wont allow us to just flat out divide by 0. In effect, the pixel shader will never execute
	
	//Texture Coordinates
    o.TexCoord0 = iTexCoord; 
	
    //Calculate  Normal       
    o.Normal = normalize(mul(iNormal, (float3x3)g_World));
    

	o.Pos = positionInWorldSpace;// / atmosphereThickness;// * ATMOSPHERE_SCALE;//(1.0 / ATMOSPHERE_SCALE);
	o.PlanetPos = ((mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz - positionInWorldSpace) * atmosphereThickness) / g_Radius;
    //Calculate Light
	o.Light = normalize(((g_Light0_Position - positionInWorldSpace) * atmosphereThickness) / g_Radius);
	
	//Calculate ViewVector
	o.View = normalize(-(positionInWorldSpace * atmosphereThickness) / g_Radius);
	
	o.PercentHeight = abs(iPosition.y)/g_Radius;          
    return o;
}

VsCloudsOutput 
RenderCloudsVS(
	float3 iPosition:POSITION, 
	float3 iNormal:NORMAL,
	float2 iTexCoord:TEXCOORD1)		
{
	return RenderCloudVertex(1.00, iPosition, iNormal, iTexCoord);
}

void RenderCloudsPS(VsCloudsOutput i, out float4 oColor0:COLOR0) 
{ 
	const bool oceanMode 					= int(g_MaterialSpecular.a) 		== 1;//planets that got oceans, support citylights too
	const bool volcanoMode 					= int(g_MaterialSpecular.a * 4.0) 	== 1;//planet that are lave based, no citylight
	const bool gasGiantMode 				= int(g_MaterialSpecular.a * 4.0) 	== 2;//planet that are lave based, no citylight


	//Light and Normal - renormalized because linear interpolation screws it up
	float3 light 			= normalize(i.Light);
	float3 normal 			= normalize(i.Normal);
//	float3 view 			= normalize(i.View);
	float3 lightColor 		= SRGBToLinear(g_Light0_DiffuseLite.rgb);		
	
#ifndef DEBUG_MAKE_PLANET_SPHEREPROBE
	#if 1
		oColor0 = 0;		

			float atmosphereThickness 	= 1.0 + frac(g_MaterialGlossiness);
			float planet_tweak_scale 	= floor(g_MaterialGlossiness) * 0.1;//earth scale
			
			float3 sunDir 				= normalize(i.Light * planet_tweak_scale);
			float3 view 				= normalize(i.View * planet_tweak_scale);
			float planetRad 			= planet_tweak_scale;
			float atmoRad 				= planet_tweak_scale * atmosphereThickness;//1.0018835;//earth scale, unfortunately we don't have the precision for that
			float4 rayData				= pow(g_MaterialDiffuse, 2.2);
			float3 rayleighBeta			= rayData.rgb * 50.0;
			
			float3 mieBeta				= rayData.a * 50.0;
	//		float3 ozoneBeta			= 0;
	//		if(oceanMode)
	//			ozoneBeta += Ozone * g_MaterialSpecular.b;
			float scatterRayleight		= 250.0;
			float scatterMie			= 250.0;
			float3 sunColor				= 100.0 * lightColor;
			float3 planetPos			= i.PlanetPos * planet_tweak_scale;//yup remap for consistency
			GetAtmosphere(	oColor0, 
							sunDir, 
							sunColor, 
							view, 
							planetRad, 
							atmoRad, 
							planetPos, 
							rayleighBeta, 
							mieBeta,
							scatterRayleight,
							scatterMie);

	#else
		oColor0 = 0;
	#endif
#else
	oColor0 = float4(i.PlanetPos, 1);//float4(frac(distance(mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz, i.Pos - mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz).xxx / 1000.0), 1);
#endif
}

technique RenderWithoutPixelShader
{
    pass Pass0
    {   	        
        VertexShader = compile vs_1_1 RenderSceneVSSimple();
        PixelShader = NULL;
		Texture[0] = <g_TextureDiffuse0>;
    }
}

technique RenderWithPixelShader
{
	pass Pass0
    {          
        VertexShader 		= compile vs_1_1 RenderSceneVS();
        PixelShader 		= compile ps_3_0 RenderScenePS();
        
		AlphaTestEnable 	= FALSE;
        AlphaBlendEnable 	= TRUE;
		SrcBlend 			= ONE;
		DestBlend 			= ZERO;    
		ZEnable 			= true;
		ZWriteEnable 		= true;
    }
    
    pass PassCloudLayer
    {
		VertexShader = compile vs_1_1 RenderCloudsVS();
		PixelShader = compile ps_3_0 RenderCloudsPS();

		ZEnable 			= true;
		ZWriteEnable 		= false;		
		AlphaTestEnable 	= TRUE;		
		AlphaBlendEnable 	= TRUE;
		SrcBlend 			= SRCALPHA;
		DestBlend 			= INVSRCALPHA;
		
    }
}