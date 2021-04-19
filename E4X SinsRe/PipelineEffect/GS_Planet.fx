//math constants
#define SF (1.0/float(0xffffffffU))
#define PI 3.1415926535897932384626433832795
#define INVPI (1.0 / PI)

//options, you can comment them out to tweak performance if you have performance problems
#define SUPPORT_PARALLAX_DISPLACEMENT
#define SUPPORT_PARALLAX_SHADOWS        //only works if SUPPORT_PARALLAX_DISPLACEMENT is also on!

#define SUPPORT_SUBSURFACE

//turns the planets in sphere probes to debug cubemaps
//#define DEBUG_MAKE_PLANET_SPHEREPROBE
#define DEBUG_SPHEREPROBE_METAL
#define DEBUG_SPHEREPROBE_MULTISCATTER true

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

sampler CloudLayerSampler = sampler_state
{
    Texture	= <g_TextureDiffuse2>;    
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
void
RenderSceneVSSimple(
	float3 iPosition : POSITION, 
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
	float3 lightDir : TEXCOORD2;
//	float3 viewDir : TEXCOORD3;
	float3 pos : TEXCOORD3;
	float3 posObj : TEXCOORD4;
	float3 normalObj : TEXCOORD5;
	float3 PlanetPos : TEXCOORD6;
};

VsSceneOutput RenderSceneVS(
	float3 position : POSITION, 
	float3 normal : NORMAL,
	float2 texCoord : TEXCOORD0)	
{
	VsSceneOutput output;  
	
	output.position = mul(float4(position, 1.0f), g_WorldViewProjection);
    output.texCoord = texCoord; 
	output.normal = mul(normal, (float3x3)g_World);
    output.lightDir = normalize(g_Light0_Position - output.position);
	float3 positionInWorldSpace = mul(float4(position, 1.f), g_World).xyz;
	
	output.PlanetPos = (positionInWorldSpace - mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz);
//    output.viewDir = normalize(-positionInWorldSpace);
    output.pos = positionInWorldSpace;
	output.posObj = position;
	output.normalObj = normal;
    return output;
}

float4 SRGBToLinear(float4 color)
{
	//return color;

	//When external colors and the data texture are redone this can be reenabled.
	return float4(color.rgb * (color.rgb * (color.rgb * 0.305306011f + 0.682171111f) + 0.012522878f), color.a);
}
float3 SRGBToLinear(float3 color)
{
	//return color;

	//When external colors and the data texture are redone this can be reenabled.
	return float3(color * (color * (color * 0.305306011f + 0.682171111f) + 0.012522878f));
}

float4 LinearToSRGB(float4 color)
{
	//return color;

	//When external colors and the data texture are redone this can be reenabled.
	float3 S1 = sqrt(color.rgb);
	float3 S2 = sqrt(S1);
	float3 S3 = sqrt(S2);
	return float4(0.662002687 * S1 + 0.684122060 * S2 - 0.323583601 * S3 - 0.225411470 * color.rgb, color.a);
}

	float Square(float X)
	{
		return X * X;
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
	
	float ToLinear(float aGamma)
	{
		return pow(aGamma, 2.2);
	}

float3 mod289(float3 x)
{
    return x - floor(x / 289.0) * 289.0;
}

float4 mod289(float4 x)
{
    return x - floor(x / 289.0) * 289.0;
}

float4 permute(float4 x)
{
    return mod289((x * 34.0 + 1.0) * x);
}

float4 taylorInvSqrt(float4 r)
{
    return 1.79284291400159 - r * 0.85373472095314;
}



float4 snoise(float3 v)
{
    const float2 C = float2(1.0 / 6.0, 1.0 / 3.0);

    // First corner
    float3 i  = floor(v + dot(v, (float3)(C.y)));
    float3 x0 = v   - i + dot(i, (float3)(C.x));

    // Other corners
    float3 g = step(x0.yzx, x0.xyz);
    float3 l = 1.0 - g;
    float3 i1 = min(g.xyz, l.zxy);
    float3 i2 = max(g.xyz, l.zxy);

    float3 x1 = x0 - i1 + C.x;
    float3 x2 = x0 - i2 + C.y;
    float3 x3 = x0 - 0.5;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    float4 p =
      permute(permute(permute(i.z + float4(0.0, i1.z, i2.z, 1.0))
                            + i.y + float4(0.0, i1.y, i2.y, 1.0))
                            + i.x + float4(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float4 j = p - 49.0 * floor(p / 49.0);  // mod(p,7*7)

    float4 x_ = floor(j / 7.0);
    float4 y_ = floor(j - 7.0 * x_); 

    float4 x = (x_ * 2.0 + 0.5) / 7.0 - 1.0;
    float4 y = (y_ * 2.0 + 0.5) / 7.0 - 1.0;

    float4 h = 1.0 - abs(x) - abs(y);

    float4 b0 = float4(x.xy, y.xy);
    float4 b1 = float4(x.zw, y.zw);

    float4 s0 = floor(b0) * 2.0 + 1.0;
    float4 s1 = floor(b1) * 2.0 + 1.0;
    float4 sh = -step(h, (float4)0.0);

    float4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    float4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    float3 g0 = float3(a0.xy, h.x);
    float3 g1 = float3(a0.zw, h.y);
    float3 g2 = float3(a1.xy, h.z);
    float3 g3 = float3(a1.zw, h.w);

    // Normalize gradients
    float4 norm = taylorInvSqrt(float4(dot(g0, g0), dot(g1, g1), dot(g2, g2), dot(g3, g3)));
    g0 *= norm.x;
    g1 *= norm.y;
    g2 *= norm.z;
    g3 *= norm.w;

    // Compute noise and gradient at P
    float4 m = max(0.6 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    float4 m2 = m * m;
    float4 m3 = m2 * m;
    float4 m4 = m2 * m2;
    float3 grad =
      -6.0 * m3.x * x0 * dot(x0, g0) + m4.x * g0 +
      -6.0 * m3.y * x1 * dot(x1, g1) + m4.y * g1 +
      -6.0 * m3.z * x2 * dot(x2, g2) + m4.z * g2 +
      -6.0 * m3.w * x3 * dot(x3, g3) + m4.w * g3;
    float4 px = float4(dot(x0, g0), dot(x1, g1), dot(x2, g2), dot(x3, g3));
    return float4(grad, dot(m4, px));
}
		
	struct PBRProperties
	{
		float3 SpecularColor;
		float3 DiffuseColor;
		float4 EmissiveColor;
		float Roughness;
		float RoughnessMip;
		float AO;
		float SubsurfaceOpacity;
	};

	PBRProperties UnpackProperties(float4 colorSample, float4 dataSample, float4 normalSample)
	{
		PBRProperties Output;
		Output.SpecularColor 		= max((float3)0.04, dataSample.r * colorSample.rgb);
		Output.DiffuseColor 		= saturate(colorSample.rgb  - Output.SpecularColor);
		Output.EmissiveColor 		= float4(Square(colorSample.rgb) * 8.0, colorSample.a) * ToLinear(dataSample.g);
		Output.Roughness 			= max(0.02, dataSample.w);
		Output.RoughnessMip 		= dataSample.w * 8.0;
		Output.AO 					= normalSample.b;
		Output.SubsurfaceOpacity	= normalSample.r;
		return Output;
	}
	
	// Frostbite presentation (moving frostbite to pbr)
	float3 GetSpecularDominantDir(float3 vN, float3 vR, PBRProperties Properties)
	{
		float InvRoughness = 1.0 - Properties.Roughness;
		float lerpFactor = saturate(InvRoughness * (sqrt(InvRoughness) + Properties.Roughness));
	
		return lerp(vN, vR, lerpFactor);
	}

	// Brian Karis(Epic's) optimized unified term derived from Call of Duty metallic/dielectric term
	float3 AmbientBRDF(float NoV, PBRProperties Properties)
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

	float3 AmbientBRDF(float NoV, float Roughness, float3 SpecularColor)
	{
		float4 r = Roughness * float4(-1.0, -0.0275, -0.572, 0.022) + float4(1.0, 0.0425, 1.04, -0.04);
		float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
		float2 AB = float2(-1.04, 1.04) * a004 + r.zw;
	
		AB.y *= (1.0 - 1.0 / (1.0 + max(0.0, 50.0 * SpecularColor.g))) * 3.0;
	
		return SpecularColor * AB.x + AB.y;
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
//			float3 Fr = max((float3)(1.0 - Properties.Roughness), NoV) - NoV;
//			float3 k_S = Properties.SpecularColor * (NoV + Fr * Pow5(1.0 - NoV));
		
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
	
	void AmbientBRDF(inout float3 diffuse, inout float3 specular, float NoV, float Roughness, float3 SpecularColor, float3 DiffuseColor, const float3 radiance = 1.0, const float3 irradiance = 1.0, const bool multiscatter = true)
	{

		float4 r = Roughness * float4(-1.0, -0.0275, -0.572, 0.022) + float4(1.0, 0.0425, 1.04, -0.04);
		float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
		float2 AB = float2(-1.04, 1.04) * a004 + r.zw;
	
		AB.y *= (1.0 - 1.0 / (1.0 + max(0.0, 50.0))) * 3.0;
		
		if(multiscatter)
		{

		    // Roughness dependent fresnel, from Fdez-Aguera
//			float3 Fr = max((float3)(1.0 - Properties.Roughness), NoV) - NoV;
//			float3 k_S = Properties.SpecularColor * (NoV + Fr * Pow5(1.0 - NoV));
		
			float3 FssEss = SpecularColor * AB.x + AB.y;
		
			// Multiple scattering, from Fdez-Aguera
			float Ems = (1.0 - (AB.x + AB.y));
			float3 F_avg = NoV + (1.0 - NoV) * rcp(21.0);
			float3 FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
			float3 k_D = DiffuseColor * (1.0 - FssEss - FmsEms);
			
			diffuse 	+= (FmsEms + k_D) * irradiance;
			specular 	+= FssEss * radiance;
		}
		else
		{
			diffuse 	+= DiffuseColor * irradiance;
			specular 	+= (SpecularColor * AB.x + AB.y) * radiance;
		}
		
	}
	
	// Frostbite presentation (moving frostbite to pbr)
	float3 GetDiffuseDominantDir(float3 N, float3 V, float NoV, PBRProperties Properties)
	{
		float a = 1.02341 * Properties.Roughness - 1.51174;
		float b = -0.511705 * Properties.Roughness + 0.755868;
		// The result is not normalized as we fetch in a cubemap
		return lerp(N , V , saturate((NoV * a + b) * Properties.Roughness));
	}
	
	struct PBRDots
	{
		float NoV;
		float NoL;
		float VoL;
		float NoH;
		float VoH;
	};
	
	PBRDots GetDots(float3 N, float3 V, float3 L)
	{	
		PBRDots Ouput;
		Ouput.NoL = dot(N, L);
		Ouput.NoV = dot(N, V);
		Ouput.VoL = dot(V, L);
		float invLenH = rcp(sqrt( 2.0 + 2.0 * Ouput.VoL));
		Ouput.NoH = saturate((Ouput.NoL + Ouput.NoV) * invLenH);
		Ouput.VoH = saturate(invLenH + invLenH * Ouput.VoL);
		Ouput.NoL = saturate(Ouput.NoL);
		Ouput.NoV = saturate(abs(Ouput.NoV * 0.9999 + 0.0001));
		return Ouput;
	}
	
	// Diffuse term
	float3 DiffuseBurley(PBRProperties Properties, PBRDots Dots)
	{
		float FD90 = 0.5 + 2.0 * Square(Dots.VoH) * Properties.Roughness;
		float FdV = 1.0 + (FD90 - 1.0) * Pow5( 1.0 - Dots.NoV );
		float FdL = 1.0 + (FD90 - 1.0) * Pow5( 1.0 - Dots.NoL );
		return Properties.DiffuseColor * INVPI * FdV * FdL; // 0.31831 = 1/pi
	}

	// Specular lobe
	float D_GGX(PBRProperties Properties, PBRDots Dots)
	{
		float a2 = Pow4(Properties.Roughness);	
		float d = Square((Dots.NoH * a2 - Dots.NoH) * Dots.NoH + 1.0);
		return a2 / (PI * d);				
	}

	// Geometric attenuation
	float V_GGX(PBRProperties Properties, PBRDots Dots)
	{
		float a = Square(Properties.Roughness);
		float Vis_SmithV = Dots.NoL * (Dots.NoV * (1.0 - a) + a);
		float Vis_SmithL = Dots.NoV * (Dots.NoL * (1.0 - a) + a);
		return 0.5 * rcp(Vis_SmithV + Vis_SmithL);
	}
	
	// Fresnel term
	float3 F_GGX(PBRProperties Properties, PBRDots Dots)
	{
		float Fc = Pow5(1.0 - Dots.VoH);

		return saturate(50.0 * Properties.SpecularColor.g) * Fc + (1.0 - Fc) * Properties.SpecularColor;
	}
	
	// All in one, Epics implementation, originally from Disney/Pixar Principled shader
	// NOTE!! To mitigate a floating point overflow, on specular peaks we saturate the result
	// we should bloom it, but we can't.
	float3 SpecularGGX(PBRProperties Properties, PBRDots Dots)
	{
		return saturate(F_GGX(Properties, Dots) * (D_GGX(Properties, Dots) * V_GGX(Properties, Dots)));
	}

	// Morten Mikkelsen cotangent derivative normal mapping, more accurate than using mesh tangents/normals, handles mirroring and radial symmetry perfectly.
	void CotangentDerivativeBase(float2 DUVX, float2 DUVY, float3 Pos, float3 N, float3 L, float3 V, float2 UV, inout float3 Lt, inout float2 OffsetParallax, inout float3x3 TBN)
	{
		#if 0
			float3 DPX 			= ddx(Pos);
			float3 DPY 			= ddy(Pos);
			float3 DPXPerp 		= cross(N, DPX);		
			float3 DPYPerp 		= cross(DPY, N);		
			float3 Tangent 		= DPYPerp * DUVX.x + DPXPerp * DUVY.x;
			float3 Cotangent 	= DPYPerp * DUVX.y + DPXPerp * DUVY.y;
			float InvMax		= pow(max(dot(Tangent, Tangent), dot(Cotangent, Cotangent)), -0.5);
			Tangent				*= InvMax;
			Cotangent 			*= InvMax;
		#else
		//THIS WILL ONLY WORK ON SPHERE UNWRAPPED PLANETS
			float3 Tangent 			= normalize(cross(N, float3(0.0, 1.0, 0.0)));
			float3 Cotangent 		= normalize(cross(N, Tangent));	
		#endif
		Lt = mul(float3x3(Tangent, Cotangent, N), L);
		float3 Vt = mul(float3x3(Tangent, Cotangent, N), V);
		
//		Lt.z = max(0.00001, (Lt.z + 0.1) * (1.0 / 1.1));//max(0.00001, sqrt(Lt.z * 0.5 + 0.5));
		Lt.z = max(0.00001, sqrt(Lt.z * 0.5 + 0.5));

		Lt.z *= 100.0;
		Lt = normalize(Lt);
		Vt = normalize(Vt);
		
		
		
//		OffsetShadow = (Lt.xy / max(0.00001, sqrt(Lt.z * 0.5 + 0.5))) * 0.01;
//		OffsetParallax = (-Vt.xy / max(0.00001, sqrt(Vt.z * 0.5 + 0.5))) * 0.01;	
//		OffsetParallax = (-Vt.xy / max(0.00001, (Vt.z + 0.2) * (1.0 / 1.2))) * 0.01;	
//		OffsetParallax = -Vt.xy * 0.004;//(Vt.xy / Vt.z) * 0.002;
		OffsetParallax = -(Vt.xy / Vt.z) * 0.01;
		TBN = float3x3(Tangent, Cotangent, N);	
	}

	
	// recreate blue normal map channel
	float DeriveZ(float2 Nxy)
	{	
	
		float Z = sqrt(abs(1.0 - Square(Square(Nxy.x) - Square(Nxy.y))));
		
		return Z;
	}
		
	// recreate blue normal map channel
	float2 ToNormalSpace(float2 Nxy)
	{	
		return Nxy * 2.0 - 1.0;
	}	
	
	float3 GetNormalDXT5(float4 N)
	{
		float2 Nxy = N.wy;
//		Nxy.y = 1.0 - Nxy.y;
		Nxy = ToNormalSpace(Nxy);
		// play safe and normalize
		return normalize(float3(Nxy, DeriveZ(Nxy)));
	}


float4 GetSpecularColor(float3 light, float3 normal, float3 view, float4 colorSample)
{
	float cosang = clamp(dot(reflect(-light, normal), view), 0.00001f, 0.95f);
	float glossScalar = colorSample.a;
	float specularScalar = pow(cosang, g_MaterialGlossiness) * glossScalar;
	return (g_Light0_Specular * specularScalar);
}

float4 GetLightColor(float dotLight, float exponent, float blackThreshold, float invertPercentage, float4 liteLightColor, float4 darkLightColor)
{
	float lerpPercent = pow(max(dotLight, 0.f), exponent);
	lerpPercent = lerp(lerpPercent, 1.f - lerpPercent, invertPercentage);
	
	float colorLerpPercent = min(lerpPercent / blackThreshold, 1.f);
		
	float blackRange = max(lerpPercent - blackThreshold, 0.f);
	float blackLerpPercent = blackRange / (1.f - blackThreshold);

	float4 newColor = lerp(liteLightColor, darkLightColor, colorLerpPercent);
	return lerp(newColor, 0.0f, blackLerpPercent);	
}
float mapClouds(float c)
{
	return (1.0 - exp(-Square(c) * g_MaterialAmbient.a * 5.0));
}

float atan2safe(float y, float x)
{
	return atan2(y, x);
}



	float rayleigh_phase_func(float VL)
	{
		return
				3. * (1. + VL*VL)
		/ //------------------------
					(16. * PI);
	}

	float henyey_greenstein_phase_func(float VL, float g)
	{
		return
							(1. - g*g)
		/ //---------------------------------------------
			((4. + PI) * pow(abs(1. + g*g - 2.*g*VL) + 0.0001, 1.5));
	}

	// Schlick Phase Function factor
	// Pharr and  Humphreys [2004] equivalence to g above
	float schlick_phase_func(float VL, float g)
	{
	const float k = 1.55*g - 0.55 * (g*g*g);				
		return
						(1. - k*k)
		/ //-------------------------------------------
			(4. * PI * (1. + k*VL) * (1. + k*VL));
	}
	
	// Hash without Sine
	// MIT License...
	// Copyright (c)2014 David Hoskins.
	// https://www.shadertoy.com/view/4djSRW
	float Hash13(float3 p3)
	{
		p3  = frac(p3 * .1031);
		p3 += dot(p3, p3.yzx + 33.33);
		return frac((p3.x + p3.y) * p3.z);
	}
#define M3 float3x3( 0.00,  0.80,  0.60, -0.80,  0.36, -0.48, -0.60, -0.48,  0.64)
	
	float Noise13(	float3 P,
				const int Detail = 1, 
				const float Dimension = 0.0, 
				const float Lacunarity = 2.0,
				//const float3 Phase	= 0.0,
				const bool Smooth = true)
{	
	float Sum = 0;
		
	for(int j = 0; j < Detail; j++)
	{
		//P += Phase;
		float3 i = floor(P);
		float3 f = frac(P);
		if(Smooth)
			f *= f * (3.0 - 2.0 * f);

		Sum += mad(lerp(lerp(lerp(	Hash13(i + float3(0, 0, 0)),
									Hash13(i + float3(1, 0, 0)), f.x),
							lerp(	Hash13(i + float3(0, 1, 0)),
									Hash13(i + float3(1, 1, 0)), f.x), f.y),
						lerp(lerp(	Hash13(i + float3(0, 0, 1)),
									Hash13(i + float3(1, 0, 1)), f.x),
							lerp(	Hash13(i + float3(0, 1, 1)),
									Hash13(i + float3(1, 1, 1)), f.x), f.y), f.z), 2, -1) * (1 - Dimension);
		P = mul(P * Lacunarity, M3);
	}
	return Sum * rcp(Detail);	
}
inline float2 Rotate(float2 P, float R)
{
	float S, C;
	sincos(R, S, C);
	return mul(float2x2(C, -S, S, C), P);
}

inline float2 VolcanoSmoke(float2 uv, float2 duvx, float2 duvy, float smokeDensity, float smokePuffyness)	
{
	float2 SmokeSample = tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).gb;
	float SmokeSpeed = (g_Time * 0.25 + (SmokeSample.r * 4 + SmokeSample.g));
	float SmokeStep = floor(SmokeSpeed) * 0.125;
	float SmokeBlend = frac(SmokeSpeed);
	float SmokeAnimation = lerp(tex2Dgrad(TextureDisplacementSampler, float2(uv.x + frac(SmokeStep - g_Time * 0.001), uv.y), duvx, duvy).b, 
								tex2Dgrad(TextureDisplacementSampler, float2(uv.x + frac(SmokeStep + 0.125 - g_Time * 0.001), uv.y), duvx, duvy).b, SmokeBlend)
								* smokePuffyness;
	return float2(max(0, Square(SmokeSample.r) * smokeDensity + SmokeSample.r * SmokeAnimation), SmokeAnimation);
}

inline float3 GetFlow(float Timer)
{
	return float3(frac(float2(Timer, Timer + 0.5)), abs(frac(Timer) * 2 - 1));
}

inline float3 LaveFlowTest(float2 uv, float2 duvx, float2 duvy)
{
	float2 FlowRaw = tex2Dlod(TextureNormalSampler, float4(uv, 0, 2)).yw * 2.0 - 1.0;
	float LavaMask = tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).r;
	float LavaSpeed = -g_Time * 0.25;
	if(dot(FlowRaw.yx, FlowRaw.yx) > 0.0001)//could lead to a NaN
	{
		FlowRaw = normalize(FlowRaw.yx) * rcp(float2(1024, 512)) * LavaMask;
	}
	else
	{
		FlowRaw = 0;	
	}
	
	float3 FlowBase = GetFlow(g_Time * 0.25);

	return lerp(tex2Dgrad(TextureDataSampler, uv - FlowRaw * FlowBase.x, duvx, duvy).rgb,
				tex2Dgrad(TextureDataSampler, uv - FlowRaw * FlowBase.y, duvx, duvy).rgb, FlowBase.z);
}

float4 mapGasGiant(float4 map)
{
	return lerp(float4(map.r, map.g, 1.0, 1.0 - map.a), map * map * (3.0 - 2.0 * map), g_MaterialSpecular.r);
}

float4 mapGasGiant(float map)
{
	return lerp(0.5, map, g_MaterialSpecular.r);
}

//float3 AuroaBorialis(float)

float4 RenderScenePS(VsSceneOutput input) : COLOR0
{
//	return float4(tex2D(TextureColorSampler, input.texCoord + float2(0.05, 0)).rgb, 1);
	const bool OceanMode = int(g_MaterialSpecular.a) == 1;//planets that got oceans, support citylights too
	const bool VolcanoMode = int(g_MaterialSpecular.a * 4) == 1;//planet that are lave based, no citylight
	const bool GasGiantMode = int(g_MaterialSpecular.a * 4) == 2;//planet that are lave based, no citylight
	float3 Reyleigh = g_MaterialDiffuse.rgb;
	
	float2 OffsetParallax;
	float3 ShadowVector;
	float3x3 TBN;
	float3 normal = normalize(input.normal);
		//small improvement to the polar singularity, not perfect but better
//	input.pos += (normal - (input.PlanetPos / g_Radius)) * g_Radius;

	float3 normalSphere = normal;

	float3 view = normalize(-input.pos);
	float3 light = normalize(input.lightDir);	
	
	float2 uv				= input.texCoord;	
	float2 duvx				= ddx(uv);
	float2 duvy				= ddy(uv);

	#if 0
		return float4(normalize(cross(ddx(input.pos), ddy(input.pos))) * 0.5 + 0.5, 1);
	#endif

	CotangentDerivativeBase(duvx, duvy, input.pos, normal, light, view, input.texCoord, ShadowVector, OffsetParallax, TBN);
	float NoV				= abs(dot(view, normal));
	float NoL 				= dot(normal, light);
	float3 lightColor 		= SRGBToLinear(g_Light0_DiffuseLite.rgb) * PI;		
	
	#ifdef DEBUG_MAKE_PLANET_SPHEREPROBE 
		//test that the mips are correct Texture
		float Roughness_debug = abs(fmod(g_Time * 0.25, 2.0) - 1);
		float3 Radiance_debug = SRGBToLinear(texCUBElod(TextureEnvironmentCubeSampler, float4(-(view - 2.0 * normal * dot(view, normal)), Roughness_debug  * 6.0))).rgb;
		float3 Irradiance_debug = SRGBToLinear(texCUBElod(EnvironmentIlluminationCubeSampler, float4(normal, 0))).rgb; 
		float3 Diffuse_debug = 0;
		float3 Specular_debug = 0;
		#ifdef DEBUG_SPHEREPROBE_METAL
			float3 SpecularColor_debug = 1.0;
			float3 DiffuseColor_debug = 0;
		#else			
			float3 SpecularColor_debug = 0.04;
			float3 DiffuseColor_debug = 1.0;
		#endif
		
		AmbientBRDF(Diffuse_debug, Specular_debug, NoV, Roughness_debug, SpecularColor_debug, DiffuseColor_debug, Radiance_debug, Irradiance_debug, DEBUG_SPHEREPROBE_MULTISCATTER);
		return LinearToSRGB(float4(Diffuse_debug + Specular_debug, 1));
	#endif	
	
	if(GasGiantMode)
	{
/*		
		float3 AuroaBorialis = 0;
//			input.texCoord
//		
		float variation = tex3Dlod(NoiseSampler, float4(input.normalObj + frac(g_Time * float3(0.01, 0.0134, 0.0976)), 0)).x - tex3Dlod(NoiseSampler, float4(input.normalObj - frac(g_Time * float3(0.0124, 0.0114, 0.0973)), 0)).x;
		//AuroaBorialis = input.texCoord.y * 10 + textureNoise(float3(input.posObj.xz, time)
		float band = tex3Dlod(NoiseSampler, float4(float3(input.texCoord.x + OffsetParallax.x, 0, 0) + frac(g_Time * float3(0.01, 0.0134, 0.0976)), 0)).x - tex3Dlod(NoiseSampler, float4(float3(input.texCoord.x + OffsetParallax.x, 0, 0) - frac(g_Time * float3(0.0124, 0.0114, 0.0973)), 0)).x;
		
		float AuroaBorialisOffset = input.texCoord.y + OffsetParallax.y;
		float AuroaBorialisAtt = 1;
		float aSteps = 20;
		float aStepsInv = rcp(aSteps);
		for(int a = 0; a < aSteps; a++)
		{
			
			AuroaBorialis += Square(saturate((1-abs((abs(AuroaBorialisOffset - 0.5) * 20) - 9 + variation)) * abs(frac(variation * 4) - 0.5))) * AuroaBorialisAtt;
//			AuroaBorialis += Square(saturate((1-abs((abs(AuroaBorialisOffset - 0.5) * 100) - 40 + variation * 5)))) * AuroaBorialisAtt;
			AuroaBorialisOffset -= OffsetParallax.y * aStepsInv;
			AuroaBorialisAtt *= 0.865;
		}
		
		return float4(1-exp(-AuroaBorialis), 1);
*/		
		float poleDensity = Pow5(abs(uv.y * 2.0 - 1.0));
		float mipBias = poleDensity * 9 + 1.0;
		//force the mips down near the poles!
		duvx *= mipBias;
		duvy *=	mipBias;
		
		float jacobi_falloff = saturate(Square(normal.y)) * 0.499 + 0.5;

		float speed = g_Time * g_MaterialSpecular.g;
		int iterations = 4;
		float2 flow_a = input.texCoord;
		float2 flow_b = input.texCoord;
		
		const float2 density = rcp(float2(32.0 * (1.0 - abs(input.texCoord.y - 0.5)) + 1.0, 32.0));

		float force_falloff = 1.0;
//		float actual_speed = g_Time * speed;// + textureNoise(input.posObj * rcp(g_Radius) * 2, 2, 2.0, g_Time * 0.01, 2.03, DPX * rcp(g_Radius), DPY * rcp(g_Radius)).x;
//		float variation = tex3Dgrad(NoiseSampler, float3(float2(input.texCoord.x + actual_speed * 0.01, input.texCoord.y) * 6, actual_speed * 0.05 - input.texCoord.y), DPX * rcp(g_Radius), DPY * rcp(g_Radius)).x;
//		variation -= tex3Dgrad(NoiseSampler, float3(float2(input.texCoord.x + actual_speed * 0.01, input.texCoord.y).yx * 5, -actual_speed * 0.05 + input.texCoord.x), DPX * rcp(g_Radius), DPY * rcp(g_Radius)).x;
//		return float4(variation.rrr, 1.0);
//		actual_speed += variation;
		float swap_a = frac(speed);
		float swap_b = frac(speed + 0.5);
		
		for(int i = 0; i < iterations; ++i)
		{

			flow_a -= (tex2Dgrad(TextureDisplacementSampler, flow_a, duvx, duvy).xy - 0.5) * density * force_falloff * swap_a;

			flow_b -= (tex2Dgrad(TextureDisplacementSampler, flow_b, duvx, duvy).xy - 0.5) * (density * force_falloff * swap_b);

			force_falloff *= jacobi_falloff;
		}

		float gas_blend = abs(swap_a * 2.0 - 1.0);
		
//		float4 gasSample;
//		= float4(lerp(flow_sample_a, flow_sample_b, gas_blend), 0.0, 1.0);
		
		float4 gasSample = mapGasGiant(lerp(tex2Dgrad(TextureColorSampler, flow_a, duvx, duvy), tex2Dgrad(TextureColorSampler, flow_b, duvx, duvy), gas_blend));
		
		flow_a += OffsetParallax * (gasSample.g - 0.5);
		flow_b += OffsetParallax * (gasSample.g - 0.5);
		
		gasSample = mapGasGiant(lerp(tex2Dgrad(TextureColorSampler, flow_a, duvx, duvy), tex2Dgrad(TextureColorSampler, flow_b, duvx, duvy), gas_blend));
		
//		return float4(gasSample.rgb, 1);
		float ao = Square(gasSample.b) * 0.95 + 0.05;
		
		float2 UVColor = float2(frac(speed * 0.01), saturate((input.texCoord.y - 0.5) + (gasSample.a - 0.5) * 0.2 + 0.5));
		
		float3 GasColor = SRGBToLinear(pow(tex2Dlod(TextureNormalSampler, float4(UVColor, 0.0, 0.0)).rgb, (float3)(1.5 - gasSample.r)));
//		GasColor = SRGBToLinear(pow(tex2Dlod(TextureNormalSampler, float4(UVColor, 0.0, 0.0)).rgb, 1.0));	
//		return float4(GasColor, 1);
		const float offset = 1.0 / 512.0;
		float4 blended_normals;
		blended_normals.x = mapGasGiant(lerp(tex2Dgrad(TextureColorSampler, flow_a - float2(offset, 0.0), duvx, duvy).g, tex2Dgrad(TextureColorSampler, flow_b + float2(offset, 0.0), duvx, duvy).g, gas_blend));
		blended_normals.y = mapGasGiant(lerp(tex2Dgrad(TextureColorSampler, flow_a + float2(offset, 0.0), duvx, duvy).g, tex2Dgrad(TextureColorSampler, flow_b - float2(offset, 0.0), duvx, duvy).g, gas_blend));
		blended_normals.z = mapGasGiant(lerp(tex2Dgrad(TextureColorSampler, flow_a + float2(0.0, offset), duvx, duvy).g, tex2Dgrad(TextureColorSampler, flow_b + float2(0.0, offset), duvx, duvy).g, gas_blend));
		blended_normals.w = mapGasGiant(lerp(tex2Dgrad(TextureColorSampler, flow_a - float2(0.0, offset), duvx, duvy).g, tex2Dgrad(TextureColorSampler, flow_b - float2(0.0, offset), duvx, duvy).g, gas_blend));

		blended_normals.xy = blended_normals.xz - blended_normals.yw;
		blended_normals.z = DeriveZ(blended_normals.xy);
		
		float3 normalGas = normalize(mul(blended_normals.xyz, TBN));	
		
		float3 Diffuse			= GasColor * SRGBToLinear(texCUBElod(EnvironmentIlluminationCubeSampler, float4(normalGas, 0))).rgb * ao;
/*
		float gasRoughness		= 0.75;
		float gasRoughnessMip	= 6.9282;
		float3 Specular			= (gasSample.a * 2.0) * SRGBToLinear(texCUBElod(TextureEnvironmentCubeSampler, float4(-(view - 2.0 * normalGas * dot(view, normalGas)), gasRoughnessMip))).rgb * ao;
		Specular 				*= AmbientDielectricBRDF(gasRoughness, saturate(dot(view, normalGas)));

		float3 light_sum 		= Specular + Diffuse;
*/		
		float3 light_sum 		= Diffuse;
		
		float cloudAbsorption = gasSample.g;
		float steps = 4;
		float ShadowBlend = (1 - saturate(NoL)) * steps + 1;
		float GasDensity = 2.0;
		if(NoL > -0.1)
		{
			float stepsInv = 1.0 / steps;	
			float2 ShadowStep = ShadowVector.xy * stepsInv;		

			float NoLinv = rcp(1.0 + NoL * steps);
			
			//steps = (steps - (steps - 1) * NoL);

			for(int g =0; g < ShadowBlend; g++)
			{
				flow_a += ShadowStep;
				flow_b += ShadowStep;
				cloudAbsorption -= mapGasGiant(lerp(tex2Dgrad(TextureColorSampler, flow_a, duvx, duvy).g, tex2Dgrad(TextureColorSampler, flow_b, duvx, duvy).g, gas_blend)) * saturate(ShadowBlend - g) * GasDensity;
			}
			cloudAbsorption = exp(min(0, cloudAbsorption * stepsInv));
		}
		else
		{
			cloudAbsorption = 0;
		}

		float3 atmosphereAbsorption	= lerp(0.075 + 0.025 * gasSample.g, Square(saturate(dot(-light, normalGas) + 0.25)), Square(cloudAbsorption)) * 200 * Reyleigh;
		atmosphereAbsorption = saturate(exp(-max((float3)0, float3(	rayleigh_phase_func(atmosphereAbsorption.r),	rayleigh_phase_func(atmosphereAbsorption.g),	rayleigh_phase_func(atmosphereAbsorption.b)))));
		NoL 				= saturate(dot(light, normalGas));
	
		light_sum += (GasColor * atmosphereAbsorption * lightColor) * (cloudAbsorption * NoL * ao);
		
		return LinearToSRGB(float4(1-exp(-light_sum), 1));
	}

	#ifdef SUPPORT_PARALLAX_DISPLACEMENT
		float p_search_steps 	= 32;//((32.0 - 24.0 * NoV));
		float p_steps_inv		= 1.0 / p_search_steps;
		float rayheight			= 1;
		float oldray			= 1;
		float2 offset			= 0;
		float oldtex			= 1;
		float texatray;
		float yintersect;
		uv						-= OffsetParallax * 0.5 * NoV;
		float2 offsetStep		= OffsetParallax * p_steps_inv * NoV;
		
		for (int i = 0; i < p_search_steps; ++i)
		{
		
			float texatray 			= tex2Dgrad(TextureDisplacementSampler, uv + offset, duvx, duvy).a;
		
			if (rayheight < texatray)
			{
				float xintersect	= (oldray - oldtex) + (texatray - rayheight);
				xintersect			= (texatray - rayheight) / xintersect;
				yintersect			= (oldray * (xintersect)) + (rayheight * (1 - xintersect));
				offset				-= (xintersect * offsetStep);
				break;
			}
		
			oldray					= rayheight;
			rayheight				-= p_steps_inv;
			offset 					+= offsetStep;
			oldtex 					= texatray;
		}
		
		uv += offset;
	#endif

	//ONLY if we need them
	#if 0
		duvx 					= ddx(uv);
		duvy 					= ddy(uv);
	#endif
	
	float3 shadow = 0.0;
	//TODO squeeze in lava flowmap in here?
	float4 sampleA 				= tex2Dgrad(TextureColorSampler, uv, duvx, duvy);
	float4 sampleB 				= tex2Dgrad(TextureDataSampler, uv, duvx, duvy);
	float4 sampleC 				= tex2Dgrad(TextureNormalSampler, uv, duvx, duvy);
	normal 						= normalize(mul(GetNormalDXT5(sampleC), TBN));
	sampleA.rgb 				= SRGBToLinear(sampleA.rgb);
	
	if(OceanMode)
	{
		float waterMask 			= saturate(sampleA.a * 4.0);
		float propertiesMask 		= saturate(waterMask * 32.0 - 31.0);

		float4 waterNgrad = 0;
		if(waterMask < 1.0)
		{
			float waterScale = 1.0 / 12.5;
			float waterSpeed = 12.5;
			float waterIntensity = 0.75;
			waterNgrad = snoise(input.posObj * waterScale + length(input.posObj * waterScale) + g_Time * waterSpeed * waterScale) * waterIntensity;
			normal = lerp(lerp(mul(normalize(input.normalObj + waterNgrad.rgb), (float3x3)g_World), normalSphere, Square(waterMask)), normal, propertiesMask);	
		}

		sampleA.a = saturate((sampleA.a - 0.25) * (1.0 / 0.75));
		if(waterMask < 1.0)
		{
			float fresnel = Pow5(1.0 - abs(dot(view, normal)));

			sampleA.rgb = pow(sampleA.rgb, 1.5 - fresnel * 1.4 + waterNgrad.a * 0.1);
			sampleA.a = lerp(0.23, sampleA.a, propertiesMask);//spec
			sampleB.w = lerp(0.1/* + fresnel * 0.16*/, sampleB.w, propertiesMask);//roughness
			sampleC.r = 0.5 * fresnel + 0.1;//subsurface
			sampleC.b = 0;//no metal water
		}	
	}
	
	NoL = saturate(dot(normal, light) * saturate(NoL + 0.2));
	
	#ifdef SUPPORT_PARALLAX_SHADOWS
		#ifdef SUPPORT_PARALLAX_DISPLACEMENT
			if(NoL > 0.0)
			{
				shadow = 1.0;
				float dist = 0;
				offset = 0;
				texatray = tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).a + 0.01;

				rayheight = texatray;
				float s_search_steps = 32.0 - 28.0 * saturate(ShadowVector.z);
				float lightstepsize = rcp(s_search_steps);
				float shadow_penumbra = 1.0;
				
				for(int j = 0; j < s_search_steps; j++)
				{
					if(rayheight < texatray)
					{
						shadow.r = 0;
						break;
					}
					else
					{
						shadow.r = min(shadow.r, (rayheight - texatray) * shadow_penumbra / dist);
					}

					oldray=rayheight;
					rayheight += ShadowVector.z * lightstepsize;

					offset += ShadowVector.xy * lightstepsize;
					oldtex = texatray;

					texatray = tex2Dgrad(TextureDisplacementSampler, uv + offset, duvx, duvy).a;
					dist += lightstepsize;
				}		
			}			
			shadow = shadow.r;
		#else
			shadow = 1;
		#endif			
	#else
		shadow = 1;
	#endif
	NoL = saturate(dot(normal, light));

	PBRProperties Properties;

	Properties.DiffuseColor = sampleA.rgb * (1.0 - sampleC.b);
	Properties.SpecularColor = sampleA.rgb * (sampleC.b) + 0.08 * sampleA.a * ((1.0 - sampleC.b));
	Properties.EmissiveColor = 0;
	Properties.Roughness = max(0.02, sampleB.a);
	Properties.RoughnessMip = sampleB.a * 8.0;
	Properties.AO = 1.0;
	Properties.SubsurfaceOpacity = sampleC.r;	
	
//	return g_Light0_Specular;
//	return ceil(g_MaterialSpecular.a * 4);
	float Smoke = 0;
	float shadowScalar 	= dot(input.lightDir, input.normal);
	float shadowTest = 0;
//	float SmokeSample = 0;
	
	float shadowSmoke = 1;

	if(VolcanoMode)
	{
		const float SmokeBaseDensity = 2.0;
		const float SmokePuffDensity = 16.0;
		float3 SmokeColor = 0;

		#if 1//def CLOUD_PARALLAX
			Smoke				= 1-exp(-VolcanoSmoke(input.texCoord, duvx, duvy, SmokeBaseDensity, SmokePuffDensity).x);
			float2 uvSmoke		= input.texCoord - OffsetParallax * (Smoke * 0.25) * NoV;// - OffsetParallax;
		#else
			float2 uvSmoke		= input.texCoord;
		#endif
		
		float2 SmokeBase	= VolcanoSmoke(uvSmoke, duvx, duvy, SmokeBaseDensity, SmokePuffDensity);
		Smoke				= SmokeBase.x;
		

		//shadow 				= saturate(shadow * exp(-Square(tex2Dgrad(TextureDisplacementSampler, uv + (ShadowVector.xy / ShadowVector.z) + SmokeRot.xy * (SmokeShadow * 2 + 0.25), duvx, duvy).g) * 10) + Smoke * saturate(shadowScalar * 10));			
//		shadowTest 			+= SmokeSample;*
		shadowSmoke = Smoke;// * saturate(shadowScalar);
		float SmokeSteps	= 15;

//		SmokeSteps 			-= shadowScalar * 15;
		float SmokeShadowBlend = (1 - saturate(shadowScalar)) * SmokeSteps + 1;
		float SmokeStepInv 	= (1.0 / SmokeSteps);
		float SmokeShadowIntensity = saturate(shadowScalar * 0.25 + 0.1);
//		return float4(ShadowVector.zzz, 1);
		float2 ShadowStep 		= ShadowVector.xy * SmokeStepInv;
		float2 uvSmokeShadow 	= uvSmoke - ShadowStep * 0.5;
		Smoke 					= 1 - exp(-Smoke);		
		Reyleigh *= 1-Smoke;
		Reyleigh += float3(0.53, 0.72, 0.95) * Smoke;
		
//		shadow *= 1-exp(-exp(-VolcanoSmoke(uv + ShadowVector.xy, duvx, duvy, SmokeBaseDensity, SmokePuffDensity)));	
		if(shadowScalar > -0.25)
		{
			shadow				= max(Smoke, shadow);

			for(int k = 0; k < SmokeShadowBlend; k++)
			{				
				uvSmokeShadow += ShadowStep;
				shadowSmoke -= VolcanoSmoke(uvSmokeShadow, duvx, duvy, SmokeBaseDensity, SmokePuffDensity).x * saturate(SmokeShadowBlend - k) * SmokeShadowIntensity;// * (1+k)* 0.01;			
			}
			shadowSmoke = exp(min(0, shadowSmoke * SmokeStepInv));
			//Reyleigh = lerp()shadow *= saturate(lerp(float3(0.118, 0.071, 0.031), float3(0.631, 0.541, 0.447), shadowSmoke), Smoke) * shadowSmoke;
			shadow *= shadowSmoke;
		}
		else
		{
			shadowSmoke = 0;
		}
//		return float4(shadowSmoke.rrr, 1);
		SmokeColor			= pow(float3(0.631, 0.541, 0.447) * Smoke, 2.2);

		Properties.AO		= Square(saturate(exp(-VolcanoSmoke(uv + OffsetParallax * NoV * 0.25, duvx, duvy, SmokeBaseDensity, SmokePuffDensity).x) + Smoke));// * 0.75 + 0.25;
//		return float4(Properties.AO.rrr, 1);
		//Properties.AO		= saturate(exp(-VolcanoSmoke(uv, duvx, duvy, SmokeBaseDensity, SmokePuffDensity)) + Smoke);
//		return float4(Properties.AO.rrr * shadow * saturate(lerp(NoL, saturate(shadowScalar + 0.25), Smoke)), 1);
		float SmokeRimGlow = Square(1-(1-Square(Smoke) * (1-Smoke))) * 2;
		Properties.EmissiveColor = float4(Square(LaveFlowTest(uv, duvx, duvy)) * 10.0 +	
														//tex2Dlod(TextureDataSampler, float4(uvSmoke + SmokeBase.g * float2(-0.0002, 0.0001), 0, 1.5)).rgb + 
														(Square(tex2Dlod(TextureDataSampler, float4(uvSmoke, 0, 2.5)).rgb) + 
														Square(tex2Dlod(TextureDataSampler, float4(uvSmoke, 0, 3.5)).rgb) +
														Square(tex2Dlod(TextureDataSampler, float4(uvSmoke, 0, 4.5)).rgb) + 
														Square(tex2Dlod(TextureDataSampler, float4(uvSmoke, 0, 5.5)).rgb)) * 0.25, 1);
		Properties.EmissiveColor *= Square(1-Smoke); 
		Properties.DiffuseColor *= (1-Smoke);
		Properties.DiffuseColor += SmokeColor * Smoke;
		Properties.Roughness 	= max(Properties.Roughness, Smoke);
		normal					= normalize(normal * (1 - Smoke) + normalSphere * Smoke);
	}

	float3 atmosphereAbsorption	= lerp(0.075 + 0.025 * Smoke, Square(saturate(dot(-light, normalSphere) + 0.25)), Square(shadowSmoke)) * 200 * Reyleigh;
	atmosphereAbsorption = saturate(exp(-max((float3)0, float3(	rayleigh_phase_func(atmosphereAbsorption.r),	rayleigh_phase_func(atmosphereAbsorption.g),	rayleigh_phase_func(atmosphereAbsorption.b)))));
	shadow *= atmosphereAbsorption;
	//tex2Dlod(TextureDataSampler, float4(uvSmoke + SmokeBase.g * float2(-0.0005, -0.00025), 0, 2.5)).rgb
//	return float4(LaveFlowTest(sampleC.yw, uv, duvx, duvy).rrr * NoL, 1);

//	return float4(Smoke.rrr, 1);
//	float3 PSmoke = input.pos + normalSphere * (SmokeRot * Smoke * 0.1 + Smoke);
	//we can't branch on derivatives
//	normal = normalize(lerp(normal, normalize(cross(ddx(PSmoke), ddy(PSmoke))) - normalize(cross(ddx(input.pos), ddy(input.pos))) + normalSphere, Smoke));
//	float3 TSmoke = ddx(PSmoke);
	//float3 BSmoke = ddy(PSmoke);
//	return float4(normalize(lerp(normal, normalize(cross(ddx(PSmoke), ddy(PSmoke))) - normalize(cross(ddx(input.pos), ddy(input.pos))) + normalSphere, Smoke)), 1);
//	return float4(shadow.rrr, 1);
	//Smoke *= 

//	return float4(SmokeColor, 1);


//	return LinearToSRGB(float4(NoL.rrr * shadow * 0.5 * Properties.DiffuseColor, 1));	

	float4 finalColor = float4((float3)0, 1.0);		
	float3 specular = 0;
	float3 diffuse = 0;
	float3 emissive = 0;

	float noiseScale = 10.f;

	float3 background_col = diffuse;
 	float3 atmoView = view;//normalize(lerp(normalSphere, view, 0.2));

	
	float NoLsphere = saturate(dot(normalSphere, light) * 0.75 + 0.25);
	float rotatationTime = g_Time / 800;
	float indexTime = g_Time/70;


	float4 cloudsShadowColor = 1;
	float cloudShadow = 1;
	float cloudSample = 0;
	
#if 0
	float2 UVcloud = float2(input.texCoord.x + rotatationTime, input.texCoord.y);
	float2 UVstep = UVcloud + OffsetParallax;// + OffsetParallax;
	UVstep += (tex2D(CloudLayerSampler, UVstep).b - 0.5) * OffsetParallax * 0.25;
	

	float cloudDensity = 0.2;
	float cloudDensityInv = 1.0 / cloudDensity;
	float cloudMip = sqrt(cloudDensityInv);
	cloudSample 				= mapClouds(tex2D(CloudLayerSampler, UVstep).b);
	Properties.DiffuseColor		= lerp(Properties.DiffuseColor, g_MaterialAmbient.rgb, cloudSample);
	Properties.SpecularColor	= lerp(Properties.SpecularColor, (float3)0.08, cloudSample);
	Properties.Roughness		= lerp(Properties.Roughness, 1.0, cloudSample);
	
	cloudShadow 				= mapClouds(tex2Dbias(CloudLayerSampler, float4(UVcloud, 0, cloudMip)).b);
	
	cloudShadow 				= exp(-cloudShadow);
	float cloudShadowRoughness	= cloudShadow;
	cloudShadow					= max(cloudShadow, cloudSample);
#endif	
	float3 DiffuseSample		= SRGBToLinear(texCUBE(EnvironmentIlluminationCubeSampler, lerp(normal, normalSphere, cloudSample))).rgb;
		
	
	float3 reflection			= -(view - 2.0 * normal * NoV);
	float3 ReflectionSample 	= SRGBToLinear(texCUBElod(TextureEnvironmentCubeSampler, float4(reflection, max(cloudShadow * 6.0, Properties.RoughnessMip)))).rgb;

	AmbientBRDF(diffuse, specular, saturate(abs(NoV)), Properties, ReflectionSample, DiffuseSample, true);
	
	diffuse 					*= cloudShadow;
	specular 					*= cloudShadow;
	

#if 0





		float cloudAbsorption = 0;
		if(NoLsphere > 0)
		{
			cloudShadow = max(exp(-mapClouds(tex2Dbias(CloudLayerSampler, float4(UVcloud + ShadowVector.xy, 0.0, cloudMip)).b)), cloudSample);

			float steps = 32.0;
			float stepsInv = 1.0 / steps;		
			
			for(int i =0; i < steps - (steps * NoLsphere); i++)
			{
				UVstep += ShadowVector.xy * stepsInv;
				cloudAbsorption = max(cloudAbsorption, mapClouds(tex2Dbias(CloudLayerSampler, float4(UVstep, 0.0, i * stepsInv * cloudMip)).b) * (1.0 - stepsInv * i));
			}
			cloudAbsorption = exp(-cloudAbsorption);

			cloudsShadowColor = float4(lerp(float3(0.025,0.03,0.035), lerp((float3)1.0, float3(1.0,0.95,0.8), cloudSample), cloudAbsorption), cloudAbsorption) * cloudShadow;
		}

		float cityMask = saturate(-4.0 * shadowScalar);
		if(cityMask > 0.0)
		{	
			float cloudEmissiveRim = 0.25;
			float cloudEmissiveAbsorption = exp(-cloudSample * cloudDensityInv + cloudEmissiveRim) * cityMask;
			float cityMipBias = max(0.0, 5.0 - exp(-cloudSample * cloudDensityInv) * 7.0);
			float3 sampleLight = (1.0 - tex2Dbias(TextureDisplacementSampler, float4(uv * 8.0, 0.0, cityMipBias)).rgb);
			emissive += SRGBToLinear(tex2Dbias(TextureDataSampler, float4(uv, 0.0, cityMipBias)) * sampleLight).rgb * cloudEmissiveAbsorption;
			//TODO cleanup
			emissive += SRGBToLinear(tex2Dlod(TextureDataSampler, float4(input.texCoord, 0.0, 4.5))).rgb * cloudEmissiveAbsorption * (0.5 - abs(dot(normalSphere, view)) * 0.4);
		}
		
		cloudsShadowColor *= saturate(shadowScalar);
		

	Properties.Roughness = max(1.0 - cloudShadow, Properties.Roughness);//scater specular light!
#endif	
	
	emissive += Properties.EmissiveColor.rgb;
	
	float3 subsurfaceScatter = 0;
//	if(shadowScalar > 0.0)
//	{
		PBRDots dots 				= GetDots(normal, view, light);
//		dots.NoL					= lerp(dots.NoL, NoLsphere, cloudSample);
		
		specular 				+= SpecularGGX(Properties, dots) * cloudsShadowColor.a * dots.NoL * shadow * lightColor * Properties.AO;
		diffuse 				+= DiffuseBurley(Properties, dots) * cloudsShadowColor.rgb * dots.NoL * shadow * lightColor * Properties.AO;
		
//	}
	#ifdef SUPPORT_SUBSURFACE
	Properties.SubsurfaceOpacity = max(Properties.SubsurfaceOpacity, cloudSample * 0.25);
	if(Properties.SubsurfaceOpacity > 0)
	{
		float3 subsurfaceColor 		= SRGBToLinear(tex2Dbias(TextureColorSampler, float4(uv, 0.0, 2.0)).rgb) * Properties.SubsurfaceOpacity * (1.0 - cloudSample) + g_MaterialAmbient * cloudSample;	

		float InScatter				= pow(saturate(dot(light, -view)), 12) * lerp(3, .1f, Properties.SubsurfaceOpacity);
		float NormalContribution	= saturate(dot(normal, normalize(view + light)) * Properties.SubsurfaceOpacity + 1.0 - Properties.SubsurfaceOpacity);
		float BackScatter		 	= Properties.AO * NormalContribution * 0.1591549431;
		subsurfaceScatter 			= lerp(BackScatter, 1, InScatter) * subsurfaceColor * saturate(shadowScalar + Properties.SubsurfaceOpacity);
		subsurfaceScatter			*= cloudsShadowColor.rgb;
		subsurfaceScatter 			*= lightColor;
	
		InScatter					= pow(NoV, 12) * lerp(3, .1f, Properties.SubsurfaceOpacity);
		NormalContribution			= saturate(dot(normal, reflection) * Properties.SubsurfaceOpacity + 1.0 - Properties.SubsurfaceOpacity);
		BackScatter		 			= Properties.AO * NormalContribution * 0.1591549431;
		subsurfaceScatter	 		+= subsurfaceColor * lerp(BackScatter, 1, InScatter) * (SRGBToLinear(tex2Dbias(TextureColorSampler, float4(uv, 0.0, 6.0)).rgb * Square(NoLsphere) * (1.0 - cloudSample) + DiffuseSample));
	}
	#endif

	return LinearToSRGB(float4(1.0 - exp(-(diffuse + specular + emissive + subsurfaceScatter)), 1.0));

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

		float3 sun_dir;
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
		Ray light_ray = make_ray(ray.origin, sky.sun_dir);
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
						float3 sun_dir, 
						float3 sun_color, 
						float3 view_dir, 
						float planet_rad, 
						float atmo_rad, 
						float3 planet_pos, 
						float3 rayleigh_beta, 
						float3 mie_beta,
						float scatter_rayleight,
						float scatter_mie)
	{
		Sky_PBR sky;

		sky.hR = (8000.0 * 0.00002) * scatter_rayleight;
		sky.hM = (1200.0 * 0.00002) * scatter_mie;

		sky.inv_hR = 1.0 / sky.hR;
		sky.inv_hM = 1.0 / sky.hM;

		sky.g = 0.8;

		sky.earth = make_earth(planet_pos, planet_rad, atmo_rad);

		sky.transmittance = 1;
		sky.optical_depthR = 0;
		sky.optical_depthM = 0;

		sky.sumR = 0;
		sky.sumM = 0;
		sky.betaR = rayleigh_beta * 1e-6;
		sky.betaM = (float3)(mie_beta * 1e-6);

		sky.sun_dir = sun_dir;

		sky.space_sumR = 0;
		sky.space_sumM = 0;

		sky.VL = dot(view_dir, sky.sun_dir);

		sky.phaseR = rayleigh_phase_func(sky.VL);
		sky.phaseM = henyey_greenstein_phase_func(sky.VL, sky.g);

		Ray view_ray = make_ray((float3)0.0, view_dir);

		float atmopshere_thickness = sky.earth.atmosphere_radius - sky.earth.earth_radius;

		float t0 = 0;
		float t1 = 0; 

		if(isect_sphere(view_ray, sky.earth.center, sky.earth.atmosphere_radius, t0, t1)){

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

				float3 wp = view_dir * dist;

				Ray ray = make_ray(wp, view_dir);
				get_incident_light_space(sky, ray, step_dist, atmopshere_thickness, blue_noise);

				avg_space_light += texCUBE(EnvironmentIlluminationCubeSampler, float4(normalize(wp - sky.earth.center), 0)).rgb;

				prev_dist = dist;
			}

			avg_space_light *= INV_NR_SAMPLE_STEPS;

			float3 rayleigh_color = (sky.sumR * sky.phaseR * sky.betaR);
			float3 mie_value = (sky.sumM * sky.phaseM * sky.betaM);

			float3 ambient_colorR = (sky.space_sumR * sky.betaR);
			float3 ambient_colorM = (sky.space_sumM * sky.betaM);

			atmosphere.rgb = (rayleigh_color + mie_value) * sun_color + (ambient_colorR + ambient_colorM) * avg_space_light;
			atmosphere.a = saturate(1.0 - sky.transmittance);
			//atmosphere.a = exp(-sky.transmittance);
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
    float3 positionInWorldSpace = mul(float4(iPosition, 1.f), g_World).xyz;
	float inflateScale = distance(positionInWorldSpace - (mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz), positionInWorldSpace) * (1.0 / 16384.0) + 1;
	float atmosphereThickness = 1.0 + frac(g_MaterialGlossiness) * inflateScale;// * 5.0;
	//Final Position
	
	o.Position = mul(float4(iPosition * atmosphereThickness, 1.0f), g_WorldViewProjection);
//	if(GasGiantMode)
//		o.Position /= 1-float(GasGiantMode);//NaN cull, compiler wont allow us to just flat out divide by 0. In effect, the pixel shader will never execute
	
	//Texture Coordinates
    o.TexCoord0 = iTexCoord; 
	
    //Calculate  Normal       
    o.Normal = normalize(mul(iNormal, (float3x3)g_World));
    

	o.Pos = positionInWorldSpace;// * ATMOSPHERE_SCALE;//(1.0 / ATMOSPHERE_SCALE);
	o.PlanetPos = (positionInWorldSpace - (mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz) * atmosphereThickness) / g_Radius;
    //Calculate Light
	o.Light = normalize(((positionInWorldSpace - g_Light0_Position) * (1 + 0.1)) / g_Radius);
	
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
	//Light and Normal - renormalized because linear interpolation screws it up
	float3 light = normalize(i.Light);
	float3 normal = normalize(i.Normal);
	float3 view = normalize(i.View);
	float3 lightColor 		= SRGBToLinear(g_Light0_DiffuseLite.rgb) * PI;		
	
#ifndef DEBUG_MAKE_PLANET_SPHEREPROBE
//	const bool GasGiantMode = round(g_MaterialSpecular.a * 4) != 2;//planet that are lave based, no citylight
	oColor0 = 0;
	
//	if(GasGiantMode)
//	{
//		#define BALANCE 0.001;
//		#define .0;//1000.0;
		float atmosphereThickness = 1.0 + frac(g_MaterialGlossiness);
		float planet_tweak_scale = floor(g_MaterialGlossiness) * 0.1;//earth scale
		
		float3 sun_dir 			= normalize(i.Light * planet_tweak_scale);
//		float3 normal = normalize(i.Normal);
		float3 view_dir 		= normalize(i.View * planet_tweak_scale);
		float planet_rad 		= planet_tweak_scale;
		float atmo_rad 			= planet_tweak_scale * atmosphereThickness;//1.0018835;//earth scale, unfortunately we don't have the precision for that
		float3 rayleigh_beta	= g_MaterialDiffuse.rgb * 50.0;//float3(5.5, 12.0, 30.0);
		float mie_beta			= g_MaterialDiffuse.a * 50.0;  //10.0;
		float scatter_rayleight	= 250.0;
		float scatter_mie		= 250.0;
		float3 sun_color		= 100.0 * lightColor;
		float3 planet_pos		= i.PlanetPos * planet_tweak_scale;//yup remap for consistency
		GetAtmosphere(	oColor0, 
						sun_dir, 
						sun_color, 
						view_dir, 
						planet_rad, 
						atmo_rad, 
						planet_pos, 
						rayleigh_beta, 
						mie_beta,
						scatter_rayleight,
						scatter_mie);
//		oColor0 += 0.1;
//		oColor0.a = 1;//oColor0.a * 0.75 + 0.15;
//		oColor0 = float4(i.PlanetPos, 1.0);
//		oColor0.rgb += tex2Dlod(TextureDataSampler, float4(i.TexCoord0, 0, 4.5)).rgb;//works follow up!
//		oColor0 = LinearToSRGB(oColor0);
//	}
#else
	oColor0 = 0;//float4(g_MaterialDiffuse.rgb * 4.0, 1);//float4(frac(distance(mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz, i.Pos - mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz).xxx / 1000.0), 1);
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
/*		
		AlphaBlendEnable = TRUE;
		SrcBlend = srcalpha;
		DestBlend = InvSrcAlpha;
*/

		//additive
		ZEnable 			= true;
		ZWriteEnable 		= false;		
		AlphaTestEnable 	= TRUE;		
		AlphaBlendEnable 	= TRUE;
		SrcBlend 			= SRCALPHA;
		//DestBlend 			= INVSRCALPHA;
		DestBlend 			= DESTALPHA;
		
    }
}