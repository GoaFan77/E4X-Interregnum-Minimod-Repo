#define SF (1.0/float(0xffffffffU))
#define PI cos(0.0)
#define INVPI 1.0 / PI
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



float4 g_Light_AmbientLite: Ambient;
float4 g_Light_AmbientDark;

float3 g_Light0_Position: Position = float3( 0.f, 0.f, 0.f );
float4 g_Light0_DiffuseLite: Diffuse = float4( 1.f, 1.f, 1.f, 1.f );
float4 g_Light0_DiffuseDark;
float4 g_Light0_Specular;

float4 g_MaterialAmbient:Ambient;
float4 g_MaterialDiffuse:Diffuse;
float g_MaterialGlossiness: Glossiness;
float4 g_GlowColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
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
	void CotangentDerivativeBase(float3 Pos, float3 N, float3 L, float3 V, float2 UV, inout float2 OffsetShadow, inout float2 OffsetParallax, inout float3x3 TBN)
	{
		float2 DUVX 		= ddx(UV);
		float2 DUVY 		= ddy(UV);
		float3 DPX 			= ddx(Pos);
		float3 DPY 			= ddy(Pos);
		float3 DPXPerp 		= cross(N, DPX);		
		float3 DPYPerp 		= cross(DPY, N);		
		float3 Tangent 		= DPYPerp * DUVX.x + DPXPerp * DUVY.x;
		float3 Cotangent 	= DPYPerp * DUVX.y + DPXPerp * DUVY.y;
		float InvMax		= pow(max(dot(Tangent, Tangent), dot(Cotangent, Cotangent)), -0.5);
		Tangent				*= InvMax;
		Cotangent 			*= InvMax;
		
		//compack parallax shadow vector, almost freeriding on derivative math, replace L to get parallax dir and divide by NoV
		
		float3 Lt = mul(float3x3(Tangent, Cotangent, N), L);
		float3 Vt = mul(float3x3(Tangent, Cotangent, N), V);

		Lt = normalize(Lt);
		Vt = normalize(Vt);
		
		
		
		OffsetShadow = (Lt.xy / max(0.00001, sqrt(Lt.z * 0.5 + 0.5))) * 0.01;
		OffsetParallax = (Vt.xy / max(0.00001, sqrt(Vt.z * 0.5 + 0.5))) * 0.01;	
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
float4 RenderScenePS(VsSceneOutput input) : COLOR0
{ 
//	return float4(g_MaterialAmbient.rgb, 1.0);
	float2 OffsetShadow, OffsetParallax;
	float3x3 TBN;
	float3 normal = normalize(input.normal);
	float3 normalSphere = normal;	
	float3 view = normalize(-input.pos);
	float3 light = normalize(input.lightDir);	
	CotangentDerivativeBase(input.pos, normal, light, view, input.texCoord, OffsetShadow, OffsetParallax, TBN);
	
	float2 uv = input.texCoord;
	
	float NoV = abs(dot(view, normal));
	float p_linear_search_steps = 16.0 - 14.0 * NoV;
	float p_binary_search_steps = 8.0;
	float p_search_height = rcp(p_linear_search_steps);
	float p_scale = p_binary_search_steps * rcp(10.0);

	OffsetParallax *= p_scale;
	uv -= OffsetParallax;

	float sum = 1.0;
	float depth = 1.0;
	
	float sample_p = 0;// = tex2D(TextureDisplacementSampler, uv).a;
	for (int i = 0; i < p_linear_search_steps; ++i)
	{
		sample_p = 1.0 - tex2D(TextureDisplacementSampler, uv).a;

		if(sample_p > sum)
			break;
		sum -= p_search_height;
		uv += OffsetParallax * p_search_height;

		if(i > 32)
			break;
	}

	 //  binary search refine
	float binary_search_height = p_search_height;
  	for (int j = 0; j < p_binary_search_steps; ++j)
  	{
  		sample_p = 1.0 - tex2D(TextureDisplacementSampler, uv).a;

  		float binary_lookup = binary_search_height * (step(sample_p, sum) - 0.5);
  		uv += OffsetParallax * binary_lookup;
  		sum -= binary_lookup;
  		binary_search_height *= 0.5;
  	}	
	
//	return float4(frac(uv * 128.0), 0.0, 1.0);
	float2 duvx = ddx(uv);
	float2 duvy = ddy(uv);
	float4 sampleA = tex2Dgrad(TextureColorSampler, uv, duvx, duvy);
	float4 sampleB = tex2Dgrad(TextureDataSampler, uv, duvx, duvy);
	float4 sampleC = tex2Dgrad(TextureNormalSampler, uv, duvx, duvy);
	float3 normalLand = normalize(mul(GetNormalDXT5(sampleC), TBN));
//	return float4(normalLand, 1.0);
	sampleA.rgb = SRGBToLinear(sampleA.rgb);
	PBRProperties Properties;

	//derivatives are used so they can't branch
	float waterMask = saturate(sampleA.a * 4.0);
	float propertiesMask = saturate(waterMask * 32.0 - 31.0);

	normal = lerp(normal, normalLand, propertiesMask);	

	float4 waterNgrad = 0;
	if(waterMask < 1.0)
	{
		float waterScale = 1.0 / 25.0;
		float waterSpeed = 12.5;
		float waterIntensity = 0.75;
		waterNgrad = snoise(input.posObj * waterScale + length(input.posObj * waterScale) + g_Time * waterSpeed * waterScale) * waterIntensity;
		normal = lerp(mul(normalize(input.normalObj + waterNgrad.rgb), (float3x3)g_World), normal, Square(waterMask));
	}
//	return float4(normal, 1.0);
	sampleA.a = saturate((sampleA.a - 0.25) * (1.0 / 0.75));
	if(waterMask < 1.0)
	{
		float fresnel = Pow5(1.0 - abs(dot(view, normal)));

//		propertiesMask *= propertiesMask * (3.0 - 2.0 * propertiesMask);
		sampleA.rgb = pow(sampleA.rgb, 1.5 - fresnel * 1.4 + waterNgrad.a * 0.1);
		sampleA.a = lerp(0.23, sampleA.a, propertiesMask);//spec
		sampleB.w = lerp(0.1/* + fresnel * 0.16*/, sampleB.w, propertiesMask);//roughness
		sampleC.r = 0.5 * fresnel + 0.1;//subsurface
		sampleC.b = 0;//no metal water
	}

	Properties.DiffuseColor = sampleA.rgb * (1.0 - sampleC.b);
	Properties.SpecularColor = sampleA.rgb * (sampleC.b) + 0.08 * sampleA.a * ((1.0 - sampleC.b));
	Properties.EmissiveColor = 0;
	Properties.Roughness = max(0.02, sampleB.a);
	Properties.RoughnessMip = sampleB.a * 8.0;
	Properties.AO = 1.0;
	Properties.SubsurfaceOpacity = sampleC.r;


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

	float2 UVcloud = float2(input.texCoord.x + rotatationTime, input.texCoord.y);
	float2 UVstep = UVcloud + OffsetParallax;// + OffsetParallax;
	UVstep += (tex2D(CloudLayerSampler, UVstep).g - 0.5) * OffsetParallax * 0.25;
	
	float cloudDensity = 0.2;
	float cloudDensityInv = 1.0 / cloudDensity;
	float cloudMip = sqrt(cloudDensityInv);
	float cloudSample 			= mapClouds(tex2D(CloudLayerSampler, UVstep).g);
	Properties.DiffuseColor		= lerp(Properties.DiffuseColor, g_MaterialAmbient.rgb, cloudSample);
	Properties.SpecularColor	= lerp(Properties.SpecularColor, (float3)0.08, cloudSample);
	Properties.Roughness		= lerp(Properties.Roughness, 1.0, cloudSample);
	
	float cloudShadow 			= mapClouds(tex2Dbias(CloudLayerSampler, float4(UVcloud, 0, cloudMip)).g);
	
	cloudShadow 				= exp(-cloudShadow);
	float cloudShadowRoughness	= cloudShadow;
	cloudShadow					= max(cloudShadow, cloudSample);
	
	float3 DiffuseSample		= SRGBToLinear(texCUBE(EnvironmentIlluminationCubeSampler, lerp(normal, normalSphere, cloudSample))).rgb;
		
	diffuse 					+= Properties.DiffuseColor * DiffuseSample * cloudShadow;	
	
	float3 reflection			= -(view - 2.0 * normal * NoV);
	float3 ReflectionSample 	= SRGBToLinear(texCUBElod(TextureEnvironmentCubeSampler, float4(reflection, max(cloudShadow * 6.0, Properties.RoughnessMip)))).rgb;
	
	specular 					+= ReflectionSample * AmbientBRDF(saturate(abs(NoV)), Properties) * cloudShadow;



	float cloudAbsorption = 0;
	float4 cloudsShadowColor = 0;
	if(NoLsphere > 0)
	{
		cloudShadow = max(exp(-mapClouds(tex2Dbias(CloudLayerSampler, float4(UVcloud + OffsetShadow, 0.0, cloudMip)).g)), cloudSample);

		float steps = 32.0;
		float stepsInv = 1.0 / steps;		
		
		for(int i =0; i < steps - (steps * NoLsphere); i++)
		{
			UVstep += OffsetShadow * stepsInv;
			cloudAbsorption = max(cloudAbsorption, mapClouds(tex2Dbias(CloudLayerSampler, float4(UVstep, 0.0, i * stepsInv * cloudMip)).g) * (1.0 - stepsInv * i));
		}
		cloudAbsorption = exp(-cloudAbsorption);

		cloudsShadowColor = float4(lerp(float3(0.025,0.03,0.035), lerp((float3)1.0, float3(1.0,0.95,0.8), cloudSample), cloudAbsorption), cloudAbsorption) * cloudShadow;
	}
	float shadowScalar 	= dot(input.lightDir, input.normal);
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
	float3 subsurfaceScatter = 0;

	if(shadowScalar > 0.0)
	{
		PBRDots dots 				= GetDots(normal, view, light);
		dots.NoL					= lerp(dots.NoL, NoLsphere, cloudSample);
		
		specular 				+= SpecularGGX(Properties, dots) * cloudsShadowColor.a * dots.NoL;
		diffuse 				+= DiffuseBurley(Properties, dots) * cloudsShadowColor.rgb * dots.NoL;
		
	}
	Properties.SubsurfaceOpacity = max(Properties.SubsurfaceOpacity, cloudSample * 0.25);
	if(Properties.SubsurfaceOpacity > 0)
	{
		float3 subsurfaceColor 		= SRGBToLinear(tex2Dbias(TextureColorSampler, float4(uv, 0.0, 2.0)).rgb) * Properties.SubsurfaceOpacity * (1.0 - cloudSample) + g_MaterialAmbient * cloudSample;	

		float InScatter				= pow(saturate(dot(light, -view)), 12) * lerp(3, .1f, Properties.SubsurfaceOpacity);
		float NormalContribution	= saturate(dot(normal, normalize(view + light)) * Properties.SubsurfaceOpacity + 1.0 - Properties.SubsurfaceOpacity);
		float BackScatter		 	= Properties.AO * NormalContribution * 0.1591549431;
		subsurfaceScatter 			= lerp(BackScatter, 1, InScatter) * subsurfaceColor * saturate(shadowScalar + Properties.SubsurfaceOpacity);
		subsurfaceScatter			*= cloudsShadowColor.rgb;
	
		InScatter					= pow(NoV, 12) * lerp(3, .1f, Properties.SubsurfaceOpacity);
		NormalContribution			= saturate(dot(normal, reflection) * Properties.SubsurfaceOpacity + 1.0 - Properties.SubsurfaceOpacity);
		BackScatter		 			= Properties.AO * NormalContribution * 0.1591549431;
		subsurfaceScatter	 		+= subsurfaceColor * lerp(BackScatter, 1, InScatter) * (SRGBToLinear(tex2Dbias(TextureColorSampler, float4(uv, 0.0, 6.0)).rgb * Square(NoLsphere) * (1.0 - cloudSample) + DiffuseSample));
	}
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

		if (d2 > radius2) return false;

		float thc = sqrt(radius2 - d2);
		t0 = tca - thc;
		t1 = tca + thc;

		return true;
	}

	Earth make_earth(float3 center, float earth_radius, float atmosphere_radius){
		Earth s;
		s.center = center;
		s.earth_radius = earth_radius;
		s.atmosphere_radius = atmosphere_radius;
		return s;
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

	void get_sun_light_space(
		in Sky_PBR sky,
		in Ray ray,
		in float blue_noise,
		inout float optical_depthR,
		inout float optical_depthM)
	{

		float inner_sphere0 = 0;
		float inner_sphere1 = 0;
		float outer_sphere0 = 0;
		float outer_sphere1 = 0;
		isect_sphere(ray, sky.earth.center, sky.earth.earth_radius, inner_sphere0, inner_sphere1);
		isect_sphere(ray, sky.earth.center, sky.earth.atmosphere_radius + 0.00001, outer_sphere0, outer_sphere1);

		float march_step = outer_sphere1;
		if(inner_sphere0 > 0){
			march_step = min(inner_sphere0 + 1000, outer_sphere1);
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

		float shadow_term = 1;//calculate_celesital_shadow(ray.origin, sky.sun_dir, 1.0, MISSION_PLANET_INDEX, atmosphere_size_scale_mult_spaceplanet);

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

		sky.g = 0.5;

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

		Ray view_ray = make_ray(float3(0,0,0), view_dir);

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

				avg_space_light += texCUBE(EnvironmentIlluminationCubeSampler, float4(normalize(wp - sky.earth.center), 0)).rgb * PI;

				prev_dist = dist;
			}

			avg_space_light *= INV_NR_SAMPLE_STEPS;

			float3 rayleigh_color = (sky.sumR * sky.phaseR * sky.betaR);
			float3 mie_value = (sky.sumM * sky.phaseM * sky.betaM);

			float3 ambient_colorR = (sky.space_sumR * sky.betaR);
			float3 ambient_colorM = (sky.space_sumM * sky.betaM);

			atmosphere.rgb = (rayleigh_color + mie_value) * sun_color + (ambient_colorR + ambient_colorM) * avg_space_light;
			atmosphere.a = saturate(1.0 - sky.transmittance);
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
	float PercentHeight : COLOR0;
};

VsCloudsOutput 
RenderCloudVertex(float thicknessModifier, float3 iPosition, float3 iNormal, float2 iTexCoord)
{
	VsCloudsOutput o;  
	
	//Final Position
	o.Position = mul(float4(iPosition * /*thicknessModifier*/(1 + frac(g_MaterialGlossiness)), 1.0f), g_WorldViewProjection);
	
	//Texture Coordinates
    o.TexCoord0 = iTexCoord; 
	
    //Calculate  Normal       
    o.Normal = normalize(mul(iNormal, (float3x3)g_World));
    
    //Position
    float3 positionInWorldSpace = mul(float4(iPosition, 1.f), g_World).xyz;
    o.Pos = positionInWorldSpace * (1 + frac(g_MaterialGlossiness));// * ATMOSPHERE_SCALE;//(1.0 / ATMOSPHERE_SCALE);
    //Calculate Light
	o.Light = -normalize(g_Light0_Position - positionInWorldSpace);
	
	//Calculate ViewVector
	o.View = normalize(-positionInWorldSpace);
	
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
	
#if 1
	oColor0 = 0;
	
	float planet_tweak_scale = floor(g_MaterialGlossiness);//earth scale
	
	float3 sun_dir 			= normalize(i.Light);
//	float3 normal = normalize(i.Normal);
	float3 view_dir 		= normalize(i.View);
	float planet_rad 		= planet_tweak_scale;
	float atmo_rad 			= planet_tweak_scale * (1.0 + frac(g_MaterialGlossiness) * 0.1);//1.0018835;//earth scale, unfortunately we don't have the precision for that
	float3 rayleigh_beta	= g_MaterialDiffuse.rgb * 50.0;//float3(5.5, 12.0, 30.0);
	float mie_beta			= g_MaterialDiffuse.a * 50.0;  //10.0;
	float scatter_rayleight	= 100.0;
	float scatter_mie		= 100.0;
	float3 sun_color		= float3(1.0, 1.0, 1.0) * 5;
	float3 planet_pos		= ((i.Pos - mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz * (1 + frac(g_MaterialGlossiness))) / (g_Radius)) * planet_tweak_scale;//yup remap for consistency
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
//	oColor0 += 0.1;
//	oColor0.a = 1;

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
        VertexShader = compile vs_1_1 RenderSceneVS();
        PixelShader = compile ps_3_0 RenderScenePS();
        
		AlphaTestEnable = FALSE;
        AlphaBlendEnable = TRUE;
		SrcBlend = ONE;
		DestBlend = ZERO;    	
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
		ZWriteEnable = true;
		AlphaTestEnable = true;
		AlphaBlendEnable = true;
		SrcBlend = InvDestColor;
		DestBlend = ONE; 
		
    }
}