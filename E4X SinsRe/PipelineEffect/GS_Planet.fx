#define SF (1.0/float(0xffffffffU))
#define PI cos(0.0)
#define INVPI 1.0 / PI
float4x4		g_World					: World;
float4x4		g_WorldViewProjection	: WorldViewProjection;

texture	g_TextureDiffuse0 : Diffuse;
texture	g_TextureSelfIllumination;
texture g_TextureDiffuse2 : Diffuse;
texture g_TextureNormal : Normal;
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
float g_MaterialGlossiness = 50;
float4 g_GlowColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
float4 g_CloudColor = float4(1.0f, 1.0f, 1.0f, 1.0f);

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
	float3 CotangentDerivativeMap(float3 Pos, float3 N, float3 L, float3 V, float3 NMap, float2 UV, inout float2 OffsetShadow, inout float2 OffsetParallax)
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
		
		return normalize(mul(NMap, float3x3(Tangent, Cotangent, N)));	
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
		Nxy.y = 1.0 - Nxy.y;
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


// scattering coeffs
#define RAY_BETA float3(5.5e-6, 13.0e-6, 22.4e-6) /* rayleigh, affects the color of the sky */
#define MIE_BETA (float3)(21e-6) /* mie, affects the color of the blob around the sun */
#define AMBIENT_BETA (float3)(0)//(float3)(21e-18) /* ambient, affects the scattering color when there is no lighting from the sun */
#define ABSORPTION_BETA float3(2.04e-5, 4.97e-5, 1.95e-6) /* what color gets absorbed by the atmosphere (Due to things like ozone) */
#define ATMOSPHERE_INTENSITY 40.0
#define G 0.7 /* mie scattering direction, or how big the blob around the sun is */
// and the heights (how far to go up before the scattering has no effect)
#define HEIGHT_RAY 10000//8000 /* rayleigh height */
#define HEIGHT_MIE 1300//1200 /* and mie */
#define HEIGHT_ABSORPTION 30000 /* at what height the absorption is at it's maximum */
#define ABSORPTION_FALLOFF 3000 /* how much the absorption decreases the further away it gets from the maximum height */
#define ATMOSPHERE_SCALE 0.2 /* 0.075 is equal to earth atmosphere ratio*/
#define BASE_SCALE 40 /* Sins scale is way too small*/
// and the steps (more looks better, but is slower)
#define PRIMARY_STEPS 64 /* primary steps, affects quality the most */
#define LIGHT_STEPS 4 /* light steps, how much steps in the light direction are taken */


/*
Next we'll define the main scattering function.
This traces a ray from start to end and takes a certain amount of samples along this ray, in order to calculate the color.
For every sample, we'll also trace a ray in the direction of the light, 
because the color that reaches the sample also changes due to scattering
*/
float3 calculate_scattering(
	float3 start, 				// the start of the ray (the camera position)
    float3 dir, 				// the direction of the ray (the camera vector)
    float max_dist, 			// the maximum distance the ray can travel (because something is in the way, like an object)
    float3 scene_color,			// the color of the scene
    float3 light_dir, 			// the direction of the light
    float3 light_intensity,		// how bright the light is, affects the brightness of the atmosphere
    float3 planet_position, 	// the position of the planet
    float planet_radius, 		// the radius of the planet
    float atmo_radius, 			// the radius of the atmosphere
    float3 beta_ray, 			// the amount rayleigh scattering scatters the colors (for earth: causes the blue atmosphere)
    float3 beta_mie, 			// the amount mie scattering scatters colors
    float3 beta_absorption,   	// how much air is absorbed
    float3 beta_ambient,		// the amount of scattering that always occurs, cna help make the back side of the atmosphere a bit brighter
    float g, 					// the direction mie scatters the light in (like a cone). closer to -1 means more towards a single direction
    float height_ray, 			// how high do you have to go before there is no rayleigh scattering?
    float height_mie, 			// the same, but for mie
    float height_absorption,	// the height at which the most absorption happens
    float absorption_falloff,	// how fast the absorption falls off from the absorption height
    int steps_i, 				// the amount of steps along the 'primary' ray, more looks better but slower
    int steps_l 				// the amount of steps along the light ray, more looks better but slower
) {
    // add an offset to the camera position, so that the atmosphere is in the correct position
    start -= planet_position;
    // calculate the start and end position of the ray, as a distance along the ray
    // we do this with a ray sphere intersect
    float a = dot(dir, dir);
    float b = 2.0 * dot(dir, start);
    float c = dot(start, start) - (atmo_radius * atmo_radius);
    float d = (b * b) - 4.0 * a * c;
    
    // stop early if there is no intersect
    if (d < 0.0) return scene_color;
    
    // calculate the ray length
    float2 ray_length = float2(
        max((-b - sqrt(d)) / (2.0 * a), 0.0),
        min((-b + sqrt(d)) / (2.0 * a), max_dist)
    );
    
    // if the ray did not hit the atmosphere, return a black color
    if (ray_length.x > ray_length.y) return scene_color;
    // prevent the mie glow from appearing if there's an object in front of the camera
    bool allow_mie = max_dist > ray_length.y;
    // make sure the ray is no longer than allowed
    ray_length.y = min(ray_length.y, max_dist);
    ray_length.x = max(ray_length.x, 0.0);
    // get the step size of the ray
    float step_size_i = (ray_length.y - ray_length.x) / float(steps_i);
    
    // next, set how far we are along the ray, so we can calculate the position of the sample
    // if the camera is outside the atmosphere, the ray should start at the edge of the atmosphere
    // if it's inside, it should start at the position of the camera
    // the min statement makes sure of that
    float ray_pos_i = ray_length.x;
    
    // these are the values we use to gather all the scattered light
    float3 total_ray = (float3)(0.0); // for rayleigh
    float3 total_mie = (float3)(0.0); // for mie
    
    // initialize the optical depth. This is used to calculate how much air was in the ray
    float3 opt_i = (float3)(0.0);
    
    // also init the scale height, avoids some float2's later on
    float2 scale_height = float2(height_ray, height_mie);
    
    // Calculate the Rayleigh and Mie phases.
    // This is the color that will be scattered for this ray
    // mu, mumu and gg are used quite a lot in the calculation, so to speed it up, precalculate them
    float mu = dot(dir, light_dir);
    float mumu = mu * mu;
    float gg = g * g;
    float phase_ray = 3.0 / (50.2654824574 /* (16 * pi) */) * (1.0 + mumu);
    float phase_mie = allow_mie ? 3.0 / (25.1327412287 /* (8 * pi) */) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg)) : 0.0;
    
    // now we need to sample the 'primary' ray. this ray gathers the light that gets scattered onto it
    for (int i = 0; i < steps_i; ++i)
	{
        
        // calculate where we are along this ray
        float3 pos_i = start + dir * (ray_pos_i + step_size_i * 0.5);
        
        // and how high we are above the surface
        float height_i = length(pos_i) - planet_radius;
        
        // now calculate the density of the particles (both for rayleigh and mie)
        float3 density = float3(exp(-height_i / scale_height), 0.0);
        
        // and the absorption density. this is for ozone, which scales together with the rayleigh, 
        // but absorbs the most at a specific height, so use the sech function for a nice curve falloff for this height
        // clamp it to avoid it going out of bounds. This prevents weird black spheres on the night side
        density.z = clamp((1.0 / cosh((height_absorption - height_i) / absorption_falloff)) * density.x, 0.0, 1.0);
        density *= step_size_i;
        
        // Add these densities to the optical depth, so that we know how many particles are on this ray.
        opt_i += density;

        // Calculate the step size of the light ray.
        // again with a ray sphere intersect
        // a, b, c and d are already defined
        a = dot(light_dir, light_dir);
        b = 2.0 * dot(light_dir, pos_i);
        c = dot(pos_i, pos_i) - (atmo_radius * atmo_radius);
        d = (b * b) - 4.0 * a * c;

        // no early stopping, this one should always be inside the atmosphere
        // calculate the ray length
        float step_size_l = (-b + sqrt(d)) / (2.0 * a * float(steps_l));

        // and the position along this ray
        // this time we are sure the ray is in the atmosphere, so set it to 0
        float ray_pos_l = 0.0;

        // and the optical depth of this ray
        float3 opt_l = (float3)(0.0);
        
        // now sample the light ray
        // this is similar to what we did before
        for (int l = 0; l < steps_l; ++l) {

            // calculate where we are along this ray
            float3 pos_l = pos_i + light_dir * (ray_pos_l + step_size_l * 0.5);

            // the heigth of the position
            float height_l = length(pos_l) - planet_radius;

            // calculate the particle density, and add it
            float3 density_l = float3(exp(-height_l / scale_height), 0.0);
            density_l.z = clamp((1.0 / cosh((height_absorption - height_l) / absorption_falloff)) * density_l.x, 0.0, 1.0);
            opt_l += density_l * step_size_l;

            // and increment where we are along the light ray.
            ray_pos_l += step_size_l;
            
        }
        
        // Now we need to calculate the attenuation
        // this is essentially how much light reaches the current sample point due to scattering
        float3 attn = exp(-(beta_mie * (opt_i.y + opt_l.y) + beta_ray * (opt_i.x + opt_l.x) + beta_absorption * (opt_i.z + opt_l.z)));

        // accumulate the scattered light (how much will be scattered towards the camera)
        total_ray += density.x * attn;
        total_mie += density.y * attn;

        // and increment the position on this ray
        ray_pos_i += step_size_i;
    	
    }
    
    // calculate how much light can pass through the atmosphere
    float3 opacity = exp(-(beta_mie * opt_i.y + beta_ray * opt_i.x + beta_absorption * opt_i.z));
    
	// calculate and return the final color
    return (
        	phase_ray * beta_ray * total_ray // rayleigh color
       		+ phase_mie * beta_mie * total_mie // mie
            + opt_i.x * beta_ambient // and ambient
    ) * light_intensity + scene_color * opacity; // now make sure the background is rendered correctly
}


float4 RenderScenePS(VsSceneOutput input) : COLOR0
{ 
	float4 sampleA = tex2D(TextureColorSampler, input.texCoord);
	float4 sampleB = tex2D(TextureDataSampler, input.texCoord);
	float4 sampleC = tex2D(TextureNormalSampler, input.texCoord);
/*	
	float4 lightSideSample = tex2D(TextureColorSampler, input.texCoord);
	float4 darkSideSample = tex2D(TextureDataSampler, input.texCoord);
	float4 normalSample = tex2D(TextureNormalSampler, input.texCoord);
*/	
	sampleA.rgb = SRGBToLinear(sampleA.rgb);
	PBRProperties Properties;

//		output.posObj = position;
//	output.normalObj = normal;
	
	//derivatives are used so they can't branch
	float waterMask = saturate(sampleA.a * 4.0);
	float propertiesMask = saturate(waterMask * 32.0 - 31.0);

	float2 OffsetShadow, OffsetParallax;
	float3 normal = normalize(input.normal);
	float3 normalSphere = normal;	
	float3 view = normalize(-input.pos);
	float3 light = normalize(input.lightDir);	
	float3 normalLand = CotangentDerivativeMap(input.pos, normal, light, view, GetNormalDXT5(sampleC), input.texCoord, OffsetShadow, OffsetParallax);
//	return float4(saturate(dot(normal, light)).rrr, 1.0);
//	return float4(0.0, frac(distance(input.texCoord, UVS) * 10.0), 0.0, 1.0);
	normal = lerp(normal, normalLand, propertiesMask);	
//	return float4(waterMask.rrr, 1.0);
	float4 waterNgrad = 0;
	if(waterMask < 1.0)
	{
//		normal = 0.0;
		float waterScale = 1.0 / 25.0;
		float waterSpeed = 25.0;
		float waterIntensity = 0.75;
		waterNgrad = snoise(input.posObj * waterScale + length(input.posObj * waterScale) + g_Time * waterSpeed * waterScale) * waterIntensity;
		normal = lerp(mul(normalize(input.normalObj + waterNgrad.rgb), (float3x3)g_World), normal, Square(waterMask));
	}

	sampleA.a = saturate((sampleA.a - 0.25) * (1.0 / 0.75));
	if(waterMask < 1.0)
	{
		float fresnel = Pow5(1.0 - abs(dot(view, normal)));

//		propertiesMask *= propertiesMask * (3.0 - 2.0 * propertiesMask);
		sampleA.rgb = pow(sampleA.rgb, 1.5 - fresnel * 1.4);
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
	float NoV					= dot(normal, view);

	float2 UVcloud = float2(input.texCoord.x + rotatationTime, input.texCoord.y);
	float2 UVstep = UVcloud + OffsetParallax;// + OffsetParallax;
	UVstep += (tex2D(CloudLayerSampler, UVstep).g - 0.5) * OffsetParallax * 0.25;
	
	float cloudDensity = 0.2;
	float cloudDensityInv = 1.0 / cloudDensity;
	float cloudMip = sqrt(cloudDensityInv);
	float cloudSample = Square(tex2D(CloudLayerSampler, UVstep).g);
	Properties.DiffuseColor		= lerp(Properties.DiffuseColor, g_CloudColor.rgb, cloudSample);
	Properties.SpecularColor	= lerp(Properties.SpecularColor, (float3)0.08, cloudSample);
	Properties.Roughness		= lerp(Properties.Roughness, 1.0, cloudSample);
	
	float cloudShadow 			= tex2Dbias(CloudLayerSampler, float4(UVcloud, 0, cloudMip)).g;
	
	cloudShadow 				= exp(-cloudShadow);
	float cloudShadowRoughness	= cloudShadow;
	cloudShadow					= max(cloudShadow, cloudSample);
	
	float3 DiffuseSample		= SRGBToLinear(texCUBE(EnvironmentIlluminationCubeSampler, lerp(normal, normalSphere, cloudSample))).rgb;
		
	diffuse 					+= (lerp(Properties.DiffuseColor, g_CloudColor.rgb, cloudSample)) * DiffuseSample * cloudShadow;	
	
	float3 reflection			= -(view - 2.0 * normal * NoV);
	float3 ReflectionSample 	= SRGBToLinear(texCUBElod(TextureEnvironmentCubeSampler, float4(reflection, max(cloudShadow * 6.0, Properties.RoughnessMip)))).rgb;
	
	specular 					+= ReflectionSample * AmbientBRDF(saturate(abs(NoV)), Properties) * cloudShadow;



	float cloudAbsorption = 0;
	float4 cloudsShadowColor = 0;
	if(NoLsphere > 0)
	{
//		cloudAbsorption += cloudSample * cloudDensityInv;
		cloudShadow = max(exp(-tex2Dbias(CloudLayerSampler, float4(UVcloud + OffsetShadow, 0.0, cloudMip)).g), cloudSample);

		float steps = 32.0;
		float stepsInv = 1.0 / steps;		
		
		for(int i =0; i < steps - (steps * NoLsphere); i++)
		{
			UVstep += OffsetShadow * stepsInv;
			cloudAbsorption = max(cloudAbsorption, tex2Dbias(CloudLayerSampler, float4(UVstep, 0.0, i * stepsInv * cloudMip)).g * (1.0 - stepsInv * i));
		}
		cloudAbsorption = exp(-cloudAbsorption);

		cloudsShadowColor = float4(lerp(float3(0.25,0.3,0.35), lerp((float3)1.0, float3(1.0,0.95,0.8) * 1.4, cloudSample), cloudAbsorption), cloudAbsorption) * cloudShadow;
	}
	float shadowScalar 	= dot(input.lightDir, input.normal);
	float cityMask = saturate(-4.0 * shadowScalar);
	if(cityMask > 0.0)
	{	
		float cloudEmissiveRim = 0.25;
		float cloudEmissiveAbsorption = exp(-cloudSample * cloudDensityInv + cloudEmissiveRim) * cityMask;
		emissive += (SRGBToLinear(tex2Dbias(TextureDataSampler, float4(input.texCoord, 0.0, 7.0 - exp(-cloudSample * cloudDensityInv) * 7.0))).rgb) * cloudEmissiveAbsorption * 2.0;
		//TODO cleanup
		emissive += Square(SRGBToLinear(tex2Dbias(TextureDataSampler, float4(input.texCoord, 0.0, 4.0))).rgb) * cloudEmissiveAbsorption * 0.5;
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
		float3 subsurfaceColor 		= SRGBToLinear(tex2Dbias(TextureColorSampler, float4(input.texCoord, 0.0, 2.0)).rgb) * Properties.SubsurfaceOpacity * (1.0 - cloudSample) + g_CloudColor * cloudSample;	

		float InScatter				= pow(saturate(dot(light, -view)), 12) * lerp(3, .1f, Properties.SubsurfaceOpacity);
		float NormalContribution	= saturate(dot(normal, normalize(view + light)) * Properties.SubsurfaceOpacity + 1.0 - Properties.SubsurfaceOpacity);
		float BackScatter		 	= Properties.AO * NormalContribution * 0.1591549431;
		subsurfaceScatter 			= lerp(BackScatter, 1, InScatter) * subsurfaceColor * saturate(shadowScalar + Properties.SubsurfaceOpacity);
		subsurfaceScatter			*= cloudsShadowColor.rgb;
	
		InScatter					= pow(NoV, 12) * lerp(3, .1f, Properties.SubsurfaceOpacity);
		NormalContribution			= saturate(dot(normal, reflection) * Properties.SubsurfaceOpacity + 1.0 - Properties.SubsurfaceOpacity);
		BackScatter		 			= Properties.AO * NormalContribution * 0.1591549431;
		subsurfaceScatter	 		+= subsurfaceColor * lerp(BackScatter, 1, InScatter) * (SRGBToLinear(tex2Dbias(TextureColorSampler, float4(input.texCoord, 0.0, 6.0)).rgb * Square(NoLsphere) * (1.0 - cloudSample) + DiffuseSample));
	}
//	return LinearToSRGB(float4(1.0 - exp(-subsurfaceScatter), 1.0));
	return LinearToSRGB(float4(1.0 - exp(-(diffuse + specular + emissive + subsurfaceScatter)), 1.0));
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
	o.Position = mul(float4(iPosition * /*thicknessModifier*/(1.0 + ATMOSPHERE_SCALE), 1.0f), g_WorldViewProjection);
	
	//Texture Coordinates
    o.TexCoord0 = iTexCoord; 
	
    //Calculate  Normal       
    o.Normal = normalize(mul(iNormal, (float3x3)g_World));
    
    //Position
    float3 positionInWorldSpace = mul(float4(iPosition, 1.f), g_World).xyz;
    o.Pos = positionInWorldSpace;// * ATMOSPHERE_SCALE;//(1.0 / ATMOSPHERE_SCALE);
    //Calculate Light
	o.Light = normalize(g_Light0_Position - positionInWorldSpace);
	
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
/*
	float noiseScale = 10.f;
	
	float rotatationTime = g_Time / 800;
	float indexTime = g_Time/70;
	
	float3 index = float3(i.TexCoord0, indexTime);
	float domainPhaseShift = tex3D(NoiseSampler, noiseScale * index).x;
	
	float4 cloudColor = tex2D(CloudLayerSampler, float2(i.TexCoord0.x + rotatationTime, i.TexCoord0.y));
//	cloudColor *= domainPhaseShift;
//	
	
	cloudColor = 1.0 - pow(2.71828182846, -(cloudColor * cloudColor) * 2.0);
	cloudColor *= g_CloudColor;
*/
	//Light and Normal - renormalized because linear interpolation screws it up
	float3 light = normalize(i.Light);
	float3 normal = normalize(i.Normal);
	float3 view = normalize(i.View);
/*	
	float dotLightNormal = saturate(dot(light , normal) * 0.9 + 0.1);	
	
	//Atmosphere Scattering
    float ratio = 1.f - max(dot(normal, view), 0.f);
	float4 atmosphere = g_GlowColor * pow(ratio, 2.f);
		
	oColor0.rgb = (cloudColor.rgb + atmosphere.rgb) * dotLightNormal;//(cloudColor + atmosphere) * dotLightNormal;
	oColor0.a = cloudColor.a;
	oColor0 *= 0;
*/	


	float NoV = abs(dot(normal, view));


	float3 DiffuseSample		= texCUBE(EnvironmentIlluminationCubeSampler, normal).rgb;
	
	float3 atmoColor = calculate_scattering(
    	mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz * BASE_SCALE,					// the position of the camera
        view, 																		// the camera vector (ray direction of this pixel)
        g_Radius * BASE_SCALE * (1.0 - ATMOSPHERE_SCALE) - 100, 					// max dist, since nothing will stop the ray here, just use some arbitrary value
        (float3)0,																	// scene color, just the background color here
        -light,																		// light direction
        ATMOSPHERE_INTENSITY,														// light intensity, 40 looks nice
		i.Pos * BASE_SCALE,															// position of the planet
        g_Radius * BASE_SCALE * (1.0 - ATMOSPHERE_SCALE),                  			// radius of the planet in meters
        g_Radius * BASE_SCALE,					                 					// radius of the atmosphere in meters
        RAY_BETA,																	// Rayleigh scattering coefficient
        MIE_BETA,                       											// Mie scattering coefficient
        ABSORPTION_BETA,                											// Absorbtion coefficient
        DiffuseSample * AMBIENT_BETA,												// ambient scattering, turned off for now. This causes the air to glow a bit when no light reaches it
        G,                          												// Mie preferred scattering direction
        HEIGHT_RAY,                     											// Rayleigh scale height
        HEIGHT_MIE,                     											// Mie scale height
        HEIGHT_ABSORPTION,															// the height at which the most absorption happens
        ABSORPTION_FALLOFF,															// how fast the absorption falls off from the absorption height
        PRIMARY_STEPS, 																// steps in the ray direction
        LIGHT_STEPS 																// steps in the light direction
    );
	atmoColor = 1.0 - exp(-atmoColor);// we aren't HDR	
	oColor0.rgb = atmoColor;
	float atmoAlpha = saturate(dot(atmoColor, (float3)(1.0 / 3.0)));//there's a ton of nice reds
	oColor0.a = atmoAlpha;
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
//		DestBlend = InvSrcAlpha;
*/
		//additive
		ZWriteEnable = false;
		AlphaTestEnable = false;
		AlphaBlendEnable = true;
		SrcBlend = SRCALPHA;
		DestBlend = ONE; 
		
    }
}