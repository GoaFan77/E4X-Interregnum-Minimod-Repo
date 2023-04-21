
//math constants, don't touch!
#define SF (1.0/float(0xffffffffU))
#define HALFPI 1.5707963267948966192313216916398
#define PI 		3.1415926535897932384626433832795
#define INVPI 	0.3183098861837906715377675267450
#define TWOPI 	6.2831853071795864769252867665590
#define INVTWOPI 	0.1591549430918953357688837633725
#define RAYLEIGHDIVIDER 0.0596831036595 // 3/(16*pi)

//master switch
#define IS_BLOOM_PASS					//turns off everything not needed for a good matching bloom!

//#define DEBUG_PLANET_MODES			// Can visulaize planet types with simple flat color following the rules below!
//#define COMPARE_UV_X					// Compares mesh uv x with procedural one
//#define COMPARE_UV_Y					// Compares mesh uv y with procedural one
//#define COMPARE_UV					// Shows parallaxed uvs with a tiling frac on for reference
//#define DEBUG_CLOUD_COLORS			// Allows you to putput raw cloud composite colors
/*
Hex values
formatting aarrggbb
//debug 
	IMPORTANT! Setting the Modes can help performance! and allows paramaters to do different things in different cases
	IMPORTANT! Some HEX values are inverse, so 00 is highest intensity and FF is lowest!
	IMPORTANT! .entity file parameters, for all planets where applicable:
								cloudColor RGB HEX  = Cloud Color
								cloudColor A HEX 	= Cloud Opacity					
								glowColor A HEX 	= Cloud Sofness, basically a mip blur
								glowColor R HEX		= Cloud Density, less is more
								glowColor G HEX		= Cloud Micro Weight
								glowColor B HEX		= Cloud Flowmap Speed
	g_GlowColor
	//debug color dark grey
	noAtmosphereMode 			Specular A HEX 		== 00, no clouds or ligting strikes supported
								Specular R HEX		= UNUSED!
								Specular G HEX		= UNUSED!								
								Specular B HEX		= Fog Density, here simulating regolithic dust obviously ;)
								Ambient RGB HEX 	= UNUSED!
								Ambient A HEX   	= UNUSED!
								Diffuse A HEX 		= Mie atmosphere scattering
								Diffuse RGB HEX 	= Rayleigh atmosphere scattering, as well as fog and lightning color!
								
	//debug color orange red
	volcanoMode 				Specular A HEX 		== 40, lava instead of cities and ash plumes are added to the clouds, subsurface scatering NOT supported
								Specular R HEX		= UNUSED!
								Specular G HEX		= UNUSED!									
								Specular B HEX		= Fog Density
								Ambient RGB HEX 	= Smoke Color!
								Ambient A HEX		= lightning strike frequency
								Diffuse A HEX 		= Mie atmosphere scattering
								Diffuse RGB HEX 	= Rayleigh atmosphere scattering, as well as fog and lightning color!
	
	//debug color bright grey
	gasGiantMode 				Specular A HEX 		== 80, planets that are lave based, no citylight
								Specular R HEX 		= Gas Displacement Intensity
								Specular G HEX 		= Gas Speed
								Specular B HEX		= UNUSED!
								Diffuse A HEX 		= Mie atmosphere scattering
								Diffuse RGB HEX 	= Rayleigh atmosphere scattering, as well as fog and lightning color!

	//debug color bright green
	greenHouseMode				Specular A HEX 		== c0, no clouds or oceans
								Specular R HEX		= UNUSED!
								Specular G HEX		= UNUSED!										
								Specular B HEX		= Fog Density
								Ambient RGB HEX 	= UNUSED!
								Ambient A HEX		= lightning strike frequency
								Diffuse A HEX 		= Mie atmosphere scattering
								Diffuse RGB HEX 	= Rayleigh atmosphere scattering, as well as fog and lightning color!
	
	//debug color blue if water is on otherwise dark green
	oceanMode 					Specular A HEX 		== ff, planets that got oceans, support citylights too
								Specular R HEX 		= if 00 will turn off water! OceanHeight, relative to heightmap,
								Specular G HEX 		= OceanOpacity
								Specular B HEX		= Fog Density
								Ambient RGB HEX 	= water color
								Ambient A HEX		= lightning strike frequency
								Diffuse A HEX 		= Mie atmosphere scattering
								Diffuse RGB HEX 	= Rayleigh atmosphere scattering, as well as fog and lightning color!
*/
#define USE_MATHEMATICALY_PERFECT_UV		//makes it so texture don't have to be baked to the specific mesh.

#define SUPPORT_OCEAN_RIPPLES
//	#define SUPPORT_SHORE_WAVES
//options, you can comment them out to tweak performance if you have performance problems
#define SUPPORT_PARALLAX_DISPLACEMENT
	//#define COMPLEX_CLOUD_PARALLAX		// very expensive but great for screenshots!
	#define NO_ATMOSPHERE_SILHOUETTE		// will allow silhouette NoAtmosphere Planets only!!
	#define PARALLAX_SCALE 0.01				// IMPORTANT, relative to Glossiness is also, high radius planet has little parallax, low radius has huge!
#define SUPPORT_PARALLAX_SHADOWS        	// only works if SUPPORT_PARALLAX_DISPLACEMENT is also on!

#define SUPPORT_SUBSURFACE

#define FOG_SCALE 4.0						//global fog density scale
#define FOG_ANGLE_SCATTER 4.0				//global fog angle density scale
#define CLOUD_SCALE 2.0						//global cloud density scale
#define CLOUD_EMISSIVE_ABSORPTION 8.0		//global cloud absorption of emissive colors
#define CLOUD_MIP_BIAS 20.0					//global cloud blurriness in the center of the clouds, .entitny glowColor.a's effect

#ifdef IS_BLOOM_PASS
	//BLOOM! careful, this will affect ALL planets!
	#define AMBIENT_BOOST 8.0
	#define EMISSIVE_INTENSITY 8.0
	#define LAVA_INTENSITY 1.0
	#define LIGHTNING_INTENSITY 1.0
	//#define SUPPORT_SPHERE_LIGHT				// more expensive but nicer, especially for planets close to main star!
	#define LIGHT_INTENSITY_FAR 0.1				// light intensity far from stars
	#define LIGHT_INTENSITY_NEAR	0.8			// light intensity close to stars

	#undef SUPPORT_OCEAN_RIPPLES
//	#define LINEAR_OUTPUT //for bloom pass!
	#define PARALLAX_DISPLACEMENT_STEPS 8
	#define PARALLAX_SHADOW_STEPS 1
	//#define CLOUD_DISPLACEMENT_STEPS 1//never actually runs in bloom pass!
	#define SAMPLE_STEPS 2
	#define SUN_STEPS 2
#else
	//careful, this will affect ALL planets!
	#define AMBIENT_BOOST 8.0
	#define EMISSIVE_INTENSITY 1.0
	#define LAVA_INTENSITY 8.0
	#define LIGHTNING_INTENSITY 1.0
	//#define SUPPORT_SPHERE_LIGHT				// more expensive but nicer, especially for planets close to main star!
	#define LIGHT_INTENSITY_FAR 0.4			// light intensity far from stars
	#define LIGHT_INTENSITY_NEAR	3.0			// light intensity close to stars
	
	#define PARALLAX_DISPLACEMENT_STEPS 32
	#define PARALLAX_SHADOW_STEPS 16
	#define SAMPLE_STEPS 6
	#define SUN_STEPS 6
	#define CLOUD_DISPLACEMENT_STEPS 8	//only aplicable if COMPLEX_CLOUD_PARALLAX
	#define DETAILED_BLURS
#endif

//atmosphere pass
#define DISTANCE_MULT_VAL 1000.0f

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
	float3 tangent : TEXCOORD2;
	float3 lightPos : TEXCOORD3;
	float3 pos : TEXCOORD4;
	float3 posObj : TEXCOORD5;
	float3 normalObj : TEXCOORD6;
};

//TODO clean up unused interpolaters later, since this is probably not something the compiler will do!
VsSceneOutput RenderSceneOceanVS(
	float3 position : POSITION, 
	float3 normal : NORMAL,
	float3 tangent : TANGENT,
	float2 texCoord : TEXCOORD0)	
{
	VsSceneOutput output;  
	const bool oceanMode 					= g_MaterialSpecular.a 					!= 1.0;
//	const bool noAtmosphereMode 			= g_MaterialSpecular.a 					== 0.0;
//	const bool volcanoMode 					= round(g_MaterialSpecular.a * 4.0) 	== 1.0;
//	const bool gasGiantMode 				= round(g_MaterialSpecular.a * 4.0) 	== 2.0;
//	const bool greenHouseMode	 			= round(g_MaterialSpecular.a * 4.0) 	== 3.0;
	
	if(oceanMode)
	{
		output.position 					= 1.0/(1-oceanMode);//0x7fc00000;
		output.texCoord						= 0;
		output.normal 						= 0;
		output.tangent 						= 0;
		output.lightPos 					= 0;
		output.pos 							= 0;
		output.posObj 						= 0;
		output.normalObj 					= 0;
	}
	else
	{
		output.position 					= mul(float4(position, 1.0f), g_WorldViewProjection);
		
		output.texCoord 					= texCoord; 
		output.normal 						= mul(normal, (float3x3)g_World);
		output.tangent 						= mul(tangent, (float3x3)g_World);
		
		float3 positionInWorldSpace 		= mul(float4(position, 1.f), g_World).xyz;
	
		output.lightPos 					= g_Light0_Position.xyz - positionInWorldSpace;
		
		output.pos 							= positionInWorldSpace;
		output.posObj 						= position;
		output.normalObj 					= normal;
	}
    return output;
}

//TODO clean up unused interpolaters later, since this is probably not something the compiler will do!
VsSceneOutput RenderSceneNoAtmosphereVS(
	float3 position : POSITION, 
	float3 normal : NORMAL,
	float3 tangent : TANGENT,
	float2 texCoord : TEXCOORD0)	
{
	VsSceneOutput output;  
//	const bool oceanMode 					= g_MaterialSpecular.a 					== 1.0;
	const bool noAtmosphereMode 			= g_MaterialSpecular.a 					!= 0.0;
//	const bool volcanoMode 					= round(g_MaterialSpecular.a * 4.0) 	== 1.0;
//	const bool gasGiantMode 				= round(g_MaterialSpecular.a * 4.0) 	== 2.0;
//	const bool greenHouseMode	 			= round(g_MaterialSpecular.a * 4.0) 	== 3.0;
	
	if(noAtmosphereMode)
	{
		output.position 					= 1.0/(1-noAtmosphereMode);//0x7fc00000;
		output.texCoord						= 0;
		output.normal 						= 0;
		output.tangent 						= 0;
		output.lightPos 					= 0;
		output.pos 							= 0;
		output.posObj 						= 0;
		output.normalObj 					= 0;
	}
	else
	{
		#ifdef NO_ATMOSPHERE_SILHOUETTE
			float3 positionInWorldSpace 		= mul(float4(position, 1.f), g_World).xyz;
			float3 normalObj					= normalize(normal);
			output.normalObj 					= normalObj;
			output.normal 						= normalize(mul(normalObj, (float3x3)g_World));
			output.tangent 						= mul(tangent, (float3x3)g_World);
			const float gravity 				= rcp(g_MaterialGlossiness / 300000.0);
			float3 displacement					= normalObj * ((1-tex2Dlod(TextureDisplacementSampler, float4(texCoord, 0.0, 2.0)).a) * gravity);
			output.position 					= mul(float4(position - displacement, 1.0f), g_WorldViewProjection);
			
			output.texCoord 					= texCoord; 
	
			output.lightPos 					= g_Light0_Position.xyz - positionInWorldSpace;
			
			output.pos 							= positionInWorldSpace;
			output.posObj 						= position;
		#else
			output.position 					= mul(float4(position, 1.0f), g_WorldViewProjection);
			
			output.texCoord 					= texCoord; 
			output.normal 						= mul(normal, (float3x3)g_World);
			output.tangent 						= mul(tangent, (float3x3)g_World);
	
			float3 positionInWorldSpace 		= mul(float4(position, 1.f), g_World).xyz;
		
			output.lightPos 					= g_Light0_Position.xyz - positionInWorldSpace;
			
			output.pos 							= positionInWorldSpace;
			output.posObj 						= position;
			output.normalObj 					= normal;	
		#endif
	}
    return output;
}
//TODO clean up unused interpolaters later, since this is probably not something the compiler will do!
VsSceneOutput RenderSceneVolcanoVS(
	float3 position : POSITION, 
	float3 normal : NORMAL,
	float3 tangent : TANGENT,
	float2 texCoord : TEXCOORD0)	
{
	VsSceneOutput output;  
//	const bool oceanMode 					= g_MaterialSpecular.a 					== 1.0;
//	const bool noAtmosphereMode 			= g_MaterialSpecular.a 					== 0.0;
	const bool volcanoMode 					= round(g_MaterialSpecular.a * 4.0) 	!= 1.0;
//	const bool gasGiantMode 				= round(g_MaterialSpecular.a * 4.0) 	== 2.0;
//	const bool greenHouseMode	 			= round(g_MaterialSpecular.a * 4.0) 	== 3.0;
	
	if(volcanoMode)
	{
		output.position 					= 1.0/(1-volcanoMode);//0x7fc00000;
		output.texCoord						= 0;
		output.normal 						= 0;
		output.tangent 						= 0;
		output.lightPos 					= 0;
		output.pos 							= 0;
		output.posObj 						= 0;
		output.normalObj 					= 0;
	}
	else
	{
		output.position 					= mul(float4(position, 1.0f), g_WorldViewProjection);
		
		output.texCoord 					= texCoord; 
		output.normal 						= mul(normal, (float3x3)g_World);
		output.tangent 						= mul(tangent, (float3x3)g_World);

		float3 positionInWorldSpace 		= mul(float4(position, 1.f), g_World).xyz;
	
		output.lightPos 					= g_Light0_Position.xyz - positionInWorldSpace;
		
		output.pos 							= positionInWorldSpace;
		output.posObj 						= position;
		output.normalObj 					= normal;
	}
    return output;
}
//TODO clean up unused interpolaters later, since this is probably not something the compiler will do!
VsSceneOutput RenderGreenhouseVS(
	float3 position : POSITION, 
	float3 normal : NORMAL,
	float3 tangent : TANGENT,
	float2 texCoord : TEXCOORD0)	
{
	VsSceneOutput output;  
//	const bool oceanMode 					= g_MaterialSpecular.a 					== 1.0;
//	const bool noAtmosphereMode 			= g_MaterialSpecular.a 					== 0.0;
//	const bool volcanoMode 					= round(g_MaterialSpecular.a * 4.0) 	== 1.0;
//	const bool gasGiantMode 				= round(g_MaterialSpecular.a * 4.0) 	== 2.0;
	const bool greenHouseMode	 			= round(g_MaterialSpecular.a * 4.0) 	!= 3.0;
	
	if(greenHouseMode)
	{
		output.position 					= 1.0/(1-greenHouseMode);//0x7fc00000;
		output.texCoord						= 0;
		output.normal 						= 0;
		output.tangent 						= 0;
		output.lightPos 					= 0;
		output.pos 							= 0;
		output.posObj 						= 0;
		output.normalObj 					= 0;
	}
	else
	{
		output.position 					= mul(float4(position, 1.0f), g_WorldViewProjection);
		
		output.texCoord 					= texCoord; 
		output.normal 						= mul(normal, (float3x3)g_World);
		output.tangent 						= mul(tangent, (float3x3)g_World);

		float3 positionInWorldSpace 		= mul(float4(position, 1.f), g_World).xyz;
	
		output.lightPos 					= g_Light0_Position.xyz - positionInWorldSpace;
		
		output.pos 							= positionInWorldSpace;
		output.posObj 						= position;
		output.normalObj 					= normal;
	}
    return output;
}

//TODO clean up unused interpolaters later, since this is probably not something the compiler will do!
VsSceneOutput RenderSceneGasGiantVS(
	float3 position : POSITION, 
	float3 normal : NORMAL,
	float3 tangent : TANGENT,
	float2 texCoord : TEXCOORD0)	
{
	VsSceneOutput output;  
//	const bool oceanMode 					= g_MaterialSpecular.a 					== 1.0;
//	const bool noAtmosphereMode 			= g_MaterialSpecular.a 					== 0.0;
//	const bool volcanoMode 					= round(g_MaterialSpecular.a * 4.0) 	== 1.0;
	const bool gasGiantMode 				= round(g_MaterialSpecular.a * 4.0) 	!= 2.0;
//	const bool greenHouseMode	 			= round(g_MaterialSpecular.a * 4.0) 	== 3.0;
	
	if(gasGiantMode)
	{
		output.position 					= 1.0/(1-gasGiantMode);//0x7fc00000;
		output.texCoord						= 0;
		output.normal 						= 0;
		output.tangent 						= 0;
		output.lightPos 					= 0;
		output.pos 							= 0;
		output.posObj 						= 0;
		output.normalObj 					= 0;
	}
	else
	{
		output.position 					= mul(float4(position, 1.0f), g_WorldViewProjection);
		
		output.texCoord 					= texCoord; 
		output.normal 						= mul(normal, (float3x3)g_World);
		output.tangent 						= mul(tangent, (float3x3)g_World);

		float3 positionInWorldSpace 		= mul(float4(position, 1.f), g_World).xyz;
	
		output.lightPos 					= g_Light0_Position.xyz - positionInWorldSpace;
		
		output.pos 							= positionInWorldSpace;
		output.posObj 						= position;
		output.normalObj 					= normal;
	}
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

float3 SRGBToLinear(float3 color)
{
	#ifdef IS_BLOOM_PASS
		return Square(color);
	#else
		return pow(color, (float3)2.2);
	#endif
}

float4 SRGBToLinear(float4 color)
{
	return float4(SRGBToLinear(color.rgb), color.a);
}

float3 LinearToSRGB(float3 color)
{
	#ifdef IS_BLOOM_PASS
		#ifdef LINEAR_OUTPUT
			return color;
		#else
			return sqrt(color.rgb);
		#endif
	#else	
		return pow(color.rgb, (float3)(1.0/2.2));
	#endif
}

float4 LinearToSRGB(float4 color)
{
	return float4(LinearToSRGB(color.rgb), color.a);
}
float Luminance(float3 X)
{
	return dot(float3(0.3, 0.59, 0.11), X);
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

float4 Hash41(float p)
{
	float4 p4 							= frac((float4)(p) * float4(.1031, .1030, .0973, .1099));
    p4 									+= dot(p4, p4.wzxy + 33.33);
    return frac((p4.xxyz + p4.yzzw) 	* p4.zywx);   
}	
	
float3 Hash(float3 p, float2x2 rot)	
{	
	p 									= float3( 	dot(p,float3(127.1, 311.7, 74.7)),
													dot(p,float3(269.5, 183.3, 246.1)),
													dot(p,float3(113.5, 271.9, 124.6)));
	p 									= -1.0 + 2.0 * frac(sin(p) * 43758.5453123);
	p.xz 								= mul(p.xz, rot);
	return p;
}

// return value noise (in x) and its derivatives (in yzw)
float4 Nnoise(float3 p, float speed)
{
    // grid
    float3 i 							= floor(p);
    float3 w 							= frac(p);
		
    #if 1	
    // quintic interpolant	
    float3 u 							= w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
    float3 du 							= 30.0 * w * w * (w * (w - 2.0) + 1.0);
    #else	
    // cubic interpolant	
    float3 u 							= w * w * (3.0 - 2.0 * w);
    float3 du 							= 6.0 * w * (1.0 - w);
    #endif    
    
	float2x2 rot 						= float2x2(float2(cos(speed), -sin(speed)), float2(sin(speed), cos(speed)));
    // gradients	
    float3 ga 							= Hash(i + float3(0.0, 0.0, 0.0), rot);
    float3 gb 							= Hash(i + float3(1.0, 0.0, 0.0), rot);
    float3 gc 							= Hash(i + float3(0.0, 1.0, 0.0), rot);
    float3 gd 							= Hash(i + float3(1.0, 1.0, 0.0), rot);
    float3 ge 							= Hash(i + float3(0.0, 0.0, 1.0), rot);
	float3 gf 							= Hash(i + float3(1.0, 0.0, 1.0), rot);
    float3 gg 							= Hash(i + float3(0.0, 1.0, 1.0), rot);
    float3 gh 							= Hash(i + float3(1.0, 1.0, 1.0), rot);
		
    // projections	
    float va 							= dot(ga, w - float3(0.0,0.0,0.0));
    float vb 							= dot(gb, w - float3(1.0,0.0,0.0));
    float vc 							= dot(gc, w - float3(0.0,1.0,0.0));
    float vd 							= dot(gd, w - float3(1.0,1.0,0.0));
    float ve 							= dot(ge, w - float3(0.0,0.0,1.0));
    float vf 							= dot(gf, w - float3(1.0,0.0,1.0));
    float vg 							= dot(gg, w - float3(0.0,1.0,1.0));
    float vh 							= dot(gh, w - float3(1.0,1.0,1.0));
	
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

float AmbientDielectricBRDF(float Roughness, float NoV)
{
	const float2 c0 					= float2(-1	, -0.0275);
	const float2 c1 					= float2(1	, 0.0425);
	float2 r 							= Roughness * c0 + c1;
	return min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
}

float GetMipRoughness(float Roughness, float MipCount)
{
	//return MipCount - 1 - (3 - 1.15 * log2(Roughness));
	return sqrt(Roughness) * MipCount;
} 
// Brian Karis(Epic's) optimized unified term derived from Call of Duty metallic/dielectric term improved with https://bruop.github.io/ibl/
void AmbientBRDF(inout float3 diffuse, inout float3 specular, float NoV, PBRProperties Properties, float3 radiance = 1.0, float3 irradiance = 1.0, const bool multiscatter = true)
{
	#ifdef IS_BLOOM_PASS
		diffuse 						+= (lerp(Properties.DiffuseColor, Properties.SubsurfaceColor, Properties.SubsurfaceOpacity) * irradiance) * Properties.AO;
		specular 						+= (Properties.SpecularColor.g * AmbientDielectricBRDF(Properties.Roughness, NoV) * Properties.AO) * radiance;
	#else		
		float4 r 						= Properties.Roughness * float4(-1.0, -0.0275, -0.572, 0.022) + float4(1.0, 0.0425, 1.04, -0.04);
		float a004 						= min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
		float2 AB 						= float2(-1.04, 1.04) * a004 + r.zw;
		
		AB.y 							*= (1.0 - 1.0 / (1.0 + max(0.0, 50.0))) * 3.0;
		
		if(multiscatter)
		{	
			float3 FssEss 				= Properties.SpecularColor * AB.x + AB.y;	
			// Multiple scattering, from Fdez-Aguera
			float Ems 					= (1.0 - (AB.x + AB.y));
			float3 F_avg 				= NoV + (1.0 - NoV) * rcp(21.0);
			float3 FmsEms 				= Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
			float3 k_D 					= Properties.DiffuseColor * (1.0 - FssEss - FmsEms);
					
			diffuse 					+= ((FmsEms + k_D) * irradiance) * Properties.AO;
			specular 					+= (FssEss * radiance) * Properties.AO;
		}		
		else		
		{		
			diffuse 					+= (lerp(Properties.DiffuseColor, Properties.SubsurfaceColor, Properties.SubsurfaceOpacity) * irradiance) * Properties.AO;
			specular 					+= ((Properties.SpecularColor * AB.x + AB.y) * radiance) * Properties.AO;
		}
	#endif
	
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
	Ouput.NoL 							= dot(normal, light);
	Ouput.NoV 							= dot(normal, view);
	Ouput.VoL 							= dot(view, light);
	float invLenH 						= rcp(sqrt( 2.0 + 2.0 * Ouput.VoL));
	Ouput.NoH 							= saturate((Ouput.NoL + Ouput.NoV) * invLenH);
	Ouput.VoH 							= saturate(invLenH + invLenH * Ouput.VoL);
	Ouput.NoL 							= saturate(Ouput.NoL);
	Ouput.NoV 							= saturate(abs(Ouput.NoV * 0.9999 + 0.0001));
	return Ouput;
}

// recreate blue normal map channel
float DeriveZ(float2 normalXY)
{	

	float normalZ 						= sqrt(abs(1.0 - Square(Square(normalXY.x) - Square(normalXY.y))));
	
	return normalZ;
}
		
float2 ToNormalSpace(float2 normalXY)
{	
	return normalXY * 2.0 - 1.0;
}	
	
float3 GetNormalDXT5(float4 normalSample)
{
	float2 normalXY 					= normalSample.wy;
	normalXY 							= ToNormalSpace(normalXY);
	// play safe and normalize
	return normalize(float3(normalXY, DeriveZ(normalXY)));
}

float RayleighPhaseFunction(float VoL)
{
	return
#if 0
			3. * (1. + VoL * VoL)
			
	/ //------------------------
				(16. * PI);
#else
	(1. + (VoL * VoL)) * RAYLEIGHDIVIDER;
#endif	
}

float3 RayleighPhaseFunction(float3 VoL)
{
	return
#if 0
			3. * (1. + VoL * VoL)
			
	/ //------------------------
				(16. * PI);
#else
	(1. + (VoL * VoL)) * RAYLEIGHDIVIDER;
#endif	
}

float HenyeyGreensteinPhaseFunction(float VoL, float g)
{
	return
						(1. - g * g)
	/ //---------------------------------------------
		((4. + PI) * pow(abs(1. + g * g - 2. * g * VoL) + 0.0001, 1.5));
}

float2 Rotate(float2 pos, float rot)
{
	float s, c;
	sincos(rot, s, c);
	return mul(float2x2(c, -s, s, c), pos);
}

float VolcanoSmoke(float2 uv, float2 duvx, float2 duvy, float smokeDensity, float smokePuffyness, float smokeScale)	
{
	#ifdef IS_BLOOM_PASS
		return Square(tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).g) * smokeDensity;
	#else
		float smokeSample 			= tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).g;
	
		float smokeSpeed 			= (g_Time * 0.0025);			
		float3 smokePhase;
 		smokePhase.r 				= (g_Time * 0.5 + (smokeSample * 16));
		smokePhase.b				= abs(frac(smokePhase.r) * 2 - 1);
		smokePhase.g				= round(smokePhase.r);
		smokePhase.r				= floor(smokePhase.r);
		smokePhase.rg				= frac(smokePhase.rg * 0.0734 + smokeSpeed);

		float smokeBias				= max(Square(uv.y * 2 - 1) * 6.0 * smokeScale, (1.0 - smokeSample) * 16);
		duvx						*= smokeBias;
		duvy						*= smokeBias;
			
		float smokeAnimation 		= lerp(	tex2Dgrad(TextureDisplacementSampler, float2(uv.x + smokePhase.r, uv.y) * smokeScale, duvx, duvy).b, 
										tex2Dgrad(TextureDisplacementSampler, float2(uv.x + smokePhase.g, uv.y) * smokeScale, duvx, duvy).b, smokePhase.b)
										* smokePuffyness;
										
		return 1-exp(-max(0, Square(smokeSample) * smokeDensity + smokeSample * smokeAnimation));
	#endif
}

float3 GetFlow(float timer)
{
	return float3(frac(float2(timer, timer + 0.5)), abs(frac(timer) * 2 - 1));
}

float4 MapGasGiant(float4 map)
{
	return lerp(float4(map.r, map.g, 1.0, 1.0 - map.a), map * map * (3.0 - 2.0 * map), g_MaterialSpecular.r);
}
//compiler failed to choose the code moved to where it was called instead
float4 MapGasGiant(float map)
{
	return lerp(0.5, map, g_MaterialSpecular.r);
}

float CloudSim(float2 uv, float2 duvx, float2 duvy, float4 speed, float density, float cloudCuttoff)
{
	
	speed.x 						*= g_GlowColor.b;
	uv.x							= frac(uv.x - speed.x * 0.025);
	float4 baseSample 				= tex2Dgrad(CloudLayerSampler, uv, duvx, duvy);
		
	float2 flowDir 					= (baseSample.yw - 0.5) * speed.y;
	float2 DetailScale 				= float2(5, 6);//must be non-decimal numbers or tiling will break along the main equirectangular X seam!
	float swapA 					= frac(speed.x);
	float swapB 					= frac(speed.x + 0.5);
	
	float variationA 				= floor(speed.x) * 0.3;		
	float variationB 				= floor(speed.x + 0.5) * 0.7;
		
	float2 offsetA 					= (flowDir * swapA);
	float2 offsetB 					= (flowDir * swapB);
		
	float cloudA;	
	float cloudB;		
	
	cloudA 							= tex2Dgrad(CloudLayerSampler, uv - offsetA, duvx, duvy).b - g_GlowColor.r;
	cloudB 							= tex2Dgrad(CloudLayerSampler, uv - offsetB, duvx, duvy).b - g_GlowColor.r;
		
	float2 cloudBias 				= (max((float2)0, float2(cloudA, cloudB)) - cloudCuttoff) * CLOUD_MIP_BIAS * g_GlowColor.a + 1;
		
	cloudA 							+= (tex2Dgrad(CloudLayerSampler, (uv - offsetA * speed.z) * DetailScale.x + variationA, duvx * DetailScale.x * cloudBias.x, duvy * DetailScale.x * cloudBias.x).r
									+ 	tex2Dgrad(CloudLayerSampler, (uv - offsetA * speed.w) * DetailScale.y + variationA, duvx * DetailScale.y * cloudBias.x, duvy * DetailScale.y * cloudBias.x).r) * g_GlowColor.g;
		
	cloudB 							+= (tex2Dgrad(CloudLayerSampler, (uv - offsetB * speed.z) * DetailScale.x + variationB, duvx * DetailScale.x * cloudBias.y, duvy * DetailScale.x * cloudBias.y).r
									+	tex2Dgrad(CloudLayerSampler, (uv - offsetB * speed.w) * DetailScale.y + variationB, duvx * DetailScale.y * cloudBias.y, duvy * DetailScale.y * cloudBias.y).r) * g_GlowColor.g;

	return 1-exp(-Square(max(0, lerp(cloudA, cloudB, abs(swapA * 2.0 - 1.0)) - cloudCuttoff)) * density);

}

float3 ToTangentSpace(float3x3 trans, float3 vec)
{
    return float3(dot(trans[0], vec), dot(trans[1], vec), dot(trans[2], vec));//mul(trans, vec);
}

float3 ToWorldSpace(float3x3 trans, float3 vec)
{
    return mul(vec, trans);
}

float2 ParallaxVector(float3 viewTS, float parallaxScale)
{
#if 1 
	float  parallaxLength    		= sqrt(dot(viewTS.xy, viewTS.xy) / (dot(viewTS.z, viewTS.z)));
    return -normalize(viewTS.xy) * (parallaxScale * parallaxLength);
#else
	return (-viewTS.xy / viewTS.z) * parallaxScale;
#endif	
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
	Dot.NoL 						= dot(normal, light);
	Dot.NoV 						= dot(normal, view);
	Dot.VoL 						= dot(view, light);
	float distInvHalfVec 			= rsqrt(2.0 + 2.0 * Dot.VoL);
	Dot.NoH 						= ((Dot.NoL + Dot.NoV) * distInvHalfVec);
	Dot.VoH 						= (distInvHalfVec + distInvHalfVec * Dot.VoL);
}

void GetSphereNoH(inout Dots Dot, float sphereSinAlpha)
{
	if(sphereSinAlpha > 0)
	{
		float sphereCosAlpha 		= sqrt(1.0 - Square(sphereSinAlpha));
	
		float RoL 					= 2.0 * Dot.NoL * Dot.NoV - Dot.VoL;
		if(RoL >= sphereCosAlpha)
		{
			Dot.NoH 				= 1;
			Dot.VoH 				= abs(Dot.NoV);
		}
		else
		{
			float distInvTR 		= sphereSinAlpha * rsqrt(1.0 - Square(RoL));
			float NoTr 				= distInvTR * (Dot.NoV - RoL * Dot.NoL);

			float VoTr 				= distInvTR * (2.0 * Square(Dot.NoV) - 1.0 - RoL * Dot.VoL);

			float NxLoV 			= sqrt(saturate(1.0 - Square(Dot.NoL) - Square(Dot.NoV) - Square(Dot.VoL) + 2.0 * Dot.NoL * Dot.NoV * Dot.VoL));

			float NoBr 				= distInvTR * NxLoV;
			float VoBr 				= distInvTR * NxLoV * 2.0 * Dot.NoV;

			float NoLVTr 			= Dot.NoL * sphereCosAlpha + Dot.NoV + NoTr;
			float VoLVTr 			= Dot.VoL * sphereCosAlpha + 1.0 + 	VoTr;

			float p 				= NoBr   * VoLVTr;
			float q 				= NoLVTr * VoLVTr;
			float s 				= VoBr   * NoLVTr;

			float xNum 				= q * (-0.5 * p + 0.25 * VoBr * NoLVTr);
			float xDenom 			= Square(p) + s * (s - 2.0 * p) + NoLVTr * ((Dot.NoL * sphereCosAlpha + Dot.NoV) * Square(VoLVTr) + q * (-0.5 * (VoLVTr + Dot.VoL * sphereCosAlpha) - 0.5));
			float TwoX1 			= 2.0 * xNum / (Square(xDenom) + Square(xNum));
			float SinTheta 			= TwoX1 * xDenom;
			float CosTheta 			= 1.0 - TwoX1 * xNum;
			NoTr 					= CosTheta * NoTr + SinTheta * NoBr;
			VoTr 					= CosTheta * VoTr + SinTheta * VoBr;

			Dot.NoL 				= Dot.NoL * sphereCosAlpha + NoTr;
			Dot.VoL 				= Dot.VoL * sphereCosAlpha + VoTr;

			float distInvHalfVec 	= rsqrt(2.0 + 2.0 * Dot.VoL);
			Dot.NoH 				= saturate((Dot.NoL + Dot.NoV) * distInvHalfVec);
			Dot.VoH 				= saturate(distInvHalfVec + distInvHalfVec * Dot.VoL);
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

float GetAttenuation(float lightFalloff)
{
	return lerp(LIGHT_INTENSITY_NEAR, LIGHT_INTENSITY_FAR, lightFalloff);
}
void SphereLight(PBRProperties Properties, float3 normal, float3 view, inout float3 light, float3 lightPos, float lightRad, float4 lightColorIntensity, float3 atmosphereAbsorption, inout float3 specular, inout float3 diffuse, inout float attenuation, inout float shadow, inout float NoL)
{

	float lightSqr					= dot(lightPos, lightPos);
	float lightDist					= sqrt(lightSqr);
	float lightFalloff				= rcp(1.0 + lightSqr);//regular pointlight
		
	light							= lightPos / lightDist;
	NoL								= dot(normal, light);
	float sphereSinAlphaSqr			= saturate(Square(lightRad) * lightFalloff);

	// Patch to Sphere frontal equation ( Quilez version )
	float lightRadSqr 				= Square(lightRad);
	// Do not allow object to penetrate the light ( max )
	// Form factor equation include a (1 / FB_PI ) that need to be cancel
	// thus the " FB_PI *"
	float illuminance 				= PI * (lightRadSqr / (max(lightRadSqr, lightSqr))) * saturate(NoL);

	attenuation 					*= GetAttenuation(illuminance);


	float sphereSinAlpha			= sqrt(sphereSinAlphaSqr);
	
#if 1
	if(NoL < sphereSinAlpha)
	{
		NoL							= max(NoL, -sphereSinAlpha);
#if 0	
		float sphereCosBeta 		= NoL;
		float sphereSinBeta 		= sqrt(1.0 - Square(sphereCosBeta));
		float sphereTanBeta 		= sphereSinBeta / sphereCosBeta;
	
		float x 					= sqrt(rcp(sphereSinAlphaSqr) - 1.0);
		float y 					= -x / sphereTanBeta;
		float z 					= sphereSinBeta * sqrt(1 - Square(y));
	
		NoL 						= NoL * acos(y) - x * z + atan(z / x) / sphereSinAlphaSqr;
		NoL 						*= INVPI;
#else	
		NoL 						= Square(sphereSinAlpha + NoL) / (4.0 * sphereSinAlpha);
#endif	
	}	
#else	
	NoL 							= saturate((NoL + sphereSinAlphaSqr) / (1.0 + sphereSinAlphaSqr)) * shadow;
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

		float3 transmission 		= SubsurfaceLight(Properties.SubsurfaceColor, 1-Properties.SubsurfaceOpacity, light, normal, view, Properties.AO, (attenuation * (1.0 - Properties.SubsurfaceOpacity) + Properties.SubsurfaceOpacity));

	#if 1
		//Lambert
		diffuse 					+= lerp(((attenuation * NoL * INVPI) * Properties.DiffuseColor), transmission * NoL, Properties.SubsurfaceOpacity) * lightColorIntensity.rgb * atmosphereAbsorption;
	#else
		//Burley
		float FD90 					= 0.5 + 2 * Square(Dot.VoH) * Properties.Roughness;
		float FdV 					= 1.0 + (FD90 - 1.0) * Pow5(1.0 - Dot.NoV);
		float FdL 					= 1.0 + (FD90 - 1.0) * Pow5(1.0 - Dot.NoL);
		diffuse						+= lerp((Properties.DiffuseColor * (attenuation * NoL * INVPI * FdV * FdL)), transmission, Properties.SubsurfaceOpacity) * lightColorIntensity.rgb * atmosphereAbsorption;
	#endif
	}
}

void PointLight(PBRProperties Properties, float3 normal, float3 view, inout float3 light, float3 lightPos, float lightRad, float4 lightColorIntensity, float3 atmosphereAbsorption, inout float3 specular, inout float3 diffuse, inout float attenuation, inout float shadow, inout float NoL)
{

	float lightSqr					= dot(lightPos, lightPos);
	float lightDist					= sqrt(lightSqr);
	float lightFalloff				= rcp(1.0 + lightSqr);//regular pointlight
		
	light							= lightPos / lightDist;
	NoL								= dot(normal, light) * shadow;
	
	attenuation 					*= GetAttenuation(lightFalloff);

	[branch]
	if(NoL > 0)
	{
	//	Properties.Roughness 		= Square(Properties.Roughness * 0.5 + 0.5);
		float roughness2 			= Square(Properties.Roughness);

		Dots Dot;	
		GetDots(Dot, normal, view, light);

		Dot.NoV 					= saturate(abs(Dot.NoV) + 0.00001);
		Dot.NoL 					= saturate(Dot.NoL);
		
		float roughness4 			= Square(roughness2);

		float sphereRoughness4 		= roughness4;
		float specularEnergy 		= 1.0;

		float Fc 					= Pow5(1.0 - Dot.VoH);		
		float3 fresnel				= saturate(50.0 * Properties.SpecularColor.g ) * Fc + (1.0 - Fc) * Properties.SpecularColor;
		float distribution			= roughness4 / (PI * Square((Dot.NoH * roughness4 - Dot.NoH) * Dot.NoH + 1.0));
		float geometricShadowing	= 0.5 / (Dot.NoL * (Dot.NoV * (1.0 - roughness2) + roughness2) + Dot.NoV * (Dot.NoL * (1.0 - roughness2) + roughness2));
		specular 					+= (fresnel * lightColorIntensity.rgb * atmosphereAbsorption) * (attenuation * NoL * distribution * geometricShadowing * specularEnergy);

		float3 transmission 		= SubsurfaceLight(Properties.SubsurfaceColor, 1-Properties.SubsurfaceOpacity, light, normal, view, Properties.AO, (attenuation * (1.0 - Properties.SubsurfaceOpacity) + Properties.SubsurfaceOpacity));

	#if 1
		//Lambert
		diffuse 					+= lerp(((attenuation * NoL * INVPI) * Properties.DiffuseColor), transmission * NoL, Properties.SubsurfaceOpacity) * lightColorIntensity.rgb * atmosphereAbsorption;
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

void GetMainStarLight(PBRProperties Properties, float3 normal, float3 view, inout float3 light, float3 lightPos, float lightRad, float4 lightColorIntensity, float3 atmosphereAbsorption, inout float3 specular, inout float3 diffuse, inout float shadow, inout float attenuation, inout float NoL)
{
	#if SUPPORT_SPHERE_LIGHT
		SphereLight(Properties, normal, view, light, lightPos, lightRad, lightColorIntensity, atmosphereAbsorption, specular, diffuse, shadow, attenuation, NoL);
	#else	
		PointLight(Properties, normal, view, light, lightPos, lightRad, lightColorIntensity, atmosphereAbsorption, specular, diffuse, shadow, attenuation, NoL);
	#endif
}

float4 GetUVFromPos(inout float3 normalObj)
{
	float4 uvBasis;
	uvBasis.xy 						= atan2(float2(normalObj.x, normalObj.x), float2(normalObj.z, -normalObj.z)) * INVTWOPI;
	uvBasis.x 						= -uvBasis.x;
	uvBasis.z 						= acos(normalObj.y) * INVPI;
	uvBasis.w 						= -normalObj.y * INVPI;
	return uvBasis;
}

void GetEquirectangularUV(float3 normalObj, inout float2 duvx, inout float2 duvy, inout float2 uv)
{
	#ifdef USE_MATHEMATICALY_PERFECT_UV
		float4 uvBasis 					= GetUVFromPos(normalObj);
	
		uv 								= uvBasis.xz;
		
		// uses a small bias to prefer the first 'UV set'
		uv.x 							= (fwidth(uv.x) - 0.001 < fwidth(frac(uv.x)) ? uv.x : frac(uv.x)) + 0.5;
									
		duvx 							= float2(	normalObj.z > 0.0 ? ddx(uvBasis.x) : ddx(uvBasis.y), 
													abs(normalObj.y) > 0.025 ? ddx(uvBasis.z) : ddx(uvBasis.w));
		duvy 							= float2(	normalObj.z > 0.0 ? ddy(uvBasis.x) : ddy(uvBasis.y),
													abs(normalObj.y) > 0.025 ? ddy(uvBasis.z) : ddy(uvBasis.w));
	#else
		duvx							= ddx(uv);
		duvy							= ddy(uv);
	#endif
}

float3 GetEqurectangularPos(float2 uv)
{
	uv 								= mad(uv, float2(TWOPI, -PI), float2(-PI, HALFPI));
	float2 s, c;
	sincos(uv, s, c);
	return float3(c.y * s.x, s.y, c.y * c.x);
}

#if 0
// return RaySphere(sphereCenter, sphereRadius, rayOrigin, rayDir)
float3 RaySphere(float3 sphereCenter, float sphereRadius, float3 rayOrigin, float3 rayDir, const float depthOffset = 0.0)
{
	float3 offset = rayOrigin - sphereCenter;
	float a 						= 1;
	float b 						= 2 * dot(offset, rayDir);
	float c 						= dot(offset, offset) - sphereRadius * sphereRadius;
	float d 						= b*b - 4 * a * c;
	float s 						= 0;
	if(d > 0)
	{
		s 							= sqrt(d);
		float distToSphereNear 		= max(0, (-b - s) / (2 * a));
		float distToSphereFar 		= (-b + s) / (2 * a);
		
		if(distToSphereFar >= 0)
		{
			return float3(distToSphereNear, distToSphereFar - distToSphereNear, s);
		}
	}
	return float3(MF, 0.0, 0.0);
}
#endif

	void GetParallaxBase(inout float2 uv, inout float shadow, float2 parallaxTS, float3 lightTS, float2 duvx, float2 duvy, float NoL, const float waterHeight = 0.0, const bool RunBaseParallax = true)
	{
	#ifdef SUPPORT_PARALLAX_DISPLACEMENT
		float rayHeight						= 1.0;
		float2 offset						= 0;
		float oldRay						= 1.0;
		float yIntersect;				
		float oldTex						= 1;
		
		if(RunBaseParallax)
		{
			const float steps 					= PARALLAX_DISPLACEMENT_STEPS;
			const float stepsInv				= rcp(steps);
						
			
			float2 offsetStep					= parallaxTS * stepsInv;
			
			[loop]
			for (int i = 0; i < steps; ++i)
			{		
				float texAtRay 					= tex2Dgrad(TextureDisplacementSampler, uv + offset, duvx, duvy).a;
				
				if (rayHeight < texAtRay)
				{
					float xIntersect			= (oldRay - oldTex) + (texAtRay - rayHeight);
					xIntersect					= (texAtRay - rayHeight) / xIntersect;
					yIntersect					= (oldRay * xIntersect) + (rayHeight * (1.0 - xIntersect));
					offset						-= (xIntersect * offsetStep);
					break;			
				}
				oldRay							= rayHeight;
				rayHeight						-= stepsInv;
				offset 							+= offsetStep;
				oldTex 							= texAtRay;
			}
		}
		uv 									+= offset;

		#ifdef SUPPORT_PARALLAX_SHADOWS
			#ifndef IS_BLOOM_PASS
				[branch]
				if(NoL > -0.1)
				{
					float stepsLight				= PARALLAX_SHADOW_STEPS * (1-saturate(NoL)) + 1;		
		
					offset 							= 0;
							
					float texAtRay 					= saturate(tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).a - waterHeight) + waterHeight + 0.01;
							
					rayHeight						= texAtRay;
					
					float stepsLightInv				= rcp(stepsLight) * 0.5;//0.5 to match lightTS.z modification
					float dist						= 0;				
					float shadowPenumbra			= 1.25;
					
					for(int j = 0; j < stepsLight; j++)
					{
						if(rayHeight < texAtRay)
						{
							shadow 					= 0;
							break;
						}
						else
						{
							if(dist > 0)//possible AMD issue was this division! (that doesn't happen)
								shadow					= min(shadow, (rayHeight - texAtRay) * shadowPenumbra / dist);
						}
			
						oldRay						= rayHeight;
						rayHeight					+= lightTS.z * stepsLightInv;
							
						offset						+= lightTS.xy * stepsLightInv;
						oldTex						= texAtRay;
							
						texAtRay 					= saturate(tex2Dgrad(TextureDisplacementSampler, uv + offset, duvx, duvy).a - waterHeight) + waterHeight;
					
						dist						+= stepsLightInv;
					}		
				}
			
				shadow								= Square(shadow);
			#endif
		#endif
	#endif
	}
	
	void GetClouds(	inout float4 shadowCloud, 
					inout float3 cloudNormal, 
					inout float cloudOpacity, 
					inout float ao, 
					inout float shadow, 
					float poleDensity, 
					float NoLflat, 
					out float cloudSteps, 
					float3 posObj, 
					float2 uv, 
					float2 uvFlat, 
					float2 parallaxTS, 
					float3 lightTS, 
					float2 duvx, 
					float2 duvy, 
					inout float4 cloudColor, 
					float3 rayleighColor)
	{
		float cloudDensity 						= cloudColor.a * CLOUD_SCALE;//
		float cloudCuttoff 						= 0.01 + poleDensity;
		float cloudHeight 						= 0.1;
		float2 cloudUV 							= uvFlat - parallaxTS * cloudHeight;
		float4 cloudSpeed 						= float4(0.1 * (g_Time + tex3Dlod(NoiseSampler, float4(posObj / 1000.0, 0)).r), 0.125, 1.55, 1.633);

		#ifndef IS_BLOOM_PASS		
			#ifdef SUPPORT_PARALLAX_DISPLACEMENT
				#ifdef COMPLEX_CLOUD_PARALLAX		
					float2 offset				= 0;
					const float steps 			= CLOUD_DISPLACEMENT_STEPS;
					const float stepsInv		= rcp(steps);
					float rayHeight				= 1.0;
					float oldRay				= 1.0;
								
					float oldTex				= 1;
					float texAtRay				= 0;
					float yIntersect;			
					
					float2 offsetStep			= parallaxTS * stepsInv * 0.1;
					
					[loop]
					for (int i = 0; i < steps; ++i)
					{		
						float texAtRay 			= CloudSim(cloudUV + offset, duvx, duvy, cloudSpeed, cloudDensity, cloudCuttoff);	
	
						if (rayHeight < texAtRay)
						{
							float xIntersect	= (oldRay - oldTex) + (texAtRay - rayHeight);
							xIntersect			= (texAtRay - rayHeight) / xIntersect;
							yIntersect			= (oldRay * xIntersect) + (rayHeight * (1.0 - xIntersect));
							offset				-= (xIntersect * offsetStep);
							break;			
						}
						oldRay					= rayHeight;
						rayHeight				-= stepsInv;
						offset 					+= offsetStep;
						oldTex 					= texAtRay;
					}
					cloudUV 					+= offset;
				#else					
					float cloudDisplacement 	= CloudSim(cloudUV, duvx, duvy, cloudSpeed, cloudDensity, cloudCuttoff);		
					cloudUV						-= parallaxTS * ((cloudDisplacement - cloudDensity * 0.5) * 0.1);				
				#endif
			#endif
			
			cloudNormal.r						= (CloudSim(cloudUV - float2(0.0004, 0.0), duvx, duvy, cloudSpeed, cloudDensity, cloudCuttoff) - CloudSim(cloudUV + float2(0.0004, 0.0), duvx, duvy, cloudSpeed, cloudDensity, cloudCuttoff)),
			cloudNormal.y						= (CloudSim(cloudUV - float2(0.0, 0.0002), duvx, duvy, cloudSpeed, cloudDensity, cloudCuttoff) - CloudSim(cloudUV + float2(0.0, 0.0002), duvx, duvy, cloudSpeed, cloudDensity, cloudCuttoff)),
		#endif
		
		cloudOpacity 							= max(CloudSim(cloudUV, duvx, duvy, cloudSpeed, cloudDensity, cloudCuttoff), cloudOpacity);
		ao										*= Square(saturate((1-CloudSim(uv, duvx, duvy, cloudSpeed, cloudDensity, cloudCuttoff) * 0.5) + cloudOpacity));
												
		float3 cloudSum 						= cloudOpacity;//cloudOpacity * pow(cloudColor, (-2 * cloudOpacity + 4.2)) * float3(1.0, 0.875, 0.6);
		#ifndef IS_BLOOM_PASS	
			cloudSteps 							= 8.0 - saturate(NoLflat) * 7.0;
		#else
			cloudSteps 							= 1.0;
		#endif
		float2 cloudDir 						= (lightTS.xy / lightTS.z);
		shadowCloud.a							= 1.0 - CloudSim(uv + cloudDir, duvx, duvy, cloudSpeed, cloudDensity, cloudCuttoff);
		shadow 									*= shadowCloud.a;
		
		cloudDir								*= 0.05;
		cloudUV 								+= cloudDir;
	
	
		if(NoLflat > -0.25)
		{
			for(int k = 0; k < cloudSteps; k++)
			{			
				cloudSum 						-= CloudSim(cloudUV, duvx, duvy, cloudSpeed, cloudDensity, cloudCuttoff) * saturate(-0.4 * NoLflat + 0.5) * saturate(cloudSteps - k) * rayleighColor;
				cloudUV 						+= cloudDir;
			}		
			shadowCloud.rgb 					*= exp(min(0., cloudSum));
		}
		else
		{
			shadowCloud 						= 0;
		}
		cloudColor.rgb							= pow(cloudColor.rgb, 1 - cloudOpacity * 0.1);	
	}

	void GetSmoke(	inout float4 shadowCloud, 
					inout float3 cloudNormal, 
					inout float cloudOpacity,
					inout float3 cloudColor,
					inout float ao, 
					inout float shadow, 
					float NoLflat, 
					float cloudSteps, 
					float2 uv, 
					float2 uvFlat, 
					float2 parallaxTS, 
					float3 lightTS, 
					float2 duvx, 
					float2 duvy, 
					float3 smokeColor, 
					float3 rayleighColor)
	{
		const float smokeHeight 			= 0.05;
		const float smokeBaseDensity 		= 4.0;
		const float smokePuffDensity 		= 4.0;
		const float smokePuffScale			= 3.0;
		float cloudOpacityInv				= 1 - cloudOpacity;

		float2 smokeUV 						= uvFlat - parallaxTS * smokeHeight;
		#ifndef IS_BLOOM_PASS
			#ifdef SUPPORT_PARALLAX_DISPLACEMENT	
				float smokeDisplacement 	= VolcanoSmoke(smokeUV, duvx, duvy, smokeBaseDensity, smokePuffDensity, smokePuffScale);
				smokeUV						-= parallaxTS * ((smokeDisplacement - 0.5) * 0.08);
			#endif
	
			cloudNormal.r					+= (VolcanoSmoke(smokeUV - float2(0.0004, 0.0), duvx, duvy, smokeBaseDensity, smokePuffDensity, smokePuffScale) - VolcanoSmoke(smokeUV + float2(0.0004, 0.0), duvx, duvy, smokeBaseDensity, smokePuffDensity, smokePuffScale));
			cloudNormal.y					+= (VolcanoSmoke(smokeUV - float2(0.0, 0.0002), duvx, duvy, smokeBaseDensity, smokePuffDensity, smokePuffScale) - VolcanoSmoke(smokeUV + float2(0.0, 0.0002), duvx, duvy, smokeBaseDensity, smokePuffDensity, smokePuffScale));
		#endif
		float smokeOpacity 					= VolcanoSmoke(smokeUV, duvx, duvy, smokeBaseDensity, smokePuffDensity, smokePuffScale);

		ao									*= Square(saturate(1-VolcanoSmoke(uv, duvx, duvy, smokeBaseDensity, smokePuffDensity, smokePuffScale) * 0.75 * cloudOpacityInv + smokeOpacity));		

		float3 smokeSum 					= smokeOpacity;
		float2 cloudDir						= lightTS.xy / lightTS.z;
		float shadowSmokeGround				= 1-VolcanoSmoke(uv + cloudDir, duvx, duvy, smokeBaseDensity, smokePuffDensity, smokePuffScale) * cloudOpacityInv;
		shadowCloud.a						*= shadowSmokeGround;
		shadow 								*= shadowSmokeGround;

		cloudDir							*= 0.05;
		smokeUV 							+= cloudDir;

		if(NoLflat > -0.25)
		{
			for(int l = 0; l < cloudSteps; l++)
			{			
			
				smokeSum 					-= VolcanoSmoke(smokeUV, duvx, duvy, smokeBaseDensity, smokePuffDensity, smokePuffScale) * saturate(-0.4 * NoLflat + 0.5) * saturate(cloudSteps - l) * rayleighColor;
				smokeUV 					+= cloudDir;
			}
			shadowCloud.rgb 				*= (exp(min(0., smokeSum)));
		
		}
		cloudOpacity 						= max(cloudOpacity, smokeOpacity);	
		cloudColor.rgb						= lerp(cloudColor.rgb, pow(smokeColor.rgb, 1 - smokeOpacity * 0.1), smokeOpacity * cloudOpacityInv);
	}
	
	float3 GetLightning(float lightingMask, float3 normal, float3 normalSphere)
	{
		float4 lightingStrikeChance			= Hash41(g_Time);
	
		if(lightingStrikeChance.w > g_MaterialAmbient.a)
		{
			float3 lightingStrikePos 		= normalize(lightingStrikeChance.xyz - 0.5);
			
			float lightingStrikeNoL 		= saturate(dot(normal, lightingStrikePos));
			float lightingStrikeDist 		= sqrt(1 - saturate(dot(normalSphere, lightingStrikePos)));
			
			return (LIGHTNING_INTENSITY * lightingMask * lightingStrikeNoL * rcp(SF + Square(lightingStrikeDist * 300 * (0.2 + dot(abs(lightingStrikeChance), (float4)0.2))))) * Square(g_MaterialDiffuse.rgb);
		}
		return 0;
	}
	
	void GetAmbientBoost(inout float3 diffuseSample, inout float3 reflectionSample)
	{
		#ifdef AMBIENT_BOOST			
			diffuseSample						= 1 - exp(-(diffuseSample		+ Square(diffuseSample)		* AMBIENT_BOOST));
			reflectionSample  					= 1 - exp(-(reflectionSample  	+ Square(reflectionSample)	* AMBIENT_BOOST));
		#endif
	}
	
	void GetTangentBasis(float2 duvx, float2 duvy, float3 pos, float3 normalSphere, float3 tangentMesh, inout float3 tangent, inout float3 cotangent)
	{
		#ifdef USE_MATHEMATICALY_PERFECT_UV
			float3 dpx								= ddx(pos);
			float3 dpy								= ddy(pos);
			float3 dpxp								= cross(normalSphere, dpx);
			float3 dpyp								= cross(dpy, normalSphere);
			tangent 								= dpyp * duvx.x + dpxp * duvy.x;
			cotangent 								= dpyp * duvx.y + dpxp * duvy.y;
			float tcNormalize						= pow(max(dot(tangent, tangent), dot(cotangent, cotangent)), -0.5);
			tangent									*= tcNormalize;
			cotangent 								*= tcNormalize; 
		#else
			tangent 								= normalize(tangentMesh);
			cotangent								= normalize(cross(normalSphere, tangent));
		#endif
	}
/*

*/
//////////////////////////////////////////////////////////////////// common macro ///////////////////////////////////////////////////////////
#define MACRO_COMMON_BASE                                                                                                                  	\
																																			\
 	float3 normal 							= normalize(input.normal);                                                                      \
	float3 normalSphere 					= normal;                                                                                       \
	float3 posObj							= input.posObj; 																				\
	float3 pos 								= input.pos;                                                                                    \
	float3 lightPos							= g_Light0_Position - pos;                                                                      \
	float3 view 							= normalize(-pos);                                                                              \
																																			\
	float lightSqr							= dot(input.lightPos, input.lightPos);                                                          \
	float lightDist							= sqrt(lightSqr);                                                                               \
	float3 light 							= input.lightPos / lightDist;	                                                                \
																																			\
	float2 duvx;                                                                                                                            \
	float2 duvy;                                                                                                                            \
	float3 normalObj						= normalize(posObj);                                                                   			\
	float2 uv								= input.texCoord;                                                                               \
																																			\
	GetEquirectangularUV(normalObj, duvx, duvy, uv);                    																	\
																																			\
	float3 tangent;																															\
	float3 cotangent;																														\
	GetTangentBasis(duvx, duvy, pos, normalSphere, input.tangent, tangent, cotangent);														\
																																			\
	float2 uvFlat							= uv;                                                                                           \
	const float gravity 					= rcp(g_MaterialGlossiness / 6000.0);                                        					\
	float  parallaxScale					= PARALLAX_SCALE * gravity;				                                                                \
	float2 parallaxTS						= ParallaxVector(ToTangentSpace(float3x3(tangent, cotangent, normal), view), parallaxScale);	\
	float3 lightTS							= ToTangentSpace(float3x3(tangent, cotangent, normal), light);                                  \
	lightTS									= normalize(float3(lightTS.xy, (1.0 + lightTS.z) * rcp(parallaxScale)));       					\
																																			\
	float3 lightColor 						= SRGBToLinear(g_Light0_DiffuseLite.rgb) * PI;                                                  \
	float4 cloudColor						= SRGBToLinear(g_CloudColor);                                                                   \
	float3 rayleighColor					= g_MaterialDiffuse.rgb;																		\
																																			\
	float NoV								= abs(dot(view, normal));                                                                       \
	parallaxTS								*= NoV;                                                                                         \
	float NoL 								= dot(normal, light);                                                                           \
	float NoLflat							= NoL;                                                                                          \
																																			\
	float poleDensity						= Pow5(abs(uv.y * 2.0 - 1.0));                                                                  \
	float mipBias							= poleDensity * 7.0 + 1.0;                                                                      \
																																			\
	duvx									*= mipBias;                                                                                     \
	duvy									*= mipBias;                                                                                     \
//	return float4(normalize(cross(ddx(pos), ddy(pos))), 1);                                                                                 \
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MACRO_COMMON_LIGHTING                                                                                                                                                                                                           	\
	float3 atmosphereAbsorption				= 0;                                                                                                                                                                                        	\
	[branch]                                                                                                                                                                                                                            	\
	if(NoLflat > -0.25)                                                                                                                                                                                                                 	\
	{                                                                                                                                                                                                                                   	\
		atmosphereAbsorption				= lerp(0.075 - cloudOpacity * 0.05, Square(saturate(-NoLflat + 0.25)), saturate(pow(shadowCloud.rgb, 0.25))) * 200 * rayleighColor.rgb;                                                  		\
		atmosphereAbsorption 				= saturate(exp(-max((float3)0, float3(RayleighPhaseFunction(atmosphereAbsorption.r),	RayleighPhaseFunction(atmosphereAbsorption.g),	RayleighPhaseFunction(atmosphereAbsorption.b)))));		\
	}                                                                                                                                                                                                                                   	\
																																																											\
	NoL										= dot(normal, light);                                                                                                                                                                       	\
	NoV										= dot(normal, view);	                                                                                                                                                                    	\
																																																											\
	float lightRad							= 25000;                                                                                                                                                                                    	\
	float3 specular							= 0;                                                                                                                                                                                        	\
	float3 diffuse							= 0;                                                                                                                                                                                        	\
																																																											\
	Properties.Roughness					= saturate(Properties.Roughness + (1-shadowCloud.a) * 0.5 + cloudOpacity);                                                                                                                        	\
	float attenuation						= 1;                                                                                                                                                                                        	\
	GetMainStarLight(Properties, normal, view, light, lightPos, lightRad, float4(lightColor.rgb, 0.1), atmosphereAbsorption, specular, diffuse, shadow, attenuation, NoL);                                                              	\
																																																											\
	float3 directLight 						= lerp(specular + diffuse, (cloudColor * lightColor.rgb * atmosphereAbsorption) * (saturate(dot(cloudNormal, light) * 0.8 + 0.2) * attenuation), fog) * shadowCloud.rgb;                    	\
																																																											\
	specular								= 0;                                                                                                                                                                                        	\
	diffuse									= 0;                                                                                                                                                                                        	\
	float3 reflection						= -(view - 2.0 * normal * NoV);                                                                                                                                                             	\
	float3 reflectionSample 				= SRGBToLinear(texCUBElod(TextureEnvironmentCubeSampler, float4(reflection, GetMipRoughness(Properties.Roughness, 5.0)/*max(cloudShadow * 6.0, Properties.RoughnessMip)*/))).rgb;           	\
	float3 diffuseSample					= SRGBToLinear(texCUBE(EnvironmentIlluminationCubeSampler, normal)).rgb;                                                                                                                    	\
	reflectionSample						*= Properties.AO;                                                                                                                                                                           	\
	diffuseSample							*= Properties.AO;                                                                                                                                                                           	\
	GetAmbientBoost(diffuseSample, reflectionSample);                                                                                                                                                                                   	\
	Properties.SpecularColor				= lerp(Properties.SpecularColor, 0.0233, fog);                                                                                                                                              	\
	Properties.DiffuseColor					= lerp(Properties.DiffuseColor, cloudColor, fog);                                                                                                                                          		\
	AmbientBRDF(diffuse, specular, saturate(abs(NoV)), Properties, reflectionSample, diffuseSample, true);                                                                                                                              	\
																																																											\
	float3 ambientLight 					= diffuse + diffuse;                                                                                                                                                                        	\
																																																											\
	return float4(LinearToSRGB(1.0 - exp(-(directLight + ambientLight + Properties.EmissiveColor.rgb))), 1);                                                                                                                            	\
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                                                                                             
float4 RenderSceneGasGiantPS(VsSceneOutput input) : COLOR0                                                                                                                                                                                   
{                                                                                                                                                                                                                                            
#ifdef DEBUG_PLANET_MODES                                                                                                                                                                                                                    
	return float4(0.75, 0.75, 0.75, 1.0);                                                                                                                                                                                                    
#endif                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                             
	MACRO_COMMON_BASE	                                                                                                                                                                                                                     
	#ifdef COMPARE_UV_X
		return float4(uv.x, input.texCoord.x, 0, 1);
	#endif
	#ifdef COMPARE_UV_Y
		return float4(uv.y, input.texCoord.y, 0, 1);
	#endif		                                                                                                                                                                                                                                     
	float jacobiFalloff					= saturate(Square(normal.y)) * 0.499 + 0.5;		                                                                                                                                                     
	float speed							= g_Time * g_MaterialSpecular.g;                                                                                                                                                                     
	int iterations						= 4;                                                                                                                                                                                                 
	float2 uvFlowA						= uv;
	float2 uvFlowB						= uv;
	#ifndef IS_BLOOM_PASS
		const float2 density			= rcp(float2(32.0 * (1.0 - abs(uv.y - 0.5)) + 1.0, 32.0));		
		float forceFalloff				= 1.0;		
		float swapA						= frac(speed);
		float swapB						= frac(speed + 0.5);
		
		[unroll]
		for(int i = 0; i < iterations; ++i)
		{
			uvFlowA 					-= (tex2Dgrad(TextureDisplacementSampler, uvFlowA, duvx, duvy).xy - 0.5) * (density * forceFalloff * swapA);
			uvFlowB 					-= (tex2Dgrad(TextureDisplacementSampler, uvFlowB, duvx, duvy).xy - 0.5) * (density * forceFalloff * swapB);
			forceFalloff 				*= jacobiFalloff;
		}
	
		float gasBlend					= abs(swapA * 2.0 - 1.0);
		float4 gasSample				= MapGasGiant(lerp(tex2Dgrad(TextureColorSampler, uvFlowA, duvx, duvy), tex2Dgrad(TextureColorSampler, uvFlowB, duvx, duvy), gasBlend));
	
		uvFlowA							+= parallaxTS * (gasSample.g - 0.5);
		uvFlowB							+= parallaxTS * (gasSample.g - 0.5);
	
		gasSample 						= MapGasGiant(lerp(tex2Dgrad(TextureColorSampler, uvFlowA, duvx, duvy), tex2Dgrad(TextureColorSampler, uvFlowB, duvx, duvy), gasBlend));
	#else
		float4 gasSample				= tex2Dgrad(TextureColorSampler, uv, duvx, duvy);
	#endif
	float ao							= Square(gasSample.b) * 0.95 + 0.05;
	
	float2 uvColor						= float2(frac(speed * 0.01), saturate((uv.y - 0.5) + (gasSample.a - 0.5) * 0.2 + 0.5));
	float3 gasColor						= SRGBToLinear(pow(tex2Dlod(TextureNormalSampler, float4(uvColor, 0.0, 0.0)).rgb, (float3)(1.5 - gasSample.r)));
	
	#ifndef IS_BLOOM_PASS
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
		
		blendedNormals.xy					= blendedNormals.xz - blendedNormals.yw;
		blendedNormals.z					= DeriveZ(blendedNormals.xy);
					
		normal								= normalize(ToWorldSpace(float3x3(tangent, cotangent, normal), blendedNormals.xyz));
	#endif
	
	float3 diffuse						= gasColor * SRGBToLinear(texCUBElod(EnvironmentIlluminationCubeSampler, float4(normal, 0.0))).rgb * ao;
	float3 lightSum 					= diffuse;
	NoV									= dot(normal, view);
	float3 reflection					= -(view - 2.0 * normal * NoV);                                                                                                                                                                  \
	float3 reflectionSample 			= SRGBToLinear(texCUBElod(TextureEnvironmentCubeSampler, float4(reflection, 5.0))).rgb;
	lightSum							+= reflectionSample * AmbientDielectricBRDF(1.0, saturate(NoV));
				
	float cloudAbsorption				= gasSample.g;
	#ifndef IS_BLOOM_PASS
	
		float steps						= 32 - 31 * saturate(NoL);
		float shadowBlend				= (1.0 - saturate(NoL)) * steps;
		float gasDensity				= 2.0;
		
		[branch]
		if(NoL > -0.1)
		{
			float stepsInv				= 1.0 / steps;	
			float2 shadowStep			= lightTS.xy * stepsInv;		
			
			float NoLinv				= rcp(1.0 + NoL * steps);
			
			[loop]
			for(int g = 0; g < shadowBlend; g++)
			{
				uvFlowA					+= shadowStep;
				uvFlowB					+= shadowStep;
				cloudAbsorption			-= lerp(0.5, lerp(tex2Dgrad(TextureColorSampler, uvFlowA, duvx, duvy).g, tex2Dgrad(TextureColorSampler, uvFlowB, duvx, duvy).g, gasBlend), g_MaterialSpecular.r) * saturate(shadowBlend - g) * gasDensity;
			}			
			cloudAbsorption				= exp(min(0.0, cloudAbsorption * stepsInv));
		}			
		else			
		{			
			cloudAbsorption				= 0.0;
		}
	#else
		cloudAbsorption					= saturate(cloudAbsorption + (NoL * 3.0 - 2.0));
	#endif
	float3 absorptionColor				= lerp(0.075 + 0.025 * gasSample.g, Square(saturate(dot(-light, normal) + 0.25)), Square(cloudAbsorption)) * 200.0 * rayleighColor;
	absorptionColor						= saturate(exp(-max((float3)0.0, RayleighPhaseFunction(absorptionColor.rgb))));

	float attenuation				= 1;
	float lightFalloff				= rcp(1.0 + lightSqr);//regular pointlight
		
	light							= lightPos / lightDist;
	NoL								= dot(normal, light);
	#ifdef SUPPORT_SPHERE_LIGHT
	
		float sphereSinAlphaSqr			= saturate(Square(lightRad) * lightFalloff);

		// Patch to Sphere frontal equation ( Quilez version )
		float lightRadSqr 				= Square(lightRad);
		// Do not allow object to penetrate the light ( max )
		// Form factor equation include a (1 / FB_PI ) that need to be cancel
		// thus the " FB_PI *"
		float illuminance 				= PI * (lightRadSqr / (max(lightRadSqr, lightSqr))) * saturate(NoL);
		attenuation 					*= GetAttenuation(illuminance);
	#else
		attenuation 					*= GetAttenuation(lightFalloff);
	#endif
	NoL 								= saturate(NoL);

	lightSum							+= (gasColor * absorptionColor * lightColor) * (cloudAbsorption * NoL * attenuation * ao);
	
	return LinearToSRGB(float4(1.0 - exp(-lightSum), 1.0));
}

float4 RenderSceneOceanPS(VsSceneOutput input) : COLOR0
{

#ifdef DEBUG_PLANET_MODES
	return float4(0.0, 0.0, 1.0, 1.0);
#endif	
	
	MACRO_COMMON_BASE
	#ifdef COMPARE_UV_X
		return float4(uv.x, input.texCoord.x, 0, 1);
	#endif
	#ifdef COMPARE_UV_Y
		return float4(uv.y, input.texCoord.y, 0, 1);
	#endif	
	float shadow							= 1.0;	
	GetParallaxBase(uv, shadow, parallaxTS, lightTS, duvx, duvy, NoL, g_MaterialSpecular.r);
	
	float4 shadowCloud						= 1;
	float ao								= 1;	
	float cloudOpacity 						= 0;
	float3 cloudNormal 						= 0;
	float cloudSteps;

	GetClouds(shadowCloud, cloudNormal, cloudOpacity, ao, shadow, poleDensity, NoLflat, cloudSteps, posObj, uv, uvFlat, parallaxTS, lightTS, duvx, duvy, cloudColor, rayleighColor);
	#ifdef DEBUG_CLOUD_COLORS
		return float4(cloudColor.rgb, 1);	
	#endif
	float fog 								= 0.0;
	float fogDistance;
	float height							= tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).a;

	float waterMask							= 0.0;
	float waterDepth						= 0.0;
	float waterBlur							= 0.0;
	float waterRippleSpeed					= 4 * g_Time;
	float waterRippleIntensity				= 0.05;
	float3 waterNormal						= normalSphere;
	float2 waterRefraction					= 0;
	
	float waterGrad 					= height - g_MaterialSpecular.r;
	#ifndef IS_BLOOM_PASS
		#if 0
			float shoreLine					= saturate(1-abs(-waterGrad * 128.0));
			float2 shoreDistort;
			sincos(tex2Dbias(CloudLayerSampler, float4(uv * 6.0, 0, poleDensity)).r * TWOPI - PI, shoreDistort.x, shoreDistort.y);
			waterRefraction 				+= shoreDistort * rcp(float2(2048, 1024)) * shoreLine;
			height 							= tex2Dbias(TextureDisplacementSampler, float4(uv + waterRefraction, 0.0, 0.0)).a;
			waterGrad 						= height - g_MaterialSpecular.r;
		#endif
	#endif
	waterMask 							= saturate((-waterGrad * 256.0));
	
	fogDistance 						= -(1-saturate((waterGrad) * rcp(1-g_MaterialSpecular.r)));
	
	if(waterMask > 0)
	{
		waterDepth						= 1.0 - saturate(exp(waterGrad * g_MaterialSpecular.g * 256.0));		
		waterBlur						= Square(waterDepth) * 16.0;
		#ifndef IS_BLOOM_PASS
			#ifdef SUPPORT_OCEAN_RIPPLES
				float2 uvWater 				= uvFlat + parallaxTS * (1.0 - g_MaterialSpecular.r);
				float3 waterPos				= GetEqurectangularPos(uvWater);
				
				#ifdef SUPPORT_SHORE_WAVES
					float shoreBlur			= 1.0;
					float shoreScale		= rcp(g_MaterialSpecular.r);
					for(int h = 0; h < 8; h++)
					{	
						shoreBlur 			*= tex2Dbias(TextureDisplacementSampler, float4(uv, 0.0, h)).a * shoreScale;
						waterPos			*= 1.0 + shoreBlur * 0.005;
					}
				#endif
				
				float4 waterRipples			= Nnoise(waterPos * g_MaterialGlossiness * 0.1, waterRippleSpeed);
				waterNormal					= normalize(mul(normalObj + waterRipples.xyz * waterRippleIntensity * rcp(length(pos * 0.0001)+1), (float3x3)g_World));
				waterRefraction				= ToTangentSpace(float3x3(tangent, cotangent, normal), refract(view, waterNormal, 1.33)).xy * rcp(float2(2048, 1024)) * waterMask;		
			#endif
		#endif
	}

	fog										= 1-exp(fogDistance * Pow4(1-g_MaterialSpecular.b) * (1 + Pow3(1-NoV) * FOG_ANGLE_SCATTER) * FOG_SCALE);	
	float blurMipBiasAboveWater				= max(fog, cloudOpacity * 16.0);
	fog										= max(fog, cloudOpacity);
	
	float blurMipBias						= max(blurMipBiasAboveWater, waterBlur);

	float4 sampleA							= 0.0;
	float4 sampleB							= 0.0;
	float4 sampleC							= 0.0;
	
	float weightBlurSum						= 0;
	
	#ifdef DETAILED_BLURS
	//fog and water blur in one go
	for(int m = -1; m < blurMipBias; m++)
	{	
		float weightBlur					= saturate(blurMipBias - m);
		weightBlurSum						+= weightBlur;
		sampleA								+= tex2Dbias(TextureColorSampler, 	float4(uv + waterRefraction, 0, m)) * weightBlur;
		sampleB								+= tex2Dbias(TextureDataSampler, 	float4(uv + waterRefraction, 0, m)) * weightBlur;
		sampleC								+= tex2Dbias(TextureNormalSampler, 	float4(uv + waterRefraction, 0, m)) * weightBlur;
	}
	#else
		sampleA								+= tex2Dbias(TextureColorSampler, 	float4(uv + waterRefraction, 0, blurMipBias));
		sampleB								+= tex2Dbias(TextureDataSampler, 	float4(uv + waterRefraction, 0, blurMipBias));
		sampleC								+= tex2Dbias(TextureNormalSampler, 	float4(uv + waterRefraction, 0, blurMipBias));		
	#endif
	
	float dayMask							= smoothstep(0.2, 0.0, NoL);	
	
	
	#ifdef DETAILED_BLURS
		sampleA								/= weightBlurSum;
		sampleB                             /= weightBlurSum;
		sampleC                             /= weightBlurSum;
	#endif
	
	normal 									= normalize(ToWorldSpace(float3x3(tangent, cotangent, normalSphere), GetNormalDXT5(sampleC)));
	sampleA.rgb 							= SRGBToLinear(sampleA.rgb);	
	sampleB.rgb 							= SRGBToLinear(sampleB.rgb);
	
	float3 sampleS							= 0.0;
	
	//subsurface
	for(int o = 0; o < 4; o++)
	{	
		sampleS								+= tex2Dbias(TextureColorSampler, 	float4(uv, 0, 2.5 + o)).rgb;
	}
	sampleS									= SRGBToLinear(sampleS * 0.25);
	
	#ifndef IS_BLOOM_PASS
		//atmosphere glow
		if(dayMask > 0)
		{
			for(float p = 0; p < 8; p += 1)
			{
				float mip 					= blurMipBias + p;
				sampleB.rgb 				+= SRGBToLinear(tex2Dbias(TextureDataSampler, float4(lerp(uv + waterRefraction, uvFlat, Square(saturate(p * 0.125))), 0.0, mip)).rgb);	
			}
		}
	#endif
	sampleB.rgb								*= dayMask;
//	if(g_MaterialSpecular.r > 0)
//	{
		normal 								= lerp(normal, waterNormal, waterMask);
		sampleB.w 							= lerp(sampleB.w, (0.5 - 0.45 * rcp(1+length(-pos) * 0.00005)), waterMask);
					
		float3 oceanColor					= SRGBToLinear(g_MaterialAmbient.rgb);
		sampleB.rgb							*= waterDepth * oceanColor + (1-waterDepth);
		sampleS								= lerp(sampleS, oceanColor, waterMask);
		sampleA								= lerp(sampleA, float4(lerp(pow(oceanColor, waterDepth * float3(1.0, 0.4, 0.45)) * pow(sampleA.rgb, 1 + Luminance(sampleA.rgb)), pow(oceanColor, 1 + NoV * 0.5), saturate(waterDepth * (1.5 - NoV))), 0.23), waterMask);
		sampleC.a							= lerp(sampleC.a, waterDepth * 0.1 * (Pow5(1-NoV) + 0.5), waterMask);
//	}
	sampleB.rgb								*= EMISSIVE_INTENSITY;
	sampleB.rgb 							*= exp(-RayleighPhaseFunction(fog * rayleighColor.rgb * CLOUD_EMISSIVE_ABSORPTION)) * (1-cloudOpacity);
	fog										*= fog;
	
	PBRProperties Properties;	
	Properties.Roughness					= sampleB.w;
	Properties.EmissiveColor				= float4(sampleB.rgb, 0);
	Properties.SpecularColor				= lerp(sampleA.a * 0.08, sampleA.rgb, sampleC.b);
	Properties.DiffuseColor					= sampleA.rgb * (1-sampleC.b);
	Properties.AO 							= ao;
	Properties.SubsurfaceColor 				= sampleS;
	Properties.SubsurfaceOpacity 			= sampleC.r;

	#ifndef IS_BLOOM_PASS
		cloudNormal.z						= DeriveZ(cloudNormal.xy);
		cloudNormal							= normalize(ToWorldSpace(float3x3(tangent, cotangent, normalSphere), cloudNormal));
		
		normal								= normalize(lerp(normal, cloudNormal, max(fog, cloudOpacity)));
	#else
		normal								= normalize(lerp(normal, normalSphere, max(fog, cloudOpacity)));
	#endif
	
	if(g_MaterialAmbient.a < 1)
	{
		Properties.EmissiveColor.rgb 		+= GetLightning(cloudOpacity, normal, normalSphere) * lerp((float3)1, oceanColor, waterDepth);
	}
	MACRO_COMMON_LIGHTING
}

float4 RenderSceneNoAtmospherePS(VsSceneOutput input) : COLOR0
{
#ifdef DEBUG_PLANET_MODES
	return float4(0.1, 0.1, 0.1, 1.0);
#endif	
	
	MACRO_COMMON_BASE
	#ifdef COMPARE_UV_X
		return float4(uv.x, input.texCoord.x, 0, 1);
	#endif
	#ifdef COMPARE_UV_Y
		return float4(uv.y, input.texCoord.y, 0, 1);
	#endif

	float shadow							= 1.0;	
	GetParallaxBase(uv, shadow, parallaxTS, lightTS, duvx, duvy, NoL, g_MaterialSpecular.r, false);
	#ifdef COMPARE_UV
		return float4(frac(uv * 128), 0, 1);
	#endif	
	float4 shadowCloud						= 1;
	float ao								= 1;	
	float cloudOpacity 						= 0;
	
	float fog 								= 0.0;
	float fogDistance;
	float height							= tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).a;

	fogDistance = -(1-height);
	
	fog										= 1-exp(fogDistance * Pow4(1-g_MaterialSpecular.b) * (1 + Pow3(1-NoV) * FOG_ANGLE_SCATTER) * FOG_SCALE);	
	float blurMipBiasAboveWater				= max(fog, cloudOpacity * 16.0);
	fog										= max(fog, cloudOpacity);
	
	float blurMipBias						= blurMipBiasAboveWater;

	float4 sampleA							= 0.0;
	float4 sampleB							= 0.0;
	float4 sampleC							= 0.0;
	
	float weightBlurSum						= 0;
	
	#ifdef DETAILED_BLURS
	//fog and water blur in one go
	for(int m = -1; m < blurMipBias; m++)
	{	
		float weightBlur					= saturate(blurMipBias - m);
		weightBlurSum						+= weightBlur;
		sampleA								+= tex2Dbias(TextureColorSampler, 	float4(uv, 0, m)) * weightBlur;
		sampleB								+= tex2Dbias(TextureDataSampler, 	float4(uv, 0, m)) * weightBlur;
		sampleC								+= tex2Dbias(TextureNormalSampler, 	float4(uv, 0, m)) * weightBlur;
	}
	#else
		sampleA								+= tex2Dbias(TextureColorSampler, 	float4(uv, 0, blurMipBias));
		sampleB								+= tex2Dbias(TextureDataSampler, 	float4(uv, 0, blurMipBias));
		sampleC								+= tex2Dbias(TextureNormalSampler, 	float4(uv, 0, blurMipBias));		
	#endif
	
	float dayMask							= smoothstep(0.2, 0.0, NoL);	
	
	#ifdef DETAILED_BLURS
		sampleA								/= weightBlurSum;
		sampleB                             /= weightBlurSum;
		sampleC                             /= weightBlurSum;
	#endif	
	normal 									= normalize(ToWorldSpace(float3x3(tangent, cotangent, normalSphere), GetNormalDXT5(sampleC)));
	sampleA.rgb 							= SRGBToLinear(sampleA.rgb);	
	sampleB.rgb 							= SRGBToLinear(sampleB.rgb);
	
	float3 sampleS							= 0.0;
	
	//subsurface
	for(int o = 0; o < 4; o++)
	{	
		sampleS								+= tex2Dbias(TextureColorSampler, 	float4(uv, 0, 2.5 + o)).rgb;
	}
	sampleS									= SRGBToLinear(sampleS * 0.25);
	
	#ifndef IS_BLOOM_PASS
		//atmosphere glow
		if(dayMask > 0)
		{
			for(float p = 0; p < 8; p += 1)
			{
				float mip 						= blurMipBiasAboveWater + p;
				sampleB.rgb 					+= SRGBToLinear(tex2Dbias(TextureDataSampler, float4(lerp(uv, uvFlat, Square(saturate(p * 0.125))), 0.0, mip)).rgb);	
			}
		}
	#endif
	sampleB.rgb								*= dayMask;
	sampleB.rgb								*= EMISSIVE_INTENSITY;
	sampleB.rgb 							*= exp(-RayleighPhaseFunction(fog * g_MaterialDiffuse.rgb * CLOUD_EMISSIVE_ABSORPTION)) * (1-cloudOpacity);
	fog										*= fog;
	
	PBRProperties Properties;	
	Properties.Roughness					= sampleB.w;
	Properties.EmissiveColor				= float4(sampleB.rgb, 0);
	Properties.SpecularColor				= lerp(sampleA.a * 0.08, sampleA.rgb, sampleC.b);
	Properties.DiffuseColor					= sampleA.rgb * (1-sampleC.b);
	Properties.AO 							= ao;
	Properties.SubsurfaceColor 				= sampleS;
	Properties.SubsurfaceOpacity 			= sampleC.r;
	
	normal									= normalize(lerp(normal, normalSphere, max(fog, cloudOpacity)));
	float3 cloudNormal						= normalSphere;
	MACRO_COMMON_LIGHTING
}

float4 RenderSceneVolcanoPS(VsSceneOutput input) : COLOR0
{

#ifdef DEBUG_PLANET_MODES
	return float4(1.0, 0.5, 0.0, 1.0);
#endif	
	
	MACRO_COMMON_BASE
	#ifdef COMPARE_UV_X
		return float4(uv.x, input.texCoord.x, 0, 1);
	#endif
	#ifdef COMPARE_UV_Y
		return float4(uv.y, input.texCoord.y, 0, 1);
	#endif	
	float shadow							= 1.0;	
	GetParallaxBase(uv, shadow, parallaxTS, lightTS, duvx, duvy, NoL, g_MaterialSpecular.r);
	
	float4 shadowCloud						= 1;
	float ao								= 1;	
	float cloudOpacity 						= 0;
	float3 cloudNormal 						= 0;
	float cloudSteps;
	GetClouds(shadowCloud, cloudNormal, cloudOpacity, ao, shadow, poleDensity, NoLflat, cloudSteps, posObj, uv, uvFlat, parallaxTS, lightTS, duvx, duvy, cloudColor, rayleighColor);
	float3 smokeColor						= SRGBToLinear(g_MaterialAmbient.rgb);
	GetSmoke(shadowCloud, cloudNormal, cloudOpacity, cloudColor.rgb, ao, shadow, NoLflat, cloudSteps, uv, uvFlat, parallaxTS, lightTS, duvx, duvy, smokeColor, rayleighColor);
	#ifdef DEBUG_CLOUD_COLORS
		return float4(cloudColor.rgb, 1);	
	#endif

	float fog 								= 0.0;
	float fogDistance;
	float height							= tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).a;
	fogDistance								= -(1-height);

	fog										= 1-exp(fogDistance * Pow4(1-g_MaterialSpecular.b) * (1 + Pow3(1-NoV) * FOG_ANGLE_SCATTER) * FOG_SCALE);	
	float blurMipBiasAboveWater				= max(fog, cloudOpacity * 16.0);
	fog										= max(fog, cloudOpacity);
	
	float blurMipBias						= blurMipBiasAboveWater;

	float4 sampleA							= 0.0;
	float4 sampleB							= 0.0;
	float4 sampleC							= 0.0;
	
	float weightBlurSum						= 0;
	
	#ifdef DETAILED_BLURS
	//fog and water blur in one go
	for(int m = -1; m < blurMipBias; m++)
	{	
		float weightBlur					= saturate(blurMipBias - m);
		weightBlurSum						+= weightBlur;
		sampleA								+= tex2Dbias(TextureColorSampler, 	float4(uv, 0, m)) * weightBlur;
		sampleB								+= tex2Dbias(TextureDataSampler, 	float4(uv, 0, m)) * weightBlur;
		sampleC								+= tex2Dbias(TextureNormalSampler, 	float4(uv, 0, m)) * weightBlur;
	}
	#else
		sampleA								+= tex2Dbias(TextureColorSampler, 	float4(uv, 0, blurMipBias));
		sampleB								+= tex2Dbias(TextureDataSampler, 	float4(uv, 0, blurMipBias));
		sampleC								+= tex2Dbias(TextureNormalSampler, 	float4(uv, 0, blurMipBias));		
	#endif
	
	float dayMask;							
	
	#ifndef IS_BLOOM_PASS
		//Animate the lava flow!
		float lavaMask							= Square(tex2Dbias(TextureDisplacementSampler, float4(uv, 0, 0)).r);		
		float2 flowSampleRaw 					= tex2Dbias(TextureNormalSampler, float4(uv, 0, 2)).yw * 2.0 - 1.0;
		flowSampleRaw 							= dot(flowSampleRaw.yx, flowSampleRaw.yx) > 0.0001 ? normalize(flowSampleRaw.yx) * rcp(float2(2048, 1024)) * lavaMask : 0;	
		float3 flowBase 						= GetFlow(g_Time * 0.25);
		sampleB.rgb								= 0;
		#ifdef DETAILED_BLURS	
			for(int n = 0; n < blurMipBias; n++)	
			{		
				float weightBlur				= saturate(blurMipBias - n);
				sampleB.rgb 					+= lerp(tex2Dbias(TextureDataSampler, float4(uv - flowSampleRaw * flowBase.x, 0, n)).rgb,
														tex2Dbias(TextureDataSampler, float4(uv - flowSampleRaw * flowBase.y, 0, n)).rgb, flowBase.z) * weightBlur;
			}
		#else
			sampleB.rgb 						+= lerp(tex2Dbias(TextureDataSampler, float4(uv - flowSampleRaw * flowBase.x, 0, blurMipBias)).rgb,
														tex2Dbias(TextureDataSampler, float4(uv - flowSampleRaw * flowBase.y, 0, blurMipBias)).rgb, flowBase.z);
		#endif
	#else
		sampleB.rgb 						+= tex2Dbias(TextureDataSampler, float4(uv, 0, blurMipBias)).rgb;		
	#endif	
	
	#ifdef DETAILED_BLURS
		sampleA								/= weightBlurSum;
		sampleB                             /= weightBlurSum;
		sampleC                             /= weightBlurSum;
	#endif
	
	normal 									= normalize(ToWorldSpace(float3x3(tangent, cotangent, normalSphere), GetNormalDXT5(sampleC)));
	sampleA.rgb 							= SRGBToLinear(sampleA.rgb);	
	sampleB.rgb 							= SRGBToLinear(sampleB.rgb);
	
	#ifndef IS_BLOOM_PASS
		//atmosphere glow
		for(float p = 0; p < 8; p += 1)
		{
			float mip 							= blurMipBiasAboveWater + p;
			sampleB.rgb 						+= SRGBToLinear(tex2Dbias(TextureDataSampler, float4(lerp(uv, uvFlat, Square(saturate(p * 0.125))), 0.0, mip)).rgb);	
		}
	#endif
	sampleB.rgb								*= LAVA_INTENSITY;
	sampleB.rgb 							*= exp(-RayleighPhaseFunction(fog * rayleighColor.rgb * CLOUD_EMISSIVE_ABSORPTION)) * (1-cloudOpacity);
	fog										*= fog;
	
	PBRProperties Properties;	
	Properties.Roughness					= sampleB.w;
	Properties.EmissiveColor				= float4(sampleB.rgb, 0);
	Properties.SpecularColor				= lerp(sampleA.a * 0.08, sampleA.rgb, sampleC.b);
	Properties.DiffuseColor					= sampleA.rgb * (1-sampleC.b);
	Properties.AO 							= ao;
	Properties.SubsurfaceColor 				= 0;
	Properties.SubsurfaceOpacity 			= 0;

	#ifndef IS_BLOOM_PASS
		cloudNormal.z							= DeriveZ(cloudNormal.xy);
		cloudNormal								= normalize(ToWorldSpace(float3x3(tangent, cotangent, normalSphere), cloudNormal));
		
		normal									= normalize(lerp(normal, cloudNormal, max(fog, cloudOpacity)));
	#else
		normal									= normalize(lerp(normal, normalSphere, max(fog, cloudOpacity)));
	#endif
	
	if(g_MaterialAmbient.a < 1)
	{
		Properties.EmissiveColor.rgb 		+= GetLightning(cloudOpacity, normal, normalSphere);
	}	
	
	MACRO_COMMON_LIGHTING
}

float4 RenderGreenhousePS(VsSceneOutput input) : COLOR0
{
	
#ifdef DEBUG_PLANET_MODES
	return float4(0.75, 1.0, 0, 1.0);

#endif	
	
	MACRO_COMMON_BASE
	#ifdef COMPARE_UV_X
		return float4(uv.x, input.texCoord.x, 0, 1);
	#endif
	#ifdef COMPARE_UV_Y
		return float4(uv.y, input.texCoord.y, 0, 1);
	#endif	
	float shadow							= 1.0;	
	GetParallaxBase(uv, shadow, parallaxTS, lightTS, duvx, duvy, NoL, g_MaterialSpecular.r);
	
	float4 shadowCloud						= 1;
	float ao								= 1;	
	float cloudOpacity 						= 0;
	float3 cloudNormal 						= 0;
	float cloudSteps;
	GetClouds(shadowCloud, cloudNormal, cloudOpacity, ao, shadow, poleDensity, NoLflat, cloudSteps, posObj, uv, uvFlat, parallaxTS, lightTS, duvx, duvy, cloudColor, rayleighColor);
	#ifdef DEBUG_CLOUD_COLORS
		return float4(cloudColor.rgb, 1);	
	#endif

	float fog 								= 0.0;
	float fogDistance;
	float height							= tex2Dgrad(TextureDisplacementSampler, uv, duvx, duvy).a;

	fogDistance = -(1-height);

	fog										= 1-exp(fogDistance * Pow4(1-g_MaterialSpecular.b) * (1 + Pow3(1-NoV) * FOG_ANGLE_SCATTER) * FOG_SCALE);	
	float blurMipBiasAboveWater				= max(fog, cloudOpacity * 16.0);
	fog										= max(fog, cloudOpacity);
	
	float blurMipBias						= blurMipBiasAboveWater;

	float4 sampleA							= 0.0;
	float4 sampleB							= 0.0;
	float4 sampleC							= 0.0;
	
	float weightBlurSum						= 0;
	
	#ifdef DETAILED_BLURS
	//fog and water blur in one go
	for(int m = -1; m < blurMipBias; m++)
	{	
		float weightBlur					= saturate(blurMipBias - m);
		weightBlurSum						+= weightBlur;
		sampleA								+= tex2Dbias(TextureColorSampler, 	float4(uv, 0, m)) * weightBlur;
		sampleB								+= tex2Dbias(TextureDataSampler, 	float4(uv, 0, m)) * weightBlur;
		sampleC								+= tex2Dbias(TextureNormalSampler, 	float4(uv, 0, m)) * weightBlur;
	}
	#else
		sampleA								+= tex2Dbias(TextureColorSampler, 	float4(uv, 0, blurMipBias));
		sampleB								+= tex2Dbias(TextureDataSampler, 	float4(uv, 0, blurMipBias));
		sampleC								+= tex2Dbias(TextureNormalSampler, 	float4(uv, 0, blurMipBias));		
	#endif
	
	float dayMask							= smoothstep(0.2, 0.0, NoL);	

	
	#ifdef DETAILED_BLURS
		sampleA								/= weightBlurSum;
		sampleB                             /= weightBlurSum;
		sampleC                             /= weightBlurSum;
	#endif	
	normal 									= normalize(ToWorldSpace(float3x3(tangent, cotangent, normalSphere), GetNormalDXT5(sampleC)));
	sampleA.rgb 							= SRGBToLinear(sampleA.rgb);	
	sampleB.rgb 							= SRGBToLinear(sampleB.rgb);
	
	#ifndef IS_BLOOM_PASS
	//atmosphere glow
		if(dayMask > 0)
		{
			for(float p = 0; p < 8; p += 1)
			{
				float mip 						= blurMipBiasAboveWater + p;
				sampleB.rgb 					+= SRGBToLinear(tex2Dbias(TextureDataSampler, float4(lerp(uv, uvFlat, Square(saturate(p * 0.125))), 0.0, mip)).rgb);	
			}
		}
	#endif
	sampleB.rgb								*= dayMask;
	sampleB.rgb								*= EMISSIVE_INTENSITY;
	sampleB.rgb 							*= exp(-RayleighPhaseFunction(fog * rayleighColor.rgb * CLOUD_EMISSIVE_ABSORPTION)) * (1-cloudOpacity);
	fog										*= fog;
	
	PBRProperties Properties;	
	Properties.Roughness					= sampleB.w;
	Properties.EmissiveColor				= float4(sampleB.rgb, 0);
	Properties.SpecularColor				= lerp(sampleA.a * 0.08, sampleA.rgb, sampleC.b);
	Properties.DiffuseColor					= sampleA.rgb * (1-sampleC.b);
	Properties.AO 							= ao;
	Properties.SubsurfaceColor 				= 0;
	Properties.SubsurfaceOpacity 			= 0;

	normal									= normalize(lerp(normal, normalSphere, max(fog, cloudOpacity)));
	
	if(g_MaterialAmbient.a < 1)
	{
		Properties.EmissiveColor.rgb 		+= GetLightning(1.0, normal, normalSphere);
	}	
	
	MACRO_COMMON_LIGHTING
}

///////////////////////////////////////////////atmosphere


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

		march_step *= (1.0 / SUN_STEPS);

		float3 s = ray.origin + ray.direction * march_step * (0.5 + (blue_noise - 0.5));

		float march_step_multed = march_step * DISTANCE_MULT_VAL;

		for (int i = 0; i < SUN_STEPS; i++) {

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
		const float stepScale = (1.0 / SAMPLE_STEPS);
		if(isect_sphere(view_ray, sky.earth.center, sky.earth.atmosphere_radius, t0, t1))
		{

			float inner_sphere0 = 0;
			float inner_sphere1 = 0;
			isect_sphere(view_ray, sky.earth.center, sky.earth.earth_radius, inner_sphere0, inner_sphere1);

			float start_dist = t0;

			float end_dist = min(inner_sphere0, t1);

			float step_lengths = stepScale * (end_dist - start_dist);

			float blue_noise = 0;
			
			float dist = max(0.01f, start_dist - step_lengths * (0.175 + blue_noise * 0.1));
			float prev_dist = dist;

			float3 avg_space_light = 0;

			for(int i = 0; i < SAMPLE_STEPS; i++){

				dist += step_lengths;

				float step_dist = (dist - prev_dist) * DISTANCE_MULT_VAL;

				float3 wp = view * dist;

				Ray ray = make_ray(wp, view);
				get_incident_light_space(sky, ray, step_dist, atmopshere_thickness, blue_noise);

				float3 diffuseSample = texCUBE(EnvironmentIlluminationCubeSampler, normalize(wp - sky.earth.center)).rgb;
				#ifdef AMBIENT_BOOST
					diffuseSample			= 1-exp(-(diffuseSample		+ Square(diffuseSample)		* AMBIENT_BOOST));
				#endif

				avg_space_light += diffuseSample;

				prev_dist = dist;
			}

			avg_space_light *= stepScale;

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
//	const bool oceanMode 					= g_MaterialSpecular.a 					== 1.0;
	const bool noAtmosphereMode 			= g_MaterialSpecular.a 					== 0.0;
//	const bool volcanoMode 					= round(g_MaterialSpecular.a * 4.0) 	== 1.0;
//	const bool gasGiantMode 				= round(g_MaterialSpecular.a * 4.0) 	== 2.0;
//	const bool greenHouseMode	 			= round(g_MaterialSpecular.a * 4.0) 	== 3.0;
	VsCloudsOutput o;
	
	if(noAtmosphereMode)
	{
		o.Position 							= 1.0/(1-noAtmosphereMode);//0x7fc00000;
		o.TexCoord0							= 0;
		o.Normal 							= 0;
		o.Pos 								= 0;
		o.PlanetPos 						= 0;
		o.Light 							= 0;
		o.View 								= 0;
		o.PercentHeight 					= 0;
	}
	else
	{
		
		//Position
		float3 positionInWorldSpace = -mul(float4(iPosition, 1.f), g_World).xyz;
		float inflateScale = distance((mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz) - positionInWorldSpace, positionInWorldSpace) * (1.0 / 131072.0) + 1;
		float atmosphereThickness = 1.0 + frac(g_MaterialGlossiness) * inflateScale;
		//Final Position
		
		o.Position = mul(float4(iPosition * atmosphereThickness, 1.0f), g_WorldViewProjection);
	
		//Texture Coordinates
		o.TexCoord0 = iTexCoord; 
		
		//Calculate  Normal       
		o.Normal = normalize(mul(iNormal, (float3x3)g_World));
		
		o.Pos = positionInWorldSpace;// / atmosphereThickness;// * ATMOSPHERE_SCALE;//(1.0 / ATMOSPHERE_SCALE);
		o.PlanetPos = ((mul(float4(0.0, 0.0, 0.0, 1.0), g_World).xyz - positionInWorldSpace) * atmosphereThickness) / g_Radius;
		//Calculate Light
		o.Light = ((g_Light0_Position - positionInWorldSpace) * atmosphereThickness) / g_Radius;
		
		//Calculate ViewVector
		o.View = normalize(-(positionInWorldSpace * atmosphereThickness) / g_Radius);
		
		o.PercentHeight = abs(iPosition.y) / g_Radius;  
	}
    return o;
}

VsCloudsOutput 
RenderAtmosphereVS(
	float3 iPosition:POSITION, 
	float3 iNormal:NORMAL,
	float2 iTexCoord:TEXCOORD1)		
{	
	return RenderCloudVertex(1.00, iPosition, iNormal, iTexCoord);
}

void RenderAtmospherePS(VsCloudsOutput i, out float4 oColor0:COLOR0) 
{ 

	float3 light 			= normalize(i.Light);
	float3 normal 			= normalize(i.Normal);
	float3 lightColor 		= SRGBToLinear(g_Light0_DiffuseLite.rgb);		
	
	#ifndef DEBUG_PLANET_MODES
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
				float lightSqr				= dot(i.Light, i.Light);                                                                                                                                  
				
				float lightFalloff			= rcp(1.0 + lightSqr);//regular pointlight
		

				lightColor 					*= GetAttenuation(lightFalloff);
				float scatterRayleight		= 250.0;
				float scatterMie			= 250.0;
				float3 sunColor				= 10 * lightColor;
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
		oColor0 = 0;
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
        VertexShader 		= compile vs_3_0 RenderSceneOceanVS();
        PixelShader 		= compile ps_3_0 RenderSceneOceanPS();
        
		AlphaTestEnable 	= FALSE;
        AlphaBlendEnable 	= TRUE;
		SrcBlend 			= ONE;
		DestBlend 			= ZERO;    
		ZEnable 			= true;
		ZWriteEnable 		= true;
    }	
	pass Pass1
    {          
        VertexShader 		= compile vs_3_0 RenderSceneNoAtmosphereVS();
        PixelShader 		= compile ps_3_0 RenderSceneNoAtmospherePS();
        
		AlphaTestEnable 	= FALSE;
        AlphaBlendEnable 	= TRUE;
		SrcBlend 			= ONE;
		DestBlend 			= ZERO;    
		ZEnable 			= true;
		ZWriteEnable 		= true;
    }
	pass Pass2
    {          
        VertexShader 		= compile vs_3_0 RenderSceneVolcanoVS();
        PixelShader 		= compile ps_3_0 RenderSceneVolcanoPS();
        
		AlphaTestEnable 	= FALSE;
        AlphaBlendEnable 	= TRUE;
		SrcBlend 			= ONE;
		DestBlend 			= ZERO;    
		ZEnable 			= true;
		ZWriteEnable 		= true;
    }
	pass Pass3
    {          
        VertexShader 		= compile vs_3_0 RenderGreenhouseVS();
        PixelShader 		= compile ps_3_0 RenderGreenhousePS();
        
		AlphaTestEnable 	= FALSE;
        AlphaBlendEnable 	= TRUE;
		SrcBlend 			= ONE;
		DestBlend 			= ZERO;    
		ZEnable 			= true;
		ZWriteEnable 		= true;
    }
	pass Pass4
    {          
        VertexShader 		= compile vs_3_0 RenderSceneGasGiantVS();
        PixelShader 		= compile ps_3_0 RenderSceneGasGiantPS();
        
		AlphaTestEnable 	= FALSE;
        AlphaBlendEnable 	= TRUE;
		SrcBlend 			= ONE;
		DestBlend 			= ZERO;    
		ZEnable 			= true;
		ZWriteEnable 		= true;
    }    
    pass PassCloudLayer
    {
		VertexShader = compile vs_3_0 RenderAtmosphereVS();
		PixelShader = compile ps_3_0 RenderAtmospherePS();

		ZEnable 			= true;
		ZWriteEnable 		= false;		
		AlphaTestEnable 	= TRUE;		
		AlphaBlendEnable 	= TRUE;
		SrcBlend 			= SRCALPHA;
		DestBlend 			= INVSRCALPHA;
    }
}