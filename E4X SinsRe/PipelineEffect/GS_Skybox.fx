//math constants, don't touch :P
#define PI 		3.1415926535897932384626433832795
#define INVPI 	0.3183098861837906715377675267450
#define TWOPI	6.283185307179586476925286766559
#define MF float(0x7fffffff)
#define SF (1.0 / MF)
//Settings
#define PROCEDURAL_REFINE									//g_MaterialGlossiness integer and fractional part controls two different kinds of refining distortion.

//ANY #define change below shoul match GS_SkyboxBloom.fx!!!
#define PARALLAX_NEBULA 									//g_MaterialDiffuse.a	if defined g_MaterialDiffuse.a = hex 00 turns it on!
	#define PARALLAX_NEBULA_STEPS 20						//not exposed			should be hardcoded, so loop can be unrolled. Can be a lot smaller in bloom pass!!
	#define PARALLAX_BACKDROP_RES float2(4096.0, 2048.0) 	//not exposed			should match SkyboxBackdropX height
	#define PARALLAX_NEBULA_PARALLAX_INTENSITY_MAX 0.01 	//g_MaterialDiffuse.r

#define GENERATE_PARALLAX_STARS 							//g_MaterialAmbient.a	if defined g_MaterialAmbient.a = hex 00 turns it on!
	#define PARALLAX_STAR_SCALE_MIN 1200 					//not exposed			PARALLAX_STAR_SCALE_MIN + PARALLAX_STAR_SCALE_MAX * (g_MaterialDiffuse.b * uniform_random_0to1)
	#define PARALLAX_STAR_SCALE_MAX 3000 					//g_MaterialDiffuse.b	Bigger = smaller! Calculation goes PARALLAX_STAR_SCALE_MIN * random + PARALLAX_STAR_SCALE_MAX,
	#define PARALLAX_STAR_COUNT_MIN 1 						//g_MaterialAmbient.r	round(PARALLAX_STAR_COUNT_MIN + PARALLAX_STAR_COUNT_MAX * g_MaterialAmbient.r)
	#define PARALLAX_STAR_COUNT_MAX 4 						//g_MaterialAmbient.r	controls this. This becomes PARALLAX_STAR_COUNT_MAX*PARALLAX_STAR_COUNT_MAX*6 in total!
	#define PARALLAX_STAR_PARALLAX_INTENSITY_MAX 0.01 		//g_MaterialDiffuse.g
	#define PARALLAX_STAR_BLACKBODY_CURVE_MIN 0.25 			//not exposed			Uniform_random_0to1^(PARALLAX_STAR_BLACKBODY_CURVE_MIN + PARALLAX_STAR_BLACKBODY_CURVE_MAX * g_MaterialAmbient.g)
	#define PARALLAX_STAR_BLACKBODY_CURVE_MAX 7.75 			//g_MaterialAmbient.g	controls this higher value means more red cold red stars, lower value means more blue hot stars
	#define PARALLAX_STAR_SEED_SCALE 1000 					//g_MaterialAmbient.b	is the seed, change it to randomize!
	
shared float4x4	g_ViewProjection : ViewProjection;
float4x4 g_World;

texture	g_TextureDiffuse0 : Diffuse;
texture	g_TextureSelfIllumination;

float4 g_MaterialAmbient:Ambient;
float4 g_MaterialSpecular:Specular;
float4 g_MaterialEmissive:Emissive;//Isn't fed from the mesh file :(
float4 g_MaterialDiffuse:Diffuse;
float g_MaterialGlossiness;
//texture g_TextureNoise3D;//doesn't work, not available

float colorMultiplier = 1.f;

sampler TextureDiffuse0Sampler = 
sampler_state
{
    Texture		= < g_TextureDiffuse0 >;    
    AddressU = WRAP;        
    AddressV = WRAP;
    MagFilter	= LINEAR;
};

sampler TextureDataSampler = sampler_state
{
    Texture = <g_TextureSelfIllumination>;
    AddressU = WRAP;        
    AddressV = WRAP;
    Filter = LINEAR;
};

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


//smooth part of smoothstep
float SmoothCurve(float x)
{
	return x * x * (3.0 - x * 2.0);
}

float2 SmoothCurve(float2 x)
{
	return x * x * (3.0 - x * 2.0);
}

float3 SmoothCurve(float3 x)
{
	return x * x * (3.0 - x * 2.0);
}

float4 SmoothCurve(float4 x)
{
	return x * x * (3.0 - x * 2.0);
}
float SumOf(float2 x)
{
	return dot(x, (float2)1);
}
float SumOf(float3 x)
{
	return dot(x, (float3)1);
}
float SumOf(float4 x)
{
	return dot(x, (float4)1);
}
float AverageOf(float2 x)
{
	return dot(x, (float2)0.5);
}
float AverageOf(float3 x)
{
	return dot(x, (float3)0.33333);
}
float AverageOf(float4 x)
{
	return dot(x, (float4)0.25);
}
float SelfDot(float2 x)
{
	return dot(x, x);
}
float SelfDot(float3 x)
{
	return dot(x, x);
}
float SelfDot(float4 x)
{
	return dot(x, x);
}
inline float selfDotInv(float3 x)
{
	return rcp(1.0 + dot(x , x));
}
float Hash12(float2 p)
{
	float3 p3  = frac(float3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}
float2 Hash23(float3 p3)
{
	p3 = frac(p3 * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return frac((p3.xx+p3.yz)*p3.zy);
}

float3 Hash32(float2 p)
{
	float3 p3 = frac(float3(p.xyx) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return frac((p3.xxy+p3.yzz)*p3.zyx);
}
float3 Hash33(float3 p3)
{
	p3 = frac(p3 * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return frac((p3.xxy + p3.yxx)*p3.zyx);

}
float4 Hash42(float2 p)
{
	float4 p4 = frac(float4(p.xyxy) * float4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return frac((p4.xxyz+p4.yzzw)*p4.zywx);

}
float4 Hash43(float3 p)
{
	float4 p4 = frac(float4(p.xyzx)  * float4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return frac((p4.xxyz+p4.yzzw)*p4.zywx);
}
float4 Hash41(float p)
{
	float4 p4 = frac((float4)(p) * float4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return frac((p4.xxyz+p4.yzzw)*p4.zywx);
    
}
float Hash13(float3 p3)
{
	p3  = frac(p3 * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}
float2 Noise23(float3 p, const bool smooth = false)
{
	float3 i = floor(p);
	float3 f = frac(p);
	if(smooth)
		f = SmoothCurve(f);
	
	return lerp(lerp(lerp(	Hash23(i + float3(0, 0, 0)), 
							Hash23(i + float3(1, 0, 0)), f.x),
					lerp(	Hash23(i + float3(0, 1, 0)), 
							Hash23(i + float3(1, 1, 0)), f.x), f.y),
				lerp(lerp(	Hash23(i + float3(0, 0, 1)), 
							Hash23(i + float3(1, 0, 1)), f.x),
					lerp(	Hash23(i + float3(0, 1, 1)), 
							Hash23(i + float3(1, 1, 1)), f.x), f.y), f.z);
}

float MapStarTemperatures(float unormRange)
{
	 return mad(unormRange, 20000.0, 2000.0);//stars do go hotter, but the blackbody function goes HD range for the super bright ones, and Sins don't really support that :P
}
//good old Star Ruler 2 black body function safe range is 2800
float MixRange(float x, float low, float hi)
{
	return saturate((x - low) / (hi - low));
}

float3 BlackBody(float temp)
{
	float3 c;
	c.r = lerp(1.0, 0.6234, MixRange(temp, 6400.0, 29800.0));
	c.b = lerp(0.0, 1.0, MixRange(temp, 2800.0, 7600.0));
	if(temp < 6600.0)
		c.g = lerp(0.22, 0.976, MixRange(temp, 1000.0, 6600.0));
	else
		c.g = lerp(0.976, 0.75, MixRange(temp, 6600.0, 29800.0));
	if(temp > 13000.0)
		c += (float3)(MixRange(temp, 13000.0, 29800.0));
	return c;
}

#define WARP_THETA 0.868734829276
#define TAN_WARP_THETA 1.18228668555

/* Return a permutation matrix whose first two columns are u and v basis
   vectors for a cube face, and whose third column indicates which axis
   (x,y,z) is maximal. */
float3x3 GetPM(in float3 p)
{

    float3 a = abs(p);
    float c = max(max(a.x, a.y), a.z);
	float3 s = step((float3)c, a);
	s.yz -= float2(s.x * s.y, (s.x + s.y - s.x * s.y) * s.z);
    s *= sign(dot(p, s));
    float3 q = s.yzx;
    return float3x3(cross(q, s), q, s);

}

/* For any point in 3D, obtain the permutation matrix, as well as grid coordinates
   on a cube face. */
void PosToGrid(in float3 Pos, out float3x3 PT, out float2 g, float Density)
{

    // Get permutation matrix and cube face id
    PT = GetPM(Pos);

    // Project to cube face
    float3 c = mul(PT, Pos);
    float2 p = c.xy / c.z;

    // Unwarp through arctan function
    float2 q = atan(p * TAN_WARP_THETA) / WARP_THETA;

    // Map [-1,1] interval to [0,N] interval
    g = (q * 0.5 + 0.5) * Density;

}


/* For any grid point on a cube face, along with projection matrix,
   obtain the 3D point it represents. */
float3 GridToPos(in float3x3 PT, in float2 g, float Density)
{

    // Map [0,N] to [-1,1]
    float2 q = g / Density * 2.0 - 1.0;

    // Warp through tangent function
    float2 p = tan(WARP_THETA * q) / TAN_WARP_THETA;

    // Map back through permutation matrix to place in 3D.
    return mul(float3(p, 1.0), PT);

}


/* Return whether a neighbor can be identified for a particular grid cell.
   We do not allow moves that wrap more than one face. For example, the
   bottom-left corner (0,0) on the +X face may get stepped by (-1,0) to
   end up on the -Y face, or, stepped by (0,-1) to end up on the -Z face,
   but we do not allow the motion (-1,-1) from that spot. If a neighbor is
   found, the permutation/projection matrix and grid coordinates of the
   neighbor are computed.
*/
bool GridNeighbor(in float3x3 PT, in float2 Grid, in float2 Delta, out float3x3 PTn, out float2 gn, float Density)
{
	float2 GridDensity = Grid.xy + Delta;
    float2 GridDensityClamp = clamp(GridDensity, 0.0, Density);

    float2 Extra = abs(GridDensityClamp - GridDensity);
    float ExtraSum = Extra.x + Extra.y;

	float3 Pos = mul(float3(GridDensityClamp / Density * 2.0 - 1.0, 1.0 - 2.0 * ExtraSum / Density), PT);
	PTn = GetPM(Pos);
	gn = (mul(PTn, Pos).xy * 0.5 + 0.5) * Density;

	return min(Extra.x, Extra.y) == 0.0 && ExtraSum < Density;
}

/* Return squared great circle distance of two points projected onto sphere. */
float SphereDist2(float3 a, float3 b)
{
	// Fast-ish approximation for acos(dot(normalize(a), normalize(b)))^2
    return 2.0 - 2.0 * dot(normalize(a), normalize(b));
}

/* Color the sphere/cube points. */
//quad sphere tiled 2D voronoi, way cheeper than 3D voronoi and produced points always on the surface, so no little stupid useless cells
float3 GetStars(float3 Pos, float3 DirP)
{
    float3x3 PT;
    float2 g;
	const float ParallaxStarCount = round(PARALLAX_STAR_COUNT_MIN + PARALLAX_STAR_COUNT_MAX * g_MaterialAmbient.x);
    // Get grid coords
    PosToGrid(Pos, PT, g, ParallaxStarCount);

    // Snap to cube face - note only needed for visualization.
    float3 PosP = Pos / dot(Pos, PT[2]);
	
	normalize(Pos - DirP * Hash13(floor((Pos * 0.5 + 0.5) * ParallaxStarCount) + g_MaterialAmbient.z * PARALLAX_STAR_SEED_SCALE) - 0.5);
	
	
    PosToGrid(Pos, PT, g, ParallaxStarCount);	
	Pos /= dot(Pos, PT[2]);				

    // Distances/colors/points for Voronoi
    float d1 = 100000.0;
    float d2 = 100000.0;

    float m1 = -1.0;
    float m2 = -1.0;

    float3 p1 = 0;
    float3 p2 = 0;

	// For drawing grid lines below
    float2 l = abs(frac(g + 0.5) - 0.5);

    // Move to center of grid cell for neighbor calculation below.
    g = floor(g) + 0.5;

    // For each potential neighbor
    for(float u = -1.0; u <= 1.0; ++u)
	{
        for(float v = -1.0; v <= 1.0; ++v)
		{
            float2 gn;
            float3x3 PTn;

            // If neighbor exists
            if (GridNeighbor(PT, g, float2(u,v), PTn, gn, ParallaxStarCount))
			{
                float face = dot(PTn[2], float3(1., 2., 3.));

                // Perturb based on grid cell ID
                gn = floor(gn);
                float3 rn = Hash32(gn * 0.123 + face);
                gn += 0.5 + (rn.xy * 2.0 - 1.0) * 0.5;
				
                // Get the 3D Position
                float3 PosN = GridToPos(PTn, gn, ParallaxStarCount);

                // Compute squared distance on sphere
                float dp = SphereDist2(Pos, PosN);

                // See if new closest point (or second closest)
                if(dp < d1)
				{
                    d2 = d1; p2 = p1;
                    d1 = dp; p1 = PosN;
                }
				else if(dp < d2)
				{
                    d2 = dp; p2 = PosN;
                }
            }
        }
    }

    float4 RandomBase = Hash43(p1 + g_MaterialAmbient.z * PARALLAX_STAR_SEED_SCALE);
    float3 StarPos = (Pos - p1) * (PARALLAX_STAR_SCALE_MAX * RandomBase.y * g_MaterialDiffuse.b + PARALLAX_STAR_SCALE_MIN);
    float StarSqr = dot(StarPos, StarPos);
    return pow(BlackBody(MapStarTemperatures(pow(RandomBase.z, PARALLAX_STAR_BLACKBODY_CURVE_MIN + g_MaterialAmbient.y * PARALLAX_STAR_BLACKBODY_CURVE_MAX))), 2.2) * rcp(1+StarSqr);
}
float3 GetTriplanarMask(float3 Normal)
{
	float3 Mask = max((float3)SF, pow(Square(Normal), 20));
	return round(Mask / dot(Mask, (float3)1.0));
}
void RenderSceneVS( 
	float3 iPosition : POSITION, 
	float3 iNormal : NORMAL,
	float3 iTangent : TANGENT,	
	float2 iTexCoord0 : TEXCOORD0,
	out float4 oPosition : POSITION,
	out float3 oNormal	 : NORMAL,
	out float3 oTangent	 : TANGENT,	
    out float4 oColor0 : COLOR0,
    out float2 oTexCoord0 : TEXCOORD0,
//	out float3 oView	 : TEXCOORD1,
	out float3 oPos	 : TEXCOORD1,
	out float3 oNormalObj	 : TEXCOORD2)
{
	oNormalObj		= normalize(iPosition.xyz);//iNormal;
	oPosition		= mul(float4(iPosition, 1), g_World);
	oPosition		= mul(oPosition, g_ViewProjection);
	oPos			= mul(float4(iPosition, 1), g_ViewProjection).xyz;
//	oView = normalize(mul(float3(0.0, 0.0, 1.0), (float3x3)g_ViewProjection));
	#if 1
		oNormal		= normalize(mul(normalize(-iPosition.xyz), (float3x3)g_ViewProjection));
	#else
		oNormal		= normalize(mul(iNormal, (float3x3)g_ViewProjection));
	#endif
	oTangent 		= normalize(mul(iTangent, (float3x3)g_ViewProjection));
	
    oColor0 		= float4(1, 1, 1, 1) * colorMultiplier;
    oTexCoord0 		= iTexCoord0;
}

void
RenderScenePS(
	float3 iNormal : NORMAL,
	float3 iTangent : TANGENT,
	float4 iColor : COLOR,
	float2 iTexCoord0 : TEXCOORD0,
//	float3 iView : TEXCOORD1,
	float3 iPos : TEXCOORD1,
	float3 iNormalObj : TEXCOORD2,	
	out float4 oColor0 : COLOR0) 
{ 
	oColor0					= float4(0.0, 0.0, 0.0, 1.0);
	float3 View				= float3(0.0, 0.0, 1.0);
	float3 Normal 			= normalize(iNormal);
	float2 dUVx 			= ddx(iTexCoord0);
	float2 dUVy 			= ddy(iTexCoord0);
	float3 DPx 				= ddx(iPos);
	float3 DPy 				= ddy(iPos);
	float3 ReflectionVS		= normalize(reflect(View, Normal));
	float3 ReflectionOS		= -float3(dot(ReflectionVS, ((float3x3)g_ViewProjection)[0]), dot(ReflectionVS, ((float3x3)g_ViewProjection)[1]), dot(ReflectionVS, ((float3x3)g_ViewProjection)[2]));
	float2 ResInv 			= 1.0 / PARALLAX_BACKDROP_RES;
	#ifdef PARALLAX_NEBULA
	float StepsNeeded		= (length(Normal.xy) * 0.75 + 0.25) * PARALLAX_NEBULA_STEPS;// 25% in the center, 200%, 100% in the middles sides and top/bottom in the corners 
		if(round(g_MaterialDiffuse.a) == 0)
		{	
			#if 1
				float3 DPXPerp 			= cross(Normal, DPx);		
				float3 DPYPerp 			= cross(DPy, Normal);		
				float3 Tanget 			= DPYPerp * dUVx.x + DPXPerp * dUVy.x;
				float3 Cotangent 		= DPYPerp * dUVx.y + DPXPerp * dUVy.y;
				float InvMax			= pow(max(dot(Tanget, Tanget), dot(Cotangent, Cotangent)), -0.5);
				Tanget					*= InvMax;
				Cotangent 				*= InvMax;	
			#else
				float3 Tanget 			= normalize(iTangent);//something is up with the basis!?
				float3 Cotangent 		= normalize(cross(Normal, Tanget));
			#endif
			float Depth 			= PARALLAX_NEBULA_PARALLAX_INTENSITY_MAX * g_MaterialDiffuse.r * (1-Square(saturate(abs(iTexCoord0.y * 2.05 - 1.025))));
			float3 ReflectionTS		= -float3(dot(ReflectionVS, Tanget), dot(ReflectionVS, Cotangent), dot(ReflectionVS, Normal));
//			float2 ViewTS			= float2(Tanget.z, Cotangent.z) / Normal.z;
			ReflectionTS.xy 		/= ReflectionTS.z;
	
			float StepsInv			= (1.0 / StepsNeeded);
			float StepOffset		= 0.0;
			float4 SumColor			= float4(0.0, 1.0, 0.0, 0.0);
			SumColor				= 0;	
			float SumDepth			= 0;
			float SumWeight			= 0;
		
			float2 SumUV			= 0;
			iTexCoord0				-= ReflectionTS.xy * Depth;
			for(int i = 0; i < StepsNeeded + 2; i++)	
			{
				float2 StepUV		= iTexCoord0 + ReflectionTS.xy * (StepOffset - (SumWeight - 0.5) * StepsInv) * Depth;
				StepOffset			+= StepsInv;		
				float SampleDepth	= tex2Dgrad(TextureDataSampler, StepUV, dUVx, dUVy).a;
				SampleDepth			= saturate(SampleDepth * StepsNeeded - StepsNeeded + i) * (1-SumDepth);
		
				SumWeight			= lerp(SumWeight, 1, SampleDepth);// * (1-SumDepth));	
				SumDepth			+= SampleDepth;
				SumUV				+= StepUV * SampleDepth;		
				if(SumDepth == 1)
					break;
			}
			iTexCoord0				= float2(SumUV.x, clamp(frac(SumUV.y), ResInv.y, 1.0 - (ResInv.y)));
		}
		//re-evaluate
		dUVx 						= ddx(iTexCoord0);
		dUVy 						= ddy(iTexCoord0);		
	#endif
	
	ResInv *= 0.5;
	#ifdef PROCEDURAL_REFINE
		float SkyboxStarMask 		= 1-tex2Dgrad(TextureDataSampler, iTexCoord0, dUVx, dUVy).g;
		float2 Sharpen				= float2(	Square(1-AverageOf(tex2Dgrad(TextureDiffuse0Sampler, iTexCoord0 + float2(ResInv.x, 0.0), dUVx, dUVy).rgb)) - Square(1-AverageOf(tex2Dgrad(TextureDiffuse0Sampler, iTexCoord0 - float2(ResInv.x, 0.0), dUVx, dUVy).rgb)), 
												Square(1-AverageOf(tex2Dgrad(TextureDiffuse0Sampler, iTexCoord0 + float2(0.0, ResInv.y), dUVx, dUVy).rgb)) - Square(1-AverageOf(tex2Dgrad(TextureDiffuse0Sampler, iTexCoord0 - float2(0.0, ResInv.y), dUVx, dUVy).rgb)));
		float2 Distortion			= 	abs(SmoothCurve(Noise23((normalize(iNormalObj) + 10) * 200)) * 2 - 1) - 
										abs(SmoothCurve(Noise23((normalize(iNormalObj) - 10) * 190)) * 2 - 1);
	
		iTexCoord0					-= (Sharpen * ResInv * round(g_MaterialGlossiness) + Distortion * length(Sharpen) * frac(g_MaterialGlossiness)) * SkyboxStarMask;

		dUVx 						= ddx(iTexCoord0);
		dUVy 						= ddy(iTexCoord0);		
	#endif
	
	oColor0 						= tex2Dgrad(TextureDiffuse0Sampler, iTexCoord0, dUVx, dUVy);	

	#ifdef GENERATE_PARALLAX_STARS
		if(g_MaterialAmbient.a == 0)
		{
			#if 0
				oColor0.rgb			= GetStars(normalize(iNormalObj), ReflectionOS * (PARALLAX_STAR_PARALLAX_INTENSITY_MAX * g_MaterialDiffuse.g));
			#else
				float3x3 PT;
				float2 g;
				// Get grid coords
				float3 Pos			= normalize(iNormalObj);
				//ReflectionOS * (PARALLAX_STAR_PARALLAX_INTENSITY_MAX * g_MaterialDiffuse.g);
				float3 UVx			= Pos.xyz / Pos.zxy;
				float3 UVy			= Pos.yzx / Pos.zxy;
				float3 BoxFace		= saturate(1-floor(max(abs(UVx), abs(UVy))));
				float3 BoxDir		= sign(Pos.zxy);
				float BoxOffset		= dot(BoxFace * BoxDir, float3(1.0, 2.0, 3.0));
				float StarDensity	= round(PARALLAX_STAR_COUNT_MIN + PARALLAX_STAR_COUNT_MAX * g_MaterialAmbient.x);
				float2 StarUV		= (float2(dot(UVx, BoxFace), dot(UVy, BoxFace)) * 0.5 + 0.5);
				float2 HashUV		= floor(StarUV * StarDensity) + g_MaterialAmbient.z * PARALLAX_STAR_SEED_SCALE;

				float2 dUVx 		= ddx(StarUV);
				float2 dUVy 		= ddy(StarUV);	
				float3 DPXPerp 		= cross(Normal, DPx);		
				float3 DPYPerp 		= cross(DPy, Normal);		
				float3 Tanget 		= DPYPerp * dUVx.x + DPXPerp * dUVy.x;
				float3 Cotangent 	= DPYPerp * dUVx.y + DPXPerp * dUVy.y;
				float InvMax		= pow(max(dot(Tanget, Tanget), dot(Cotangent, Cotangent)), -0.5);
				Tanget				*= InvMax;
				Cotangent 			*= InvMax;	

				float3 ReflectionTS	= -float3(dot(ReflectionVS, Tanget), dot(ReflectionVS, Cotangent), dot(ReflectionVS, Normal));
				ReflectionTS.xy 	/= ReflectionTS.z;
				
				StarUV				+= ReflectionTS.xy * (Hash12(HashUV) - 0.5) * (PARALLAX_STAR_PARALLAX_INTENSITY_MAX * g_MaterialDiffuse.g);
				StarUV 				*= StarDensity;
				float4 RandomBase	= Hash42(HashUV);
				StarUV				= frac(StarUV) * 2 - 1.5 + RandomBase.xy;
				float StarScale		= (PARALLAX_STAR_SCALE_MAX * RandomBase.z * g_MaterialDiffuse.b + PARALLAX_STAR_SCALE_MIN);
				StarUV				*= StarScale * rcp(StarDensity * 3);
				oColor0.rgb			+= pow(BlackBody(MapStarTemperatures(pow(RandomBase.w, PARALLAX_STAR_BLACKBODY_CURVE_MIN + g_MaterialAmbient.y * PARALLAX_STAR_BLACKBODY_CURVE_MAX))), 2.2) * rcp(1 + dot(StarUV, StarUV));	
			#endif
		}
	#endif

//	oColor0.rgb = float3(Sharpen, 0);
//	oColor0.rgb = tex2Dgrad(TextureDataSampler, iTexCoord0, dUVx, dUVy).rgb;
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

