

///////////////////////////////////////////////////////////////
//							To do:							 //
///////////////////////////////////////////////////////////////
//	- Thanks + references
//	- Main focus:
//		60 FPS
//		Physically based rendered
//		Time of day and weather manager
//
//	- Clean code and debug:
//		Separated functions				100% (WIP)
//		Naming convention				100% (WIP)
//		Consistency						100% (WIP)
//	
//	- Shaders:
//		PBR / BRDF Opaque Shader		100%
//		Water							50%  (WIP)
//		Wet surfaces					0%
//	
//	- Time of day: 
//		LightingÂ 						100% 
//		Colors							100%
//		Soft shadows					100%
//		Skylight / ambient				80%
//		Night / Stars					100%
//	
//	- Clouds:
//		Weather system					100%
//		Cloud lighting color			100%
//		Shadows							100%
//		Birds silhouettes				0%
//		God rays						0% 	 (Nice to have)
//	
//	- Raining:
//		- Two directions + depth pass	0% 	 (Just a draft for now)
//		- ScreenSpace droplet			50%	 (To polish screen space oriented)
//		- Rainbow 						90%	 (To polish the fade on ground)
//	
//	- Terrain:
//		Terrain function				100%  (WIP)
//		Terrain textures				75%
//	
//	- Camera: 
//		Camera path function			75%	 (Nice to have)
//	
//	- Post process:
//		Fog								100%
//		Bloom							50%  (Sun glow only for now)
//		Gamma correction				100%
//		Color tint						100% (Implemented, to tweak at the end with cloudy)
//		Color desaturation				100% (Implemented, to tweak at the end with cloudy
//		Contrast						100% (Implemented, to tweak at the end)
//		VignetteÂ 						100% (Implemented, to tweak at the end)
//		Grain							100% (Implemented, to tweak at the end)
//		Lens flare effect				50%  (Implemented, to tweak at the end)
//		DOF for horizon					0%	 (Nice to have)
//		Auto-exposure					0%	 (Nice to have)
//		Motion blur						0%	 (Nice to have)
//		Chromatic aberration			0%	 (Nice to have)
//		Musique (if so, I need a visual related to support it)
//
///////////////////////////////////////////////////////////////

//-----------------------------------------------------------//
// --------------------- DEBUG VALUES ---------------------- //
//-----------------------------------------------------------//
// 0  == RENDER_ALL_PASS
// 1  == NORMAL_PASS
// 2  == DEPTH_PASS
// 3  == POS_PASS
// 4  == NdotL_PASS
// 5  == NdotV_PASS
// 6  == NdotH_PASS
// 7  == VdotH_PASS
// 8  == CLOUDS_PASS
// 9 == AMBIANT_PASS
// 10 == SHADOW_PASS
#define DEBUG_PASS 0

// 0 == fixed cam
// 1 == control path cam
// 2 == cam path
// 3 == rotated cam
#define DEBUG_CAM 2

//#define DEBUG_NO_AMBIANT
//#define DEBUG_NO_SHADOW
//#define DEBUG_NO_DIFF
//#define DEBUG_NO_CLOUDS
//#define DEBUG_NO_NIGHT
//#define DEBUG_NO_FOG
//#define DEBUG_NO_WATERDROPLET
//#define DEBUG_NO_RAIN
//#define DEBUG_NO_RAINBOW
//TBD DEBUG_RAINBOW ?! 
//#define DEBUG_NO_TERRAIN
//#define DEBUG_NO_SANDSTRIPS
//#define DEBUG_NO_POSTPROCESS
//#define DEBUG_SPEED_FAST

// Time of day activated
#define ToD_ACTIVATED
#define PBR_SKY

// Clouds
#define CLOUDYCHANGE

// DEBUG TEXT
#define kCharBlank 12.0
#define kCharMinus 11.0
#define kCharDecimalPoint 10.0
#define FONT_RATIO vec2(4.0, 5.0)
#define FONT_SCALE 10.0
#define DIGITS 1.0
#define DECIMAL 3.0

//-----------------------------------------------------------//
// --------------------- GLOBAL VALUES --------------------- //
//-----------------------------------------------------------//
#define PI 3.14159
#define TAU 6.28319

#define EPSILON		0.0001
#define NEARCLIP 	0.1

#ifndef DEBUG_NO_TERRAIN
	#define FARCLIP 	300.0
#else
	#define FARCLIP 	400.0
#endif

#define FOV 			PI / 2.5 //FOV in rad == 39.598 == 50.0mm lens

#define AMBIENT_COLOR vec3(0.54117647058824, 0.78431372549020, 1.0)
#define AMBIENT_POW 0.08

#define SCENE_SAMPLE 128
#define SHADOW_SAMPLE 16

//-----------------------------------------------------------//
// -------------------- RAINING VALUES --------------------- //
//-----------------------------------------------------------//
#ifndef DEBUG_NO_RAIN
	//x = radius, y= size, z = intensity, w = depth %age;
	#define RAINBOW_PARAMS vec4(0.25, 0.15, 0.8, 0.4)
	#define MAX_DROPLET_NMBR 50.0 //number of drop on screen
	float rainingValue;

	#ifdef DEBUG_SPEED_FAST
		#define RAIN_SPEED 0.4
	#else
		#define RAIN_SPEED 0.075
	#endif

#endif

//-----------------------------------------------------------//
// ---------------------- SKY VALUES ----------------------- //
//-----------------------------------------------------------//
#ifdef PBR_SKY
	#define SKY_SAMPLES 16
	#define SKYLIGHT_SAMPLE 8
	#define SKY_HR 7.994e3 		// Rayleigh == scattering of light by air mollecules
	#define SKY_HM 1.2e3		// Mie == scattering of light by aerosols
	#define SKY_EARTH_RADIUS 6360e3
	#define SKY_V3_EARTH_RADIUS vec3(0.0, -SKY_EARTH_RADIUS, 0.0) //For scattering only
	#define SKY_ESCAPE_V vec3(0.0, 63625e2, 0.0)
	#define SKY_ESCAPE_C -73499375e4;
	//#define SKY_G 0.76 //--> G^2 below
	#define SKY_G2 0.5776

	//Sky at sea level for wavelengths 440, 550 and 680
	#define SKY_BETA_R vec3(5.8e-6, 13.5e-6, 33.1e-6)
	#define SKY_BETA_M vec3(21e-6)
#endif

//-----------------------------------------------------------//
// --------------------- NIGHT VALUES ---------------------- //
//-----------------------------------------------------------//
#ifdef ToD_ACTIVATED
    // Gaz density during the night. 0.0 == more, 0.5 == less
    #define NIGHT_GAZDENSITY 0.2
    // Color multiplier for intensity and color correction
    #define NIGHT_GAZCOLOR vec3(0.5, 0.45, 0.5)
    // Star size, biggest is smaller
    #define NIGHT_STARSIZE 80.0
#endif

//-----------------------------------------------------------//
// ---------------------- SUN VALUES ----------------------- //
//-----------------------------------------------------------//
float sunAmount;
vec2 sunPos;
//Sun radius
#define SUN_RADIUS 1500.0
#define FIXED_LIGHT vec3(-0.773, 0.635, -0.01)

#ifdef ToD_ACTIVATED
	//Start time:	17hrs 30min 6/21/2014 latt: 37.795 long: -122.394 
	#define SUN_STARTPOS vec3(0.001, 0.927, -0.376)
	//End time:		20hrs 0min 6/21/2014 latt: 37.795 long: -122.394
	//#define SUN_ENDPOS vec3(-0.824, 0.0, 0.567)
    //#define SUN_ENDPOS vec3(-0.636, -0.181, 0.751)
	#ifndef DEBUG_NO_NIGHT
		#define SUN_ENDPOS vec3(-0.142, -0.847, 0.512)
	#else
		#define SUN_ENDPOS vec3(-0.636, -0.181, 0.751)
	#endif
	/*17hrs 30min spectrum == 0%
	const vec3 sunStartColor = vec3(1.0, 1.0, 0.8784313725490196);
	19hrs 20min spectrum == 73%
	const vec3 sunMidColor1 = vec3(0.99607843137255, 0.65490196078431, 0.03529411764705);
	19hrs 30min spectrum == 80%
	const vec3 sunMidColor2 = vec3(0.80392156862745, 0.21568627450980, 0.85098039215686);
	19hrs 37min spectrum == 85%
	const vec3 sunMidColor3 = vec3(0.35686274509804, 0.14901960784314, 0.63137254901961);
	20hrs 00min spectrum == 100%
	const vec3 sunMidColor4 = vec3(0.10980392156862, 0.11764705882352, 0.23921568627450);*/
	vec3 sunDirection;
	vec4 sun_Color; //xyz == color, w == intensity

	#ifdef DEBUG_SPEED_FAST
		#ifdef DEBUG_NO_NIGHT
			#define SUN_SPEED 0.125
		#else
			#define SUN_SPEED 0.0625
		#endif
	#else
		#ifdef DEBUG_NO_NIGHT
			#define SUN_SPEED 0.043
		#else
			#define SUN_SPEED 0.043
		#endif
	#endif
#else
	//fixed time: 	18hrs 40min 6/21/2014 latt: 37.795 long: -122.394
	#define sunDirection vec3(-0.773, 0.635, -0.01)
	#define sun_Color vec4(0.84954, 0.99958, 0.99797, 1.0)
#endif

//-----------------------------------------------------------//
// --------------------- CLOUDS VALUES --------------------- //
//-----------------------------------------------------------//
#define CLOUD_SAMPLES 20.0
#define CLOUD_LOWER 1000.0
#define CLOUD_UPPER 1500.0
//#define CLOUD_LOWER 2500.0
//#define CLOUD_UPPER 4000.0
#ifdef DEBUG_SPEED_FAST
	#define CLOUD_MOVE 0.5
	#define CLOUD_EVAP 0.6
#else
	#define CLOUD_MOVE 0.5
	#define CLOUD_EVAP 0.6
#endif

#ifdef CLOUDYCHANGE
	//Cloudy value from 0.15(DarkClouds) to 0.74 (clear sky)
	float cloudy;
	#ifdef DEBUG_SPEED_FAST
		#define CLOUD_SPEED 0.25
	#else
		#define CLOUD_SPEED 0.1
	#endif

#else
	#ifndef DEBUG_NO_RAIN
		#define cloudy 0.375
	#else
		#define cloudy 0.5
	#endif
#endif

//-----------------------------------------------------------//
// -------------------- TERRAIN VALUES --------------------- //
//-----------------------------------------------------------//
#ifndef DEBUG_NO_TERRAIN
	#define TERRAIN_SCALE 0.3
	#define TERRAIN_HEIGHT 8.0
	#define TERRAIN_MAXHEIGHT 1.5
#endif

//-----------------------------------------------------------//
// --------------------- WATER VALUES ---------------------- //
//-----------------------------------------------------------//
//used for the plane if no terrain and no water
#define WATER_HEIGHT 7.35			// Global Water height

//-----------------------------------------------------------//
// ---------------------- SAND STRIP ----------------------- //
//-----------------------------------------------------------//
#ifndef DEBUG_NO_SANDSTRIPS
    #define STRIP_SAMPLE 100
    #define STRIP_BUMP_CONTRAST 1.0
    #define STRIP_SIZE 8.0 //higher value is smaller strip
    #define STRIP_FREQ 12.0 
#endif

//-----------------------------------------------------------//
// ------------------ POSTPROCESS VALUES ------------------- //
//-----------------------------------------------------------//
#ifndef DEBUG_NO_POSTPROCESS
    #define GAMMA (1.0/2.2)
    #define CONTRAST 1.2
    #define BRIGHTNESS 0.45
    #define DESATURATION 1.125
    #define TINT_COLOR vec3(1.012, 0.988, 1.0)
    #define VIGNETING_POWER 0.33
	#define GRAIN_STRENGTH 16.0
#endif



////////////////////////////////////////////////////////////////
//-----------------------------------------------------------//
// ------------------------ STRUCTS -------------------------//
//-----------------------------------------------------------//
// Materials
// Energine conservation == microsurface + reflectivity
// 1. Diffuse: 		The color of diffused light
// 2. Reflectivity: How reflective a surface is when viewing head
struct material 
{
	vec3		albedo;
	float		reflectivity;
};

// Ray intersection
struct rayIntersect 
{
	vec3		mPos;				// Pos
    vec2		mUV;				// Screen space pos
	vec3		nor; 				// Normal
	float		dist;				// Distance
    vec3		rd;					// Ray direction
	material	mat; 				// Object material
};


//-----------------------------------------------------------//
// -------------------- MATH FUNCTIONS --------------------- //
//-----------------------------------------------------------//
#ifndef DEBUG_NO_SANDSTRIPS
    // Standard 2D rotation formula
    mat2 Rot2(const in float angle)
    { 
        float c = cos(angle), s = sin(angle); 
        return mat2(c, s, -s, c);
    }  
#endif
    
// Quadratic easing in - accelerating from zero velocity--------
#define EaseInQuad(value) (value) * (value)

// Quartic easing in - accelerating from zero velocity
#define EaseInQuart(value) pow((value), 4.0)

// Exponential easing in - accelerating from zero velocity
#define EaseInExpo(value) pow(2.0, 10.0 * abs((value) - 1.0))

// Exponential easing in/out - accelerating until halfway, then decelerating
float  EaseInOutExpo(const in float value) 
{
    if(value * 2.0 < 1.0) 
        return 0.5 * pow(2.0, 10.0 * abs((value) * 2.0 - 1.0));

    return 0.5 * (-pow(2.0, abs(-10.0 * ((value) * 2.0 - 1.0)) + 2.0));
}

// Cubic easing out - decelerating to zero velocity
#define EaseOutCubic(value) (pow((value) * 2.0 - 1.0, 3.0) + 1.0)

// Circular easing in - accelerating from zero velocity
#define EaseInCirc(value) (-(sqrt(abs(1.0 - EaseInQuad(value))) - 1.0))
float EaseInCircFct(const in float value)
{
    return EaseInCirc(value);
}

// Quintic easing out - decelerating to zero velocity
#define EaseOutQuint(value) (pow((value) - 1.0, 5.0) + 1.0)

// Sinusoidal easing out - decelerating to zero velocity
#define EaseOutSine(value) (sin((value) * (PI/2.0)))

// used for terrain and rain screen space
float Hash1D(const in float n) 
{
	return fract(sin(n)*4378.5453);
}

// used for water, stars 
float Hash2D(const in vec2 mPos) 
{
    float h = dot(mPos, vec2(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

#ifndef DEBUG_NO_SANDSTRIPS
    // used for sand stripes
    vec2 Hash22(const in vec2 mPos) 
    {
        float n = sin(dot(mPos, vec2(113.0, 1.0)));

        return fract(vec2(2097152.0, 262144.0) * n) * 2.0 - 1.0;
    }
#endif

#ifdef ToD_ACTIVATED
	// Heat color gradient
	vec3 HeatmapGradient(const in float value) 
	{
		return clamp((pow(value, 1.5) * 0.8 + 0.2) * vec3(smoothstep(0.0, 0.35, value) + value * 0.5, 
													  smoothstep(0.5, 1.0, value), 
													  max(1.0 - value * 1.7, value * 7.0 - 6.0)), 
					 0.0, 1.0);
	}
#endif

#ifndef DEBUG_NO_RAIN
	// Calculate rainbow color
	vec3 RainbowGradient(const in float value) 
	{
		vec3 c = 1.0 - pow(abs(vec3(value) - vec3(0.65, 0.5, 0.2)) 
						   * vec3(3.0, 3.0, 5.0), vec3(1.5, 1.3, 1.7));
		c.r = max((0.15 - EaseInQuad(abs(value - 0.04) * 5.0)), c.r);
		c.g = (value < 0.5) ? smoothstep(0.04, 0.45, value) : c.g;
		return clamp(c, 0.0, 1.0);
	}
#endif

#ifndef DEBUG_NO_CLOUDS
	// 3D noise for clouds
	float Noise3D(const in vec3 mPos)
	{
		vec3 p = floor(mPos);
		vec3 f = fract(mPos);
		f = f * f * (2.0 - 1.0 * f); //EaseInExpo
		
		vec2 noiseUV = (p.xy + vec2(37.0,17.0) * p.z) + f.xy;
		vec2 rg = texture(iChannel0, (noiseUV + 0.5)/256.0, -100.0).yx;
		return mix(rg.x, rg.y, f.z);
	}

	float FBM(in vec3 p)
	{
		p *= 0.35;
		float f;
		
		f = 0.5000 * Noise3D(p); 
		p = p * 3.02; 
		p.x += iTime * CLOUD_MOVE;
		
		f += 0.2500 * Noise3D(p); 
		p = p * 3.03; 
		p.y -= iTime * CLOUD_EVAP;
		
		f += 0.1250 * Noise3D(p); 
		p = p * 3.01;
		
		f += 0.0625   * Noise3D(p); 
		p =  p * 3.03;
		
		f += 0.03125  * Noise3D(p); 
		p =  p * 3.02;

		return f;
	}

	float MapClouds(in vec3 p)
	{
		p *= 0.001;
		return FBM(p);
	}
#endif

#ifndef DEBUG_NO_DUST
// More concise, self contained version of IQ's original 3D noise function.
float Noise3D_Dust(in vec3 mPos)
{
    // Just some random figures, analogous to stride. You can change this, if you want.
	const vec3 s = vec3(113, 157, 1);
	
	vec3 ip = floor(mPos); // Unique unit cell ID.
    
    // Setting up the stride vector for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    vec4 h = vec4(0.0, s.yz, s.y + s.z) + dot(ip, s);
    
	mPos -= ip; // Cell's fractional component.
	
    // A bit of cubic smoothing, to give the noise that rounded look.
    mPos = mPos * mPos * (3.0 - 2.0 * mPos);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: fract(sin(x)*largeNumber).
    h = mix(fract(sin(h) * 43758.5453), fract(sin(h + s.x) * 43758.5453), mPos.x);
	
    // Interpolating along Y.
    h.xy = mix(h.xz, h.yw, mPos.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    return mix(h.x, h.y, mPos.z); // Range: [0, 1].
}
#endif
//-----------------------------------------------------------//
// ------------------- END MATH FUNCTIONS ------------------ //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// -------------------- SAND STRIPS ------------------------ //
//-----------------------------------------------------------//
#ifndef DEBUG_NO_SANDSTRIPS
	// Gradient noise based on IQ's implementation
    float GradN2D(in vec2 f)
    {
        const vec2 e = vec2(0.0, 1.0);
        vec2 p = floor(f);
        f -= p; // Fractional position within the cube.

        vec2 w = f * f * (3.0 - 2.0 * f); // Cubic smoothing. 
        float c = mix(mix(dot(Hash22(p + e.xx), f - e.xx), dot(Hash22(p + e.yx), f - e.yx), w.x),
                      mix(dot(Hash22(p + e.xy), f - e.xy), dot(Hash22(p + e.yy), f - e.yy), w.x), w.y);

        return c*.5 + .5; // Range: [0, 1].
    }

    float Grad(in float x, const in float offs)
    {
        x = abs(fract(x / TAU + offs - 0.25) - 0.5) * 2.0;

        float x2 = clamp(x * x * (-1.0 + 2.0 * x), 0.0, 1.0); // Customed smoothed, peaky triangle wave.
        x = smoothstep(0.0, 1.0, x);
        return mix(x, x2, 0.15);
    }

	// One sand function layer
	float SandL(const in vec2 mPos)
    {
        // Layer one. 
        vec2 q = Rot2(PI / 18.0) * mPos; // Rotate the layer, but not too much.
        q.y += (GradN2D(q * 18.0) - 0.5) * 0.05; // Perturb the lines to make them look wavy.
        float grad1 = Grad(q.y * 80.0, 0.0); // Repeat gradient lines.

        q = Rot2(-PI / 20.0) * mPos; // Rotate the layer back the other way, but not too much.
        q.y += (GradN2D(q * 12.0) - 0.5) * 0.05; // Perturb the lines to make them look wavy.
        float grad2 = Grad(q.y * 80.0, 0.5); // Repeat gradient lines.


        // Mix the two layers above with an underlying 2D function. The function you choose is up to you,
        // but it's customary to use noise functions. However, in this case, I used a transcendental 
        // combination, because I like the way it looked better.
        q = Rot2(PI / 4.0) * mPos;

        // The mixes above will work, but I wanted to use a subtle screen blend of grad1 and grad2.
        float a2 = dot(sin(q * 12.0 - cos(q.yx * 12.0)), vec2(0.25)) + 0.5;
        float a1 = 1.0 - a2;

        // Screen blend.
        float c = 1.0 - (1.0 - grad1 * a1) * (1.0 - grad2 * a2);

        return c;
    }

    float Sand(in vec2 mPos, const in float dist)
    {
        // Rotating by 45 degrees + zoomed in by a factor of 4.
        mPos = vec2(mPos.y - mPos.x, mPos.x + mPos.y) * 0.7071 / 4.0;

        // Sand layer 1.
        float c1 = SandL(mPos);

        // Second layer.
        // Rotate, then increase the frequency -- The latter is optional.
        vec2 q = Rot2(PI / 12.0) * mPos;
        float c2 = SandL(q * 1.25);

        // Mix the two layers with some underlying gradient noise.
        c1 = mix(c1, c2, smoothstep(0.1, 0.9, GradN2D(mPos * vec2(4.0))));

        // A surprizingly simple and efficient hack to get rid of the super annoying Moire pattern 
        // formed in the distance. Simply lessen the value when it's further away. Most people would
        // figure this out pretty quickly, but it took me far too long before it hit me. :)
        return c1/(1.0 + dist * dist * 0.015);
    }

    // Surface bump function..
    float BumpSurf3D(const in vec3 mPos, const in float dist)
    {
        return Sand(mPos.xz, dist);
    }

    vec3 DoBumpMap(const in vec3 p, const in vec3 nor, const in float bumpfactor, const in float dist)
    {
        const vec2 e = vec2(0.001, 0); 

        float ref = BumpSurf3D(p, dist);
        vec3 grad = (vec3(BumpSurf3D(p - e.xyy, dist),
                          BumpSurf3D(p - e.yxy, dist),
                          BumpSurf3D(p - e.yyx, dist)) - ref)/e.x; 

        grad -= nor*dot(nor, grad);          
        return normalize(nor + grad*bumpfactor);
    }
#endif
//-----------------------------------------------------------//
// ------------------ END SAND STRIPS ---------------------- //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// ---------------- DISTANCE FUNCTIONS --------------------- //
//-----------------------------------------------------------//
// By Iq: https://iquilezles.org/articles/distfunctions
// Union with Material ID
vec2 OperationUnion(const in vec2 distance1, const in vec2 distance2)
{
    return (distance1.x < distance2.x) ? distance1 : distance2;
}

#ifndef DEBUG_NO_TERRAIN
    float TerrainDisplacement(const in vec2 mPos)
    {
        vec2 currPos = mPos * TERRAIN_SCALE;
        float g = (sin(currPos.x + sin(currPos.y * 1.7)) 
                   + sin(currPos.y + sin(currPos.x * 1.3))) * 0.2;
		
		return g * 2.5 * TERRAIN_MAXHEIGHT + TERRAIN_HEIGHT;
    }

	float TerrainHighRez(const in vec2 mPos, const in vec2 eps, const in float waveFactor)
    {
        float dg1 = TerrainDisplacement(mPos - eps);
        float dg2 = TerrainDisplacement(mPos + eps);
        
        float wave = (sin(mPos.x * 30.0) + 1.0) * 0.25 * max(waveFactor, 0.0);
        
		return (dg1 - dg2) + wave;
    }
#endif
//-----------------------------------------------------------//
// -------------- END DISTANCE FUNCTIONS ------------------- //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// -------------------- SCENE DISTANCE --------------------- //
//-----------------------------------------------------------//
vec2 SceneDistance(const in vec3 mPos)
{
    #ifndef DEBUG_NO_TERRAIN
    	return vec2(mPos.y - TerrainDisplacement(mPos.xz), 1.0);
    #else
    // Unsigned distance plane
    	return vec2(mPos.y - WATER_HEIGHT, 0.0);
	#endif
}
//-----------------------------------------------------------//
// ------------------ END SCENE DISTANCE ------------------- //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// ----------------------- NIGHT SKY ----------------------- //
//-----------------------------------------------------------//
// [NV15] Space curvature from Iq : https://www.shadertoy.com/view/llj3Rz#
#ifdef ToD_ACTIVATED
    #ifndef DEBUG_NO_NIGHT
        vec3 FancyCube(const in vec3 rayDirection, const in float s, const in float b)
        {
            vec3 colx = texture(iChannel1, 0.5 + s * rayDirection.yz / rayDirection.x, b).xyz;
            vec3 coly = texture(iChannel1, 0.5 + s * rayDirection.zx / rayDirection.y, b).xyz;
            vec3 colz = texture(iChannel1, 0.5 + s * rayDirection.xy / rayDirection.z, b).xyz;

            vec3 n = rayDirection * rayDirection;

            return (colx * n.x + coly * n.y + colz * n.z) / (n.x + n.y + n.z);
        }

        vec2 Voronoi(const in vec2 mPos)
        {
            vec2 n = floor(mPos);
            vec2 f = fract(mPos);

            vec3 m = vec3(8.0);
            for(int j = -1; j <= 1; ++j)
            for(int i = -1; i <= 1; ++i)
            {
                vec2  g = vec2( float(i), float(j));
                vec2  o = vec2(Hash2D(n + g));
                vec2  r = g - f + o;
                float d = dot(r, r);
                if(d < m.x)
                    m = vec3(d, o);
            }

            return vec2(sqrt(abs(m.x)), m.y + m.z);
        }

        vec3 StarSky(const in vec3 rayDirection)
        {
            vec3 col = vec3(0.0);

            // Big blueish gaz
            col += 0.8 * pow(FancyCube(rayDirection, 0.05, 5.0).zyx, vec3(2.0));

            // More detailed purpleish gaz
            col += 0.8 * vec3(0.8, 0.5, 0.6) * pow(FancyCube(rayDirection, 0.1, 0.0).xxx, vec3(6.0));

            //Stars
            vec3 n = pow(abs(rayDirection), vec3(3.0));
            vec2 vxy = Voronoi(NIGHT_STARSIZE * rayDirection.xy);
            vec2 vyz = Voronoi(NIGHT_STARSIZE * rayDirection.yz);
            vec2 vzx = Voronoi(NIGHT_STARSIZE * rayDirection.zx);

            float stars = smoothstep(0.3, 0.7, FancyCube(rayDirection, 0.91, 0.0).x);
            vec2 r = (vyz * n.x + vzx * n.y + vxy * n.z) / (n.x + n.y + n.z);
            col += 0.9 * stars * clamp(1.0 - (3.0 + r.y * 5.0) * r.x, 0.0, 1.0);

            // Gaz density
            col = 1.5 * col - NIGHT_GAZDENSITY;

            // Gaz color correction
            col *= NIGHT_GAZCOLOR;

            // TBD: MOON instead of sun
            /*float s = clamp(dot(rayDirection, sunDirection), 0.0, 1.0);
            col += 0.4 * pow(s, 5.0)  * vec3(1.0, 0.7, 0.6) * 2.0;
            col += 0.4 * pow(s, 64.0) * vec3(1.0, 0.9, 0.8) * 2.0;*/

            return col * rayDirection.y;
        }
	#endif
#endif
//-----------------------------------------------------------//
// --------------------- END NIGHT SKY --------------------- //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// -------------- ATMOSPHERIC SCATTERING PBR --------------- //
//-----------------------------------------------------------//
// http://www.scratchapixel.com/lessons/3d-advanced-lessons/simulating-the-colors-of-the-sky/atmospheric-scattering/
#ifdef PBR_SKY
	void Densities(const in vec3 mPos, out float rayleigh, out float mie) 
	{
		float height = length(mPos - SKY_V3_EARTH_RADIUS) - SKY_EARTH_RADIUS;
		rayleigh =  exp(-height / SKY_HR);
		mie = exp(-height / SKY_HM);
	}

	float Escape(const in vec3 rayDirection) 
	{
		float b = dot(SKY_ESCAPE_V, rayDirection);
		float det2 = b * b - SKY_ESCAPE_C;
		
		if (det2 < 0.0) 
			return -1.0;
			
		float det = sqrt(abs(det2));
		float t1 = -b - det;
		float t2 = -b + det;
		
		return (t1 >= 0.0) ? t1 : t2;
	}

	//vec3 SkyScatterring(const in vec2 mUV, const in vec3 rayDirection)
	vec3 SkyScatterring(const in vec2 mUV, in vec3 rayDirection)                                      
	{
		//Clamp min height rd
		rayDirection.y = max(rayDirection.y, -0.07);
		
		float segment = Escape(rayDirection);
		float segmentLength = segment / float(SKY_SAMPLES);

		vec3 sumR , sumM = vec3(0.0);
		float opticalDepthR, opticalDepthM = 0.0;
		float mu = dot(rayDirection, sunDirection);
		float opmu2 = 1.0 + mu * mu;
		float phaseR = 3.0 / (16.0 * PI) * opmu2;
		float phaseM = 	3.0 / (8.0 * PI) * ((1.0 - SKY_G2) * opmu2  / ((2.0 + SKY_G2) * pow(1.0 + SKY_G2 - 2.0 * 0.76 * mu, 1.5)));
		
		for(int i = 0; i < SKY_SAMPLES; ++i) 
		{
			float samplePosition = float(i) * segmentLength;
			vec3 height = vec3(0.0, 25e2, 0.0) + rayDirection * samplePosition;
			float rayleigh, mieScaleHeight = 0.0;
			
			// Compute optical depth for light
			Densities(height, rayleigh, mieScaleHeight);
			rayleigh *= segmentLength; 
			mieScaleHeight *= segmentLength;
			opticalDepthR += rayleigh;
			opticalDepthM += mieScaleHeight;
			
			// Light optical depth
			float lightRay = Escape(sunDirection);

			float segmentLengthLight = lightRay / float(SKYLIGHT_SAMPLE);
			float opticalDepthLightR, opticalDepthLightM = 0.0;
			for(int j = 0; j < SKYLIGHT_SAMPLE; ++j) 
			{
				float lightRayleigh, LightMieScaleHeight = 0.0;
				float samplePositionLight = float(j) * segmentLengthLight;
				vec3 heightLight = height + sunDirection * samplePositionLight;
				Densities(heightLight, lightRayleigh, LightMieScaleHeight);
				opticalDepthLightR += lightRayleigh * segmentLengthLight;
				opticalDepthLightM += LightMieScaleHeight * segmentLengthLight;
			}

			vec3 attenuation = exp(-(SKY_BETA_R * (opticalDepthLightR + opticalDepthR) + SKY_BETA_M * 1.1 * (opticalDepthM + opticalDepthLightM)));
			sumR += attenuation * rayleigh;
			sumM += attenuation * mieScaleHeight;
		}
		return 20.0 * sun_Color.w * (sumR * phaseR * SKY_BETA_R + sumM * phaseM * SKY_BETA_M);
	}
#else
	vec3 SkyScatterring(const in vec2 mUV, in vec3 rayDirection)                                      
	{
		float sun = clamp(dot(sunDirection, rayDirection), 0.0, 1.0);
		vec3 hor = mix(1.2 * vec3(0.70, 1.0, 1.0), vec3(1.5, 0.5, 0.05), 0.25 + 0.75 * sun);
		
        vec3 skyColor = mix(vec3(0.2, 0.6, 0.9), 
                            hor, 
                            exp(-(4.0 + 2.0 * (1.0 - sun)) * max(0.0, rayDirection.y - 0.1)));
        
		skyColor *= 0.5;
		skyColor += 0.8 * vec3(1.0, 0.8, 0.7) * pow(sun, 512.0);
		skyColor += 0.2 * vec3(1.0, 0.4, 0.2) * pow(sun, 32.0);
		skyColor += 0.1 * vec3(1.0, 0.4, 0.2) * pow(sun, 4.0);
        
        return skyColor;
	}
#endif
//-----------------------------------------------------------//
// ------------ END ATMOSPHERIC SCATTERING PBR ------------- //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// ------------------------ SKYBOX ------------------------- //
//-----------------------------------------------------------//
    ////////!\\\\\\\\
    //    Notes    \\
    ////////!\\\\\\\\^
    /*sun color, and shadowed clouds, like a gradient but separated from the intensity
        I should manage color and intensity separatly: 
            Try HSV or HSL?!
            Hue by using kind of heat map (scale factor: SunAmount, cloudsDensity, TimeOfDay);
            Saturation by using easing function (cloudy, cloudsDensity, cloudHeight);
            Value by using easing function (cloudy, SunAmount);


        Shadowed cloud during day is kind of blue ish(sky?!):
            vec3(0.47058823529412, 0.58823529411765, 0.72156862745098)
            same as above but 100% saturated: vec3(0.0, 0.33725490196078, 0.72156862745098)
            HSV: 212, 35, 72
        Lighted cloud during day is kind of light orange (sunColor?!)
            vec3(0.98039215686275, 0.92156862745098, 0.84313725490196)
            100% saturated: vec3(0.98039215686275, 0.55686274509804, 0.0)
            HSV: 34, 14, 98

        Global clouds luminosity variation for a half clear sky is from 72% to 98% (day)

        Shadowed cloud during sunset
        I believed it was a dark purple but it seems it's more a dark red
            vec3(0.32549019607843, 0.2, 0.25098039215686);
            100% saturated: vec3(0.32941176470588, 0.0, 0.1333333333333333);
            HSV: 336, 39, 33
        Lighted cloud during the sunset depends on the sun position/sunamount
        Goes from saturated orange to red, seems to have relation with suncolor
            close from sun pos: (orange)
                vec3(0.99607843137255, 0.70980392156863, 0.24705882352941);
                100% saturated: vec3(0.98823529411765, 0.61176470588235, 0.0);
                HSV: 37, 75, 99
            far from the sun: (red)
                vec3(0.97647058823529, 0.30980392156863, 0.16862745098039);
                100% saturated: vec3(0.98039215686275, 0.16470588235294, 0.0);
                HSV: 10, 83, 98

        Global clouds luminosity variation for a half clear sky is from 33% to 99% (sunset)

        Shadowed cloud during the night very dark blue
            vec3(0.09019607843137, 0.10196078431373, 0.13725490196078);
            100% saturated: vec3(0.0, 0.03529411764706, 0.14117647058824);
            HSV: 225, 34, 14
        Lighted cloud during the night grey blue
            vec3(0.21568627450980, 0.21176470588235, 0.23137254901961);
            100% saturated: vec3(0.04705882352941, 0.0, 0.23137254901961);
            HSV: 252, 8, 23

        Global clouds luminosity variation for a half clear sky is from 14% to 23% (night)*/

    ////////!\\\\\\\\
    // TO DO sunRay\\
    ////////!\\\\\\\\
//#ifndef DEBUG_NO_CLOUDS	
    float GetCloudShadow(const in vec3 pos)
    {
        float cloudyChange = abs(1.0 - (0.1 + (cloudy - 0.15) * (0.8/0.6)));
        vec2 cuv = pos.xz + sunDirection.xz * (100.0 - pos.y) / sunDirection.y;
        float cc = 0.1 + 0.9 * smoothstep(0.0, cloudyChange, 
                                          texture( iChannel1, 0.0008 * cuv 
                                                    + 0.005 * iTime).x);
	
        return cc;
    }
//#endif

vec3 CheapSkyBox(const in vec3 rayOrigin, const in rayIntersect rIntersec)
{
    vec3 sky = SkyScatterring(rIntersec.mUV, rIntersec.rd);

    //Start and End of the cloud layer
    float cloudStart = (CLOUD_LOWER - rayOrigin.y) / abs(rIntersec.rd.y);
    float cloudEnd = (CLOUD_UPPER - rayOrigin.y) / abs(rIntersec.rd.y);

    // Clouds start rayOrigin
    vec3 cloudPos = vec3(rayOrigin.x + rIntersec.rd.x * cloudStart,
                         0.0,
                         rayOrigin.z + rIntersec.rd.z * cloudStart);

    // Raymarch steps to raytrace clouds
    vec3 add = rIntersec.rd * ((cloudEnd - cloudStart) / CLOUD_SAMPLES);
    vec2 shade = vec2(0.0, 0.0);

    // x == heigh value, y == mask (density on sky)
    vec2 cloudsValue = vec2(0.0, 0.0);

    // Loop for cloud density
    for(float i = 0.0; i < CLOUD_SAMPLES; ++i)
    {
        if (cloudsValue.y >= 1.0) 
            break;

        // Intensity of the current layer
        shade.y = max(MapClouds(cloudPos) - cloudy, 0.0);

        // Heigh value of the current layer
        shade.x = i/CLOUD_SAMPLES;

        // Merging new infos with backuped one
        cloudsValue += shade * (1.0 - cloudsValue.y);
        cloudPos += add;
    }
    cloudsValue.x /= 10.0;
    cloudsValue = min(cloudsValue, 1.0);

    float cloudsIntensity = cloudsValue.x * rIntersec.rd.y;
    float sunHeight = clamp(sunDirection.y, 0.5, 1.0);
    
    // Applying contrast with cloudContrast and brightness with sunHeight
    vec3 clouds = mix(((AMBIENT_COLOR * cloudsIntensity) - 0.5) + sunHeight,
                      vec3(1.0),
                      cloudsValue.x);

    //clouds intensity by ToD
    clouds *= max(sunDirection.y, 0.05);

    //Composite sky and clouds
    return mix(sky, clouds, cloudsValue.y);
}
                                     
vec3 GetSkybox(const in vec3 rayOrigin, const in rayIntersect rIntersec)
{
    vec3 sky = SkyScatterring(rIntersec.mUV, rIntersec.rd);

    #ifdef ToD_ACTIVATED
        #ifndef DEBUG_NO_NIGHT
            float nightMult = abs(min(0.16667, sun_Color.w * 6.0 - 1.0));
            if(nightMult > 0.16667)
                sky += StarSky(rIntersec.rd) * nightMult;
        #endif
	#endif
    
    #ifdef DEBUG_NO_CLOUDS
        //Apply sun on sky and return
    	sky += sun_Color.xyz * pow(sunAmount, abs(SUN_RADIUS)) * 40.0;
        return sky;
    
    #else
		//Start and End of the cloud layer
		float cloudStart = (CLOUD_LOWER - rayOrigin.y) / abs(rIntersec.rd.y);
		float cloudEnd = (CLOUD_UPPER - rayOrigin.y) / abs(rIntersec.rd.y);

		// Clouds start rayOrigin
		vec3 cloudPos = vec3(rayOrigin.x + rIntersec.rd.x * cloudStart,
							 0.0,
							 rayOrigin.z + rIntersec.rd.z * cloudStart);

		// Raymarch steps to raytrace clouds
		vec3 add = rIntersec.rd * ((cloudEnd - cloudStart) / CLOUD_SAMPLES);
		vec2 shade = vec2(0.0, 0.0);

		// x == heigh value, y == mask (density on sky)
		vec2 cloudsValue = vec2(0.0, 0.0);

		// Loop for cloud density
		for(float i = 0.0; i < CLOUD_SAMPLES; ++i)
		{
			if (cloudsValue.y >= 1.0) 
				break;

			// Intensity of the current layer
			shade.y = max(MapClouds(cloudPos) - cloudy, 0.0);

			// Heigh value of the current layer
			shade.x = i/CLOUD_SAMPLES;

			// Merging new infos with backuped one
			cloudsValue += shade * (1.0 - cloudsValue.y);
			cloudPos += add;
		}
		cloudsValue.x /= 10.0;
		cloudsValue = min(cloudsValue, 1.0);
		
		// cloudsValue.x [0.17647058823529; 1.0] need to transpose to [0.61, 0.91];
		float cloudsIntensity = EaseInCircFct(cloudsValue.x / 2.75 + 0.54636);
		float sunHeight = clamp(sunDirection.y, 0.5, 1.0);
		float cloudContrast = 1.6875 - 1.25 * cloudy;

		// Applying contrast with cloudContrast and brightness with sunHeight
		vec3 clouds = mix(((AMBIENT_COLOR * cloudsIntensity) - 0.5) * cloudContrast + sunHeight,
						  vec3(1.0),
						  cloudsValue.x);

		//clouds intensity by ToD
		clouds *= max(sunDirection.y, 0.05);

		//Composite Sun on clouds
		clouds += sun_Color.xyz * min(pow(sunAmount, abs(SUN_RADIUS) * 0.025) * (cloudy + 0.2) / 3.2,
									  exp(sun_Color.w+0.4) * 0.3);

		//Composite sun on sky
		sky += sun_Color.xyz * pow(sunAmount, abs(SUN_RADIUS)) * 40.0;

		//Composite sky and clouds
		return mix(sky, min(clouds, 1.0), cloudsValue.y);
    #endif
}
//-----------------------------------------------------------//
// ---------------------- END SKYBOX ----------------------- //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// ------------------------- RAIN -------------------------- //
//-----------------------------------------------------------//
#ifndef DEBUG_NO_RAIN
	vec4 RainFX(const in vec4 colorResult, const in vec2 mUV, const in vec2 mPos)
    {
        vec2 texScroll =  mUV * vec2((mPos.y+2.5)*0.3, 0.05)
                        + vec2(iTime * 0.165 + mPos.y *0.4, iTime * 0.1);
        vec2 texScroll2 = mUV * vec2((mPos.y+2.5)*-0.5, 0.05) 
                        + vec2(iTime * 0.25 + mPos.y * -0.2, iTime * 0.1);

        // Rain
        float rainingFactor = texture(iChannel0, texScroll).y 
                              * texture(iChannel0, texScroll * 0.773).x * 1.55;
        rainingFactor += texture(iChannel0, texScroll2).y 
                         * texture(iChannel0, texScroll2 * 0.770).x * 1.55;

        rainingFactor = pow(rainingFactor, abs(colorResult.w));

        //TBD: better zDepth 'cause it shit right now
       return vec4(colorResult.xyz
                   + (colorResult.xyz + vec3(1.2) * 0.001)
                   * (vec3(0.4, 0.4, 0.45)* rainingFactor),
                   colorResult.w);
    }

	#ifndef DEBUG_NO_RAINBOW
    vec4 ApplyRainbow(const in vec3 rayDirection, const in vec4 colorResult)
    {
        float rainbowAmount = dot(vec2(-FIXED_LIGHT.xy), rayDirection.xy);

        float colorValue = smoothstep(RAINBOW_PARAMS.x, RAINBOW_PARAMS.x + RAINBOW_PARAMS.y, 
                                      rainbowAmount);
        
        if(colorValue < 1.0)
        {
            rainbowAmount = max((1.0 - (RAINBOW_PARAMS.w * EaseInQuad(rayDirection.z)
                                       ))//* (cloudy - 0.75)))
                                //* sin(PI * (-rainingValue + 1.0) * (-rainingValue + 1.0))
                                * sin(PI * (-rainingValue + 2.0) * (-rainingValue + 2.0))
                                * RAINBOW_PARAMS.z 	// Intensity max
                                * colorResult.w 	// ZDepth
                                ,0.0);
            
            //return vec4(rainbowAmount * min(cloudy + 0.5, 1.0) * sunDirection.y);
            
            return vec4(colorResult.xyz + RainbowGradient(1.0 - colorValue)* RAINBOW_PARAMS.z
                        * rainbowAmount * min(cloudy + 0.5, 1.0) * sunDirection.y
                        , colorResult.w);
        }
        
        return colorResult;
    }
	#endif
#endif
//-----------------------------------------------------------//
// ----------------------- END RAIN ------------------------ //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// ------------------- SHADOW / NORMAL --------------------- //
//-----------------------------------------------------------//
#ifndef DEBUG_NO_SHADOW
    float SoftShadow(const in vec3 rayOrigin)
    {
        float res = 1.0;
        float t = 0.1;
        float sceneSample_F = float(SHADOW_SAMPLE);

        for(int i=0; i < SHADOW_SAMPLE; ++i)
        {
            vec3  p = rayOrigin + sunDirection * t;
            float h = SceneDistance(p).x;
            res = min(res, sceneSample_F * h / t);

            if(res < 0.1)
                return 0.1;
            
            t += h;
        }

        return res;
    }
#endif

#ifndef DEBUG_NO_TERRAIN
// Calculate normals
vec3 CalcTerrainNormal(const in vec3 mPos, const in float dist)
{
    vec2 eps = vec2(1.0 / iResolution.y * dist * dist, 0.0);
    vec3 nor;
    nor.x = TerrainDisplacement(mPos.xz - eps.xy) - 
        	TerrainDisplacement(mPos.xz + eps.xy);
    nor.y = 0.5 * eps.x;    
    
    nor.z = TerrainDisplacement(mPos.xz - eps.yx) - 
        	TerrainDisplacement(mPos.xz + eps.yx);
    
    nor = normalize(nor);
    
    vec3 msand = DoBumpMap(mPos, nor, 0.07, dist);
    
    return normalize(msand);
}
#endif

//-----------------------------------------------------------//
// ------------------ END SHADOW / NORMAL ------------------ //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// ------------------- SHADING FUNCTIONS ------------------- //
//-----------------------------------------------------------//
vec3 Shading(vec3 rayOrigin, rayIntersect rIntersec)
{
    float NdotL = clamp(dot(rIntersec.nor, sunDirection), 0.0, 1.0);
    float NdotV = clamp(dot(rIntersec.nor, -rIntersec.rd), 0.0, 1.0);
    
    // Fake GI
    float NdotF = clamp(dot(rIntersec.nor, FIXED_LIGHT), 0.2, 0.6);
	float shadow = 1.0;
    vec3 amb, diff = vec3(0.0);

    
    // Ambient
    #ifndef DEBUG_NO_AMBIANT
	    amb = (abs(sun_Color.w - 1.0) * 0.03 + AMBIENT_POW) * AMBIENT_COLOR * rIntersec.mat.albedo;
    #endif
    
    // Diffuse
    #ifndef DEBUG_NO_DIFF
    	diff = sun_Color.xyz * rIntersec.mat.albedo - rIntersec.mat.reflectivity / (2.0 * PI);
    #endif
    
    #ifndef DEBUG_NO_SHADOW
    	shadow = SoftShadow(rIntersec.mPos + sunDirection);
    	#ifndef DEBUG_NO_CLOUDS
    		shadow *= GetCloudShadow(rIntersec.mPos.xyz);
    	#endif
    #endif
    
    return amb + mix(NdotV * amb * rIntersec.mat.albedo, 
                     diff * NdotL * shadow,
                     sun_Color.w);
}
//-----------------------------------------------------------//
// ----------------- END SHADING FUNCTIONS ----------------- //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// -------------------- COLOR FUNCTIONS -------------------- //
//-----------------------------------------------------------//
void ColorScene(inout rayIntersect rIntersec)
{
	#ifndef DEBUG_NO_TERRAIN
    	//if(rIntersec.mat.microsurface - 0.5 < 1.0)
    {
        rIntersec.mat.albedo = vec3(244.0, 164.0, 96.0)/255.0;//vec3(0.929412, 0.788235, 0.686275);
        rIntersec.mat.reflectivity = 0.9;
        rIntersec.nor = CalcTerrainNormal(rIntersec.mPos, rIntersec.dist);
        return;
    }

    rIntersec.nor = vec3(0.0, 1.0, 0.0);
    
    #endif
}
//-----------------------------------------------------------//
// ------------------ END COLOR FUNCTIONS ------------------ //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// ---------------------- RAYMARCHING ------------------------ //
//-----------------------------------------------------------//
vec3 RayMarchScene(vec3 rayOrigin, inout rayIntersect rIntersec)
{
    float t = NEARCLIP;
    vec2 res = vec2(NEARCLIP, -1.0);
    
	for(int i=0; i < SCENE_SAMPLE; ++i)
	{
        vec3 pos = rayOrigin + t * rIntersec.rd;
        res = SceneDistance(pos);
        if(res.x < (EPSILON * t) || t > FARCLIP)
            break;
        t += res.x * 0.5;
	}
    
    vec3 pos = rayOrigin + t * rIntersec.rd;
    rIntersec.mPos = pos;
    rIntersec.dist = t;
	
    material mat;
    mat.albedo = vec3(244.0, 164.0, 96.0)/255.0;
    mat.reflectivity = 0.8;
    rIntersec.mat = mat;
    
    #if DEBUG_PASS == 0
        if (t > FARCLIP)
        {
            rIntersec.dist = FARCLIP;
            return GetSkybox(rayOrigin, rIntersec);
        }
        else
        {
            #ifndef DEBUG_NO_FOG
                float sundot = clamp(dot(rIntersec.rd, sunDirection), 0.0, 1.0);
            	vec3 sky = CheapSkyBox(rayOrigin, rIntersec);
            	ColorScene(rIntersec);
            
            	float fogFactor = EaseOutSine(rIntersec.dist / FARCLIP);
            	return mix(Shading(rayOrigin, rIntersec), sky, fogFactor);
            #else
            	ColorScene(rIntersec);
            	return Shading(rayOrigin, rIntersec);
            #endif
        }
    
    #elif DEBUG_PASS == 1
    	if(t < FARCLIP)
            ColorScene(rIntersec);
    	return rIntersec.nor;
    
    #elif DEBUG_PASS == 2
        if (t > FARCLIP)
            rIntersec.dist = FARCLIP;
        return vec3(rIntersec.dist) / FARCLIP;
    
    #elif DEBUG_PASS == 3
    	return rIntersec.mPos;
    
    #elif DEBUG_PASS == 4
    	if(t < FARCLIP)
            ColorScene(rIntersec);
        vec3 sunLightPos = normalize(sunDirection);
        float NdotL = clamp(dot(rIntersec.nor, sunLightPos), 0.0, 1.0);
        return vec3(NdotL);
    
    #elif DEBUG_PASS == 5
    	if(t < FARCLIP)
            ColorScene(rIntersec);
        float NdotV = clamp(dot(rIntersec.nor, -rIntersec.rd), 0.0, 1.0);
        return vec3(NdotV);
    
    #elif DEBUG_PASS == 6
    	if(t < FARCLIP)
            ColorScene(rIntersec);
        vec3 sunLightPos = normalize(sunDirection);
        vec3 HalfAngleV = normalize(-rIntersec.rd + sunLightPos);
        float NdotH = clamp(dot(rIntersec.nor, HalfAngleV), 0.0, 1.0);
        return vec3(NdotH);
    
    #elif DEBUG_PASS == 7
    	if(t < FARCLIP)
            ColorScene(rIntersec);
        vec3 sunLightPos = normalize(sunDirection);
        vec3 HalfAngleV = normalize(-rIntersec.rd + sunLightPos);
        float VdotH = clamp(dot(-rIntersec.rd, HalfAngleV), 0.0, 1.0);
        return vec3(VdotH);
    
    #elif DEBUG_PASS == 8
    	if (t > FARCLIP)
        {
            rIntersec.dist = FARCLIP;
            return GetSkybox(rayOrigin, rIntersec);
        }
        else
	    	return CheapSkyBox(rayOrigin, rIntersec);
    
    #elif DEBUG_PASS == 9
    	if(t < FARCLIP)
            ColorScene(rIntersec);
    
    	vec3 amb = (abs(sun_Color.w - 1.0) * 0.03 + AMBIENT_POW) * AMBIENT_COLOR;
    	return amb;

    #elif DEBUG_PASS == 10
   		float shadow = SoftShadow(rIntersec.mPos + sunDirection);
    	return vec3(shadow);
    
    #endif
}
//-----------------------------------------------------------//
// -------------------- END RAYMARCHING -------------------- //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// --------------------- SCENE RENDER ---------------------- //
//-----------------------------------------------------------//
vec4 RenderScene(vec2 mUV, vec3 rayOrigin, vec3 rayDirection, inout rayIntersect rIntersec)
{
	rIntersec.mPos = vec3(0.0);
    rIntersec.mUV = mUV;
	rIntersec.nor = vec3(0.0);
	rIntersec.dist = 0.0;
	rIntersec.rd = rayDirection;
    
	// Opaque
    vec3 accum = RayMarchScene(rayOrigin, rIntersec);
    
    float fogFactor = EaseOutSine(rIntersec.dist / FARCLIP);
    
    vec4 colorResult = vec4(clamp(accum, 0.0, 1.0), fogFactor);
    
    return colorResult;
}
//-----------------------------------------------------------//
// ------------------- END SCENE RENDER -------------------- //
//-----------------------------------------------------------//

//-----------------------------------------------------------//
// ------------------------ DUST --------------------------- //
//-----------------------------------------------------------//
#ifndef DEBUG_NO_DUST
float GetDust(in vec3 rayOrigin, in vec3 rayDirection, in vec3 light, in float dist)
{
    float mist = 0.0;
    
    rayOrigin -= vec3(0.0, 0.0, iTime*0.5);
    
    float t0 = 0.0;
    
    for (int i = 0; i < 4; i++)
    {
        // If we reach the surface, don't accumulate any more values.
        if (t0 > dist) 
            break; 
        // Lighting. Technically, a lot of these points would be
        // shadowed, but we're ignoring that.
        float sDi = length(light - rayOrigin) / FARCLIP; 
	    float sAtt = 1.0 / (1.0 + sDi * 0.25);
	    
        // Noise layer.
        vec3 ro2 = (rayOrigin + rayDirection * t0) * 2.5;
        float c = Noise3D_Dust(ro2) * 0.65 + Noise3D_Dust(ro2 * 3.0) * 0.25 + Noise3D_Dust(ro2 * 9.0) * 0.1;

        mist += c*sAtt;
        
        // Advance the starting point towards the hit point. You can 
        // do this with constant jumps (FAR/8., etc), but I'm using
        // a variable jump here, because it gave me the aesthetic 
        // results I was after.
        t0 += clamp(c * 0.33, 0.25, 0.85);
        
    }
    
    // Add a little noise, then clamp, and we're done.
    return max(mist/12., 0.);
    
    // A different variation (float n = (c. + 0.);)
    //return smoothstep(.05, 1., mist/32.);
}

void ApplyDust(inout vec4 mColor, const in vec3 rayOrigin, const in vec3 rayDirection, const in vec3 light, const in rayIntersect rIntersec)
{
    float dust = GetDust(rayOrigin, rayDirection, sunDirection, rIntersec.dist) * sun_Color.w;
    vec3 mistCol = vec3(1.0, 0.95, 0.9); 
    mColor.xyz += (mix(mColor.xyz, mistCol, 1.66) * 0.66 + mColor.xyz * mistCol * 1.0) * dust;
}
#endif
//-----------------------------------------------------------//
// --------------------- END DUST -------------------------- //
//-----------------------------------------------------------//



//-----------------------------------------------------------//
// --------------------- POST PROCESS ---------------------- //
//-----------------------------------------------------------//
#ifndef DEBUG_NO_POSTPROCESS
	// Grain by jcant0n == https://www.shadertoy.com/view/4sXSWs
	void Grain(inout vec3 colorResult, const in vec2 mUV)
	{
        float x = (mUV.x + 4.0 ) * (mUV.y + 4.0 ) * (iTime * 10.0);
        vec3 grain = 1.0 - vec3(mod((mod(x, 13.0) + 1.0) * 
                                    (mod(x, 123.0) + 1.0), 
                                    0.01)-0.005) * GRAIN_STRENGTH;
    
        colorResult *= grain;
    }

    void LensFX(inout vec3 colorResult, const in vec2 mUV)
    {
        vec2 uvT = mUV * (length(mUV));

        float lensR1 = max(1.0 / (1.0 + 32.0 * pow(length(uvT + 0.80 * sunPos), 2.0)), 0.0) * 0.25;
        float lensG1 = max(1.0 / (1.0 + 32.0 * pow(length(uvT + 0.85 * sunPos), 2.0)), 0.0) * 0.23;
        float lensB1 = max(1.0 / (1.0 + 32.0 * pow(length(uvT + 0.90 * sunPos), 2.0)), 0.0) * 0.21;

        vec2 uvx = mix(mUV, uvT, -0.5);
        lensR1 += max(0.01 - pow(length(uvx + 0.400 * sunPos), 2.4), 0.0) * 6.0;
        lensG1 += max(0.01 - pow(length(uvx + 0.450 * sunPos), 2.4), 0.0) * 5.0;
        lensB1 += max(0.01 - pow(length(uvx + 0.500 * sunPos), 2.4), 0.0) * 3.0;

        uvx = mix(mUV, uvT, -0.4);
        lensR1 += max(0.01 - pow(length(uvx + 0.200 * sunPos), 5.5), 0.0) * 2.0;
        lensG1 += max(0.01 - pow(length(uvx + 0.400 * sunPos), 5.5), 0.0) * 2.0;
        lensB1 += max(0.01 - pow(length(uvx + 0.600 * sunPos), 5.5), 0.0) * 2.0;

        uvx = mix(mUV, uvT, -0.5);
        lensR1 += max(0.01 - pow(length(uvx - 0.300 * sunPos), 1.6), 0.0) * 6.0;
        lensG1 += max(0.01 - pow(length(uvx - 0.325 * sunPos), 1.6), 0.0) * 3.0;
        lensB1 += max(0.01 - pow(length(uvx - 0.350 * sunPos), 1.6), 0.0) * 5.0;

        // Dirty screen
        float dirt = 1.0-texture( iChannel1, mUV).r;
        dirt *= (pow(sunAmount,30.0) + 0.05) * 0.8;

        colorResult += (vec3(lensR1, lensG1, lensB1) * sunAmount * 5.0 + dirt) * sun_Color.w * cloudy;
    }

    void FlareFX(inout vec4 colorResult, const in vec2 mUV)
    {
        //flare
        vec2 sunScreen = mUV - sunPos.xy;
        float ang = atan(sunScreen.y, sunScreen.x);
        float dist = length(sunPos); 
        dist = pow(dist, 0.1);

        float flare = 1.0 / (length(sunScreen) * 8.0);

        flare = flare + flare * (sin((ang + abs(ang) * 2.0) * 12.0) * 0.1 + dist * 0.1 + 0.8);

        colorResult.xyz += (sun_Color.xyz * flare * sun_Color.w);
    }

    void GammaCorrection(inout vec3 colorResult)
    {
        colorResult = pow(colorResult, abs(vec3(GAMMA)));
    }

    void ContrastBrightness(inout vec3 colorResult)
    {
        colorResult = (colorResult - 0.5) * CONTRAST + BRIGHTNESS;
    }

    void Desaturate(inout vec3 colorResult)
    {
        #ifndef DEBUG_NO_CLOUDS
            //colorResult = mix(vec3(dot(colorResult, vec3(0.33))), colorResult, min(cloudy * 2.5, 1.0));
        	colorResult = mix(colorResult, vec3(dot(colorResult, vec3(0.33))), min(rainingValue, 0.4));
        #else
            colorResult = mix(vec3(dot(colorResult, vec3(0.33))), colorResult, DESATURATION);
        #endif
    }

    void Tint(inout vec3 colorResult)
    {
        colorResult *= TINT_COLOR;
    }

    void Vigneting(inout vec3 colorResult, const in vec2 fragCoord)
    {
        vec2 mUV = fragCoord.xy / iResolution.xy;
        colorResult *= 0.5 + 0.5 * pow(16.0 * mUV.x * mUV.y
                                       * (1.0 - mUV.x) * (1.0 - mUV.y), abs(VIGNETING_POWER));
    }

    void ApplyPostProcess(inout vec4 colorResult, const in vec2 mUV, const in vec2 fragCoord)
    {
        #ifndef DEBUG_NO_RAIN
    		//colorResult = RainFX(colorResult, mUV, fragCoord / iResolution.xy);
    	#endif
        
        //LensFlareFX
        if (sun_Color.w > 0.0)
        {
            LensFX(colorResult.xyz, mUV);
            //FlareFX(colorResult, mUV);
        }

        Grain(colorResult.xyz, mUV);
        GammaCorrection(colorResult.xyz);
        ContrastBrightness(colorResult.xyz);
        Desaturate(colorResult.xyz);
        Tint(colorResult.xyz);
        Vigneting(colorResult.xyz, fragCoord);
    }
#endif
//-----------------------------------------------------------//
// ------------------- END POST PROCESS -------------------- //
//-----------------------------------------------------------//

// Camera orientation + Raining post process
vec3 GetCameraRayDir(const in vec2 mUV, const in vec3 camPosition, const in vec3 camTarget)
{
	vec3 forwardVector = normalize(camTarget - camPosition);
	vec3 rightVector = normalize(cross(vec3(0.0, 1.0, 0.0), forwardVector));
	vec3 upVector = normalize(cross(forwardVector, rightVector));	
    
	vec3 camDirection = normalize(mUV.x * rightVector * FOV
                                  + mUV.y * upVector * FOV
                                  + forwardVector);

    
	sunPos = vec2(dot(sunDirection, rightVector), 
                  dot(sunDirection, upVector));
    
    #ifndef DEBUG_NO_RAIN
    	//Raining
    	#ifndef DEBUG_NO_WATERDROPLET
			float t = floor(mUV.x * MAX_DROPLET_NMBR);
            float r = Hash1D(t);
    
            //used for radius of droplet, smaller float == bigger drop
            float fRadiusSeed = fract(r * 40.0);
            float radius = fRadiusSeed * fRadiusSeed * 0.02 + 0.001;
    
            float fYpos = r * r - clamp(mod(iTime * radius * 2.0, 10.2) - 0.2, 0.0, 1.0);
            radius *= rainingValue;
            vec2 vPos = vec2((t + 0.5) * (1.0 / MAX_DROPLET_NMBR), fYpos * 3.0 - 1.0);
            vec2 vDelta = mUV - vPos;

            const float fInvMaxRadius = 1.0 / (0.02 + 0.001);
            vDelta.x /= (vDelta.y * fInvMaxRadius) * -0.15 + 1.85; // big droplets tear shaped

            vec2 vDeltaNorm = normalize(vDelta);
            float l = length(vDelta);
            if(l < radius)
            {		
                l = l / radius;

                float lz = sqrt(abs(1.0 - l * l));
                vec3 vNormal = l * vDeltaNorm.x * rightVector 
                               + l* vDeltaNorm.y * upVector 
                               - lz * forwardVector;
                vNormal = normalize(vNormal);
                camDirection = refract(camDirection, vNormal, 0.7);
            }
        #endif
    #endif
	return camDirection;
}

// Screen Coordinates
vec2 GetScreenSpaceCoord(const in vec2 fragCoord)
{
	vec2 mUV = (fragCoord.xy / iResolution.xy) * 2.0 - 1.0;
	mUV.x *= iResolution.x / iResolution.y;

	return mUV;	
}

vec3 CamPath(float time)
{
    vec2 p = 1100.0 * vec2(cos(0.23 * time), cos(1.5 + 0.205 * time));
    
	return vec3(p.x, 0.0, p.y);
}
                                     
vec2 CameraPath(float time)
{
    return 100.0 * vec2(cos(0.23 * time), cos(1.5 + 0.205 * time));
	//return vec3(p.x + 55.0, 0.0, -94.0 + p.y);
}
                                     
// Main loop
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
	//Get screen position between 0..1
	vec2 mUV = GetScreenSpaceCoord(fragCoord);
    
	#if DEBUG_CAM == 0
		float time = 10.5;
    
	#elif DEBUG_CAM == 1
		float time = 40.0 * iMouse.x/iResolution.x;
    
    #elif DEBUG_CAM == 2
    	float time = (iTime * 0.2);
    
    #elif DEBUG_CAM == 3
		vec2 vMouse = vec2(0.0);
		if (iMouse.x > 0.0) 
		{
			vMouse = iMouse.xy / iResolution.xy;
			vMouse *= 5.0;
			vMouse.y -= 1.5;
		}

		vec3 rayOrigin = vec3(-0.5 + 20.0 * cos(6.0 * vMouse.x),
							   1.0 + 5.0 * vMouse.y + TERRAIN_HEIGHT,
							   0.5 + 20.0 * sin(6.0 * vMouse.x));
		vec3 camTarget = vec3(0.0, 0.5 + TERRAIN_HEIGHT, 0.0);

		if (rayOrigin.y < (TERRAIN_HEIGHT + TERRAIN_MAXHEIGHT))
		{
			camTarget = vec3(0.0, TERRAIN_HEIGHT + TERRAIN_MAXHEIGHT, 0.0);
			rayOrigin.y = TERRAIN_HEIGHT + TERRAIN_MAXHEIGHT;
		}
	#endif
	
    #if DEBUG_CAM != 3
    	vec3 rayOrigin, camTarget;
    	rayOrigin.xz = CameraPath(time);
		camTarget.xz = CameraPath(time + 8.0);
		rayOrigin.y = TerrainDisplacement(rayOrigin.xz) + 1.0;
    	
		camTarget.y = rayOrigin.y - 5.0;
    #endif
    
    #ifdef CLOUDYCHANGE
		cloudy = sin((iTime + 0.0) * CLOUD_SPEED) * 0.3 + 0.45;
	#endif

    #ifndef DEBUG_NO_RAIN
		rainingValue = max(0.0, (1.48 - cloudy * 2.0)) * (sin((iTime + 43.0) * RAIN_SPEED) * 0.5 + 0.5);
    #endif
	
    #ifdef ToD_ACTIVATED
    	sunDirection = normalize(mix(SUN_STARTPOS, SUN_ENDPOS, EaseInCirc(sin(iTime * SUN_SPEED))));
    	sun_Color.w = max(EaseOutSine(sunDirection.y), 0.0);
    	sun_Color.xyz = HeatmapGradient(EaseOutQuint(sunDirection.y + 0.180));
    #endif

    vec3 rayDirection = GetCameraRayDir(mUV, rayOrigin, camTarget);

	sunAmount = max(dot(sunDirection, rayDirection), 0.0);

    rayIntersect mIntersection;
	vec4 colorResult = RenderScene(mUV, rayOrigin, rayDirection, mIntersection);
    
    //Add rain
    #ifndef DEBUG_NO_RAIN
    	#ifndef DEBUG_NO_RAINBOW
    		colorResult = ApplyRainbow(rayDirection, colorResult);
    	#endif
    #endif
    
    #ifndef DEBUG_NO_DUST
    	ApplyDust(colorResult, rayOrigin, rayDirection, sunDirection, mIntersection);
	#endif
    
    #ifndef DEBUG_NO_POSTPROCESS
		ApplyPostProcess(colorResult, mUV / 2.0, fragCoord);
    #endif
    
    /// DEBUG ///
    /*float debugText = PrintValue((mUV - vec2(-0.65,0.85)) * FONT_SCALE, rainingValue);
    float tt = max(sin(PI * (rainingValue + 1.0) * (rainingValue + 1.0)), 0.0);
    float debugText2= PrintValue((mUV - vec2(0.65,0.85)) * FONT_SCALE, tt);
    colorResult.xyz = mix(colorResult.xyz, abs(debugText + debugText2) - colorResult.xyz, debugText + debugText2);*/
    
    /// TEST ///
    /*float fogFactor = (-pow(2.0, -10.0 * colorResult.w) + 1.0) * rainingValue;
    float topCloudOverlay = abs(max(mUV.y, 0.0)-1.0) * fogFactor;
    colorResult.xyz = mix(colorResult.xyz, vec3(0.6588), topCloudOverlay);*/
    
    fragColor = colorResult;
}