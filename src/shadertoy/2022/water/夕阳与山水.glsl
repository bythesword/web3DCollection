//#version 330 core
#define PI 3.1415926535897932384626433832795
#define inf 9e100

/**
	Generates terrain with fbm noise.
	Volcano is always centered in the origin.
	Volcano elevation is calculated as e^(-sqrt(x^2+y^2)), and summed to fbm.
	Changingin the seed generates a new archipelagus.
	Drag the mouse to move the camera.
	All the variables declared here at the start are parameters that can be 
	changed, I hope they are not that obscure.

	Special thanks to Inigo Quielez, his awesome work is truly inspiring,
	and his tutorials are amazing.
	Thanks also to Frankenburg's water shader, and nimitz's lens flare shader.
	Finally, thanks also to my sister who gave some fundamental artistic advice.

	Here is a short pdf I made to explain a bit better how this was made:
	https://drive.google.com/open?id=1U4zynGm8o4i8VodnaWTOjgoayj9rhega

	This is my first shader so please understand if it's not
	that orignal or well done...
*/



float seed = 104.0;             //input a float < 10000.0 to generate terrain 


// --------- TERRAIN GENERATION OPTIONS
int fbm_octaves = 11;            //octaves for the terrain generator. More octaves = more details
float elevate = 2.0;             //height of volcano
float crater_width = 0.05;       //width and depth of the crater
float large = 1.5;               //width of volcano (higher = thinner)
float water_level = 130.0;       //starting height of water. Set to a very low number to remove water
float wavegain = 0.3;            //roughness of the water


// ----------- RENDERING OPTIONS -----------
int AA = 1;                      //antialias
float camera_distance = 250.0;   //distance of camera from origin
float jitter_factor = 2.0;       //jitter when too close to object
int max_raymarching_steps = 128; //increase if you see artifacts near object edges
float max_distance = 5000.0;     //max rendering distance
float step_size = 0.75;          //decrease if you see artifacts near slopes


//toggle between 0 and 1
int shadows = 1; 
int water_refl = 1; 
int clouds = 0;
int gamma = 0;
int lens_flare = 0;
int hq_water = 1;
int lava = 0;					 //just a red light on the crater 

int sha_octaves = 6;             //octaves for shadows
float sha_stepsize = 3.0;        //shadow step size. Decrease if there are artifacts in shadows 
float softshadows = 7.0;         //Increase to have harder shadows

float fog = 0.3;                 //fog level. Set between 0 and 1
float sun_dispersion = 0.5;	     //sun dispersion in fog




const vec3  kSunDir = normalize(vec3(-0.624695,0.168521,-0.624695)); 
vec3 _LightDir = kSunDir*100000.0; 
vec3 _CameraDir = normalize(vec3(7,1.2,5))*5.00;

float ka = 0.05;                 //ambient light
float kd = 0.2;			         //diffuse light





// --  Value noise by Inigo Quielez: https://iquilezles.org/articles/morenoise
float hash2( vec2 p ) {
    p  = 50.0*fract( p*0.3183099 );
    return fract( p.x*p.y*(p.x+p.y) );
}

float noise( in vec2 x ) {
    vec2 p = floor(x);
    vec2 w = fract(x);
    vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);

    float a = hash2(p+vec2(0,0));
    float b = hash2(p+vec2(1,0));
    float c = hash2(p+vec2(0,1));
    float d = hash2(p+vec2(1,1));

    return -1.0+2.0*( a + (b-a)*u.x + (c-a)*u.y + (a - b - c + d)*u.x*u.y );
}

vec3 noised2( in vec2 x ) {
    vec2 p = floor(x);
    vec2 w = fract(x);

    vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec2 du = 30.0*w*w*(w*(w-2.0)+1.0);

    float a = hash2(p+vec2(0,0));
    float b = hash2(p+vec2(1,0));
    float c = hash2(p+vec2(0,1));
    float d = hash2(p+vec2(1,1));

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k4 = a - b - c + d;

    return vec3( -1.0+2.0*(k0 + k1*u.x + k2*u.y + k4*u.x*u.y), 
                      2.0* du * vec2( k1 + k4*u.y,
                                      k2 + k4*u.x ) );
}


mat2 m2 = mat2(  0.80,  0.60, -0.60,  0.80 );
mat2 m2i = mat2( 0.80, -0.60, 0.60,  0.80 );

float fbm( in vec2 x, int octaves) {
    float f = 1.9;
    float s = 0.50;
    float a = 0.0;
    float b = 0.5;
    for( int i=0; i< octaves; i++ )
    {
        float n = noise(x);
        a += b*n;
        b *= s;
        x = f*m2*x;
    }
	return a;
}

//returns fbm with derivatives
vec3 fbmd( in vec2 x , int octaves) {
    float f = 1.9;
    float s = 0.50;
    float a = 0.0;
    float b = 0.5;
    vec2  d = vec2(0.0);
    mat2  m = mat2(1.0,0.0,0.0,1.0);
    for( int i=0; i< octaves; i++ )
    {
        vec3 n = noised2(x);
        a += b*n.x;          // accumulate values		
        d += b*m*n.yz;       // accumulate derivatives
        b *= s;
        x = f*m2*x;
        m = f*m2i*m;
    }
	return vec3( a, d );
}


// -- Sky adapted from Inigo quielez's shader: https://www.shadertoy.com/view/4ttSWf
vec3 renderSky( in vec3 ro, in vec3 rd ) {
    vec3 col = vec3(0);


    // clouds
    if (clouds > 0) {
        float t = (1000.0-ro.y)/rd.y;
        if( t>0.0 ) {
            vec2 uv = (ro+t*rd).xz;
            float cl = fbm( uv*0.001 + vec2(-100.0, 20.0) , fbm_octaves);
            float dl = smoothstep(-0.2,0.6,cl);
            col = mix( col, vec3(1.0), 0.4*dl );
        }
    }

    float sundot = clamp(dot(kSunDir,rd), 0.0, 1.0 );
    col += mix(vec3(1.0,0.1,0.1)*1.1, normalize(vec3(172, 133, 102)), rd.y*rd.y + 0.588*(1.0 - sundot));

    col += normalize(vec3(250,206,150))*pow(sundot,2048.0)*20.0;
    col += normalize(vec3(250,206,14)) *pow(sundot,64.0);
    col += normalize(vec3(244,162,5))  *pow(sundot,5.0);

    //horizon
    col = mix( col, 0.5*normalize(vec3(118,34,8)), pow( 1.0-max(rd.y + 0.1,0.0), 10.0 ) );

    return col;
}



//length of a vector with derivatives
vec3 length_d(vec2 p) {
    vec3 res = vec3(0);
    float size = p.x*p.x + p.y*p.y;
    res.x = pow(size, 0.5);
    res.y = pow(size, -0.5)*p.x;
    res.z = pow(size, -0.5)*p.y;
    return res;
}


float length_n(vec2 p, float exponent) {
    float size = pow(p.x, exponent) + pow(p.y, exponent);
    return pow(size, 1.0/exponent);
}

float terrainMap(in vec2 p, int octaves) {
    float sca = 0.0010;
    float amp = 300.0;
    p *= sca;
    float e = fbm(p + vec2(1.0, -2.0)*(1.0 + seed), octaves);

    float length2 = length(p);
    e += elevate*exp(-large*(max(length2, crater_width))); //volcano
    if (length2 < crater_width) {
        e -= 2.0 * (crater_width - length2); //crater
    }

    e *= amp;
    return e;
}

//return terrain map with normal coordinates 
vec4 terrainMapD( in vec2 p ) {
    float sca = 0.0010;
    float amp = 300.0;
    p *= sca;
    vec3 e = fbmd( p + vec2(1.0,-2.0)*(1.0 + seed) , fbm_octaves);


    vec3 length2 = length_d(p);
    e.x += elevate*exp(-large*(max(length2.x, crater_width))); //volcano
    if (length2.x < crater_width) {
        e.x -= 2.0 * (crater_width - length2.x); //crater
        //crater derivatives
        e.y -= 2.0 * -length2.y;
        e.z -= 2.0 * -length2.z;
    }
    else {
        //volcano derivatives
        e.y += elevate*exp(-large*(max(length2.x, crater_width)))*(-large*length2.y);
        e.z += elevate*exp(-large*(max(length2.x, crater_width)))*(-large*length2.z);
    }

    e.x *= amp;
    e.yz *= amp*sca;
    return vec4( e.x, normalize( vec3(-e.y,1.0,-e.z) ) );
}



// -- Fog function adapted from Inigo's tutorial: https://iquilezles.org/articles/fog
vec3 applyFog(in vec3 rgb, in float dist, in vec3 ro, in vec3  rayDir, in vec3  sunDir )
{
    float fogAmount = 1.0 - exp( -(dist*dist/max_distance)*((fog*fog) / 200.0) );
    float sunAmount = clamp( dot( rayDir, sunDir ), 0.0, 1.0 );
    vec3 fogColor  = mix( 0.5*normalize(vec3(118,34,8)), vec3(1.0,0.9,0.1), 
            pow(sunAmount, (0.5/(sun_dispersion)) * 8.0 * ((dist) / 1000.0)) );
    return mix(rgb, fogColor, fogAmount );
}


// -- Lens Flare adapted from nimitz's shader: https://www.shadertoy.com/view/XtS3DD
float pent(in vec2 p) {
    vec2 q = abs(p);
    return max(max(q.x*1.176-p.y*0.385, q.x*0.727+p.y), -p.y*1.237)*1.;
}

float circle(in vec2 p){
    return length(p);
}

vec3 flare(vec2 p, vec2 pos) 
{
	vec2 q = p-pos;
    vec2 pds = p*(length(p))*0.75;
	float a = atan(q.x,q.y);
    float rz = 0.0;

    vec2 p2 = mix(p,pds,-.5); //Reverse distort
    rz += max(0.01-pow(circle(p2 - 0.2*pos),1.7),.0)*3.0;
    rz += max(0.01-pow(pent(p2 + 0.4*pos),2.2),.0)*3.0;
    rz += max(0.01-pow(pent(-(p2 + 1.*pos)),2.5),.0)*5.0;
    rz += max(0.01-pow(pent(-(p2 - .5*pos)),2.),.0)*4.0;
    rz += max(0.01-pow(circle(-(p2 + 1.8*pos)),3.0),.0)*3.0;

    rz *= (1.0 - length(q));
    rz *= 5.0;

    return vec3(clamp(rz,0.,1.));
}


// -- Water adapted from frankenburg's shader: https://www.shadertoy.com/view/4sXGRM
float large_waveheight = 0.7; 
float small_waveheight = 1.0; 
vec3 watercolor  = vec3(0.2, 0.25, 0.3);
float water( vec2 p ) {
    vec2 shift2 = 0.0003*vec2( iTime*190.0*2.0, -iTime*130.0*2.0 );

    float wave = 0.0;
    wave += sin(p.x * 0.021  +                     shift2.x)         * 4.5;
    wave += sin(p.x * 0.0172 +p.y        * 0.010 + shift2.x * 1.121) * 4.0;
    wave -= sin(p.x * 0.00104+p.y        * 0.005 + shift2.x * 0.121) * 4.0;
    wave += sin(p.x * 0.02221+p.y        * 0.01233+shift2.x * 3.437) * 5.0;
    wave += sin(p.x * 0.03112+p.y        * 0.01122+shift2.x * 4.269) * 2.5 ;
    wave *= large_waveheight;
    wave -= fbm(p*0.004-shift2*.5 + vec2(1.0, -2.0), fbm_octaves)*small_waveheight*24.;

    return water_level + wave;
}


// Pseudo-random number generator adapted from: lumina.sourceforge.net/Tutorials/Noise.html
float rand(vec2 co){
    return fract(cos(dot(co,vec2(4.898,7.23))) * 23421.631);
}


vec4 raymarch_terrain(vec3 ro, vec3 rd) {
    vec4 ret = vec4(0,0,0,0);

    float t = 0.01; // current distance traveled along ray
    for (int i = 0; i < max_raymarching_steps; ++i) {
        if (t > max_distance) break;
        vec3 p = ro + rd * t; //point hit on the surface
        float terrain = terrainMap(p.xz, fbm_octaves);
        float d = p.y - terrain;


        if (p.y < water_level) {
            if (hq_water == 0) return vec4(0,0,1,0);
            t = ((water_level - ro.y) / rd.y); //adjust position to remove water artefacts
            p = ro + rd * t;
            vec3 col = vec3(0.0);

            // calculate water-mirror
            vec2 xdiff = vec2(0.1, 0.0)*wavegain*4.;
            vec2 ydiff = vec2(0.0, 0.1)*wavegain*4.;

            // get the reflected ray direction
            vec3 rd_bis = reflect(rd, normalize(vec3(water(p.xz-xdiff) - water(p.xz+xdiff), 1.0, water(p.xz-ydiff) - water(p.xz+ydiff))));
            float refl = 1.0-clamp(dot(rd_bis,vec3(0.0, 1.0, 0.0)),0.0,1.0);


            //reflection
            vec3 refl_col = renderSky(p, rd_bis);
            // raymarch to see if the sky is visible
            if (rd_bis.y > 0.0 && water_refl > 0) {
                vec3 myro = p;
                vec3 myrd = rd_bis;
                float tt = 0.1;
                vec3 pp = p;
                for (int j = 0; j < max_raymarching_steps; j++) {
                    if (tt > max_distance)  break;
                    pp = myro + myrd * tt; 
                    terrain = terrainMap(pp.xz, sha_octaves);
                    d = pp.y - terrain;
                    if (d < 0.001*tt) {
                        refl_col = normalize(vec3(114, 53, 6))/2.0;
                        break;
                    }
                    tt += sha_stepsize*step_size * d;
                }
            }



            col = refl*0.5*refl_col;
            col += watercolor;

            col -= vec3(.3,.3,.3);

            terrain = terrainMap(p.xz, fbm_octaves);

            col += vec3(.15,.15,.15)*(clamp(20.0 / (water_level - terrain), 0.0, 1.0));
            col = applyFog(col, t, ro, rd, kSunDir);

            //fading away in the distance
            float alpha = pow(t / max_distance, 3.0);
            if (alpha > 0.2) 
                col = mix(col, renderSky(ro, rd), alpha - 0.2);


            return vec4(col, 1);

        }


        if (d < 0.002 * t) { 
            vec3 n = terrainMapD(p.xz).yzw;
            vec3 col = vec3(ka);

            //lights
            float diffuse_sun = clamp(kd * dot(kSunDir, n), 0.0, 1.0);
            float indirect_sun = clamp(dot(n, normalize(kSunDir*vec3(-1.0, 0.0, -1.0))), 0.0, 1.0);
            float diffuse_sky = clamp(0.5 + 0.5*n.y, 0.0, 1.0);

            //shadows
            float shadow = 1.0;
            if (diffuse_sun > 0.01 && shadows > 0) {
                vec3 myro = p + kSunDir*15.0; //Start a bit higher than terrain. Helps when sha_octaves is low (< 5)
                vec3 myrd = kSunDir;
                float tt = 0.1;
                for (int j = 0; j < max_raymarching_steps; j++) {
                    if (tt > max_distance) break;
                    p = myro + myrd * tt; 
                    terrain = terrainMap(p.xz, sha_octaves);
                    d = p.y - terrain;
                    if (d < 0.001*tt) {
                        shadow = 0.0;
                        break;
                    }
                    shadow = min(shadow, softshadows*(d/tt));
                    tt += sha_stepsize*step_size * d;
                }
            }



            col += diffuse_sun * normalize(vec3(250,206,14))* pow(vec3(shadow), vec3(1.0, 1.2, 1.5));
            col += indirect_sun *  vec3(0.40, 0.28, 0) * 0.14; 
            col += diffuse_sky * vec3(0.596, 0.182, 0.086) * 0.2;

            if (lava > 0) {
                float lava_height = terrainMap(vec2(0), fbm_octaves) + 50.0;
                float diffuse_lava = 0.031*clamp(dot(n, normalize(vec3(0.0,lava_height, 0.0))), 0.0, 1.0);
                col += 100000.0 * 0.013 * vec3(1,.1,.1) / (p.x*p.x + (p.y-lava_height)*(p.y-lava_height) + p.z*p.z);
            }

            // far away terrain fades away
            float alpha = pow(t / max_distance, 3.0);
            if (alpha > 0.2) 
                col = mix(col, renderSky(ro, rd), alpha - 0.2);
            
            col = applyFog(col, t, ro, rd, kSunDir);

            return vec4(col, 1);
        }

        t += d*step_size;
    }

    vec3 sky = renderSky(ro, rd); 
    return vec4(sky.xyz, 1.0);
}

//Adapted from: https://github.com/zackpudil/raymarcher/blob/master/src/shaders/scenes/earf_day.frag
mat3 camera(vec3 e, vec3 l) {
    vec3 f = normalize(l - e);
    vec3 r = cross(vec3(0, 1, 0), f);
    vec3 u = cross(f, r);
    
    return mat3(r, u, f);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    
    //camera movement
    if (iMouse.y != 0.0) {
        _CameraDir.y =  max(5.0*(iMouse.y / iResolution.y - 0.5)*3.0,0.70); // to avoid going underwater
        _CameraDir.z = -5.0*cos(iMouse.x / iResolution.x * 2.0 * PI);
        _CameraDir.x = -5.0*sin(iMouse.x / iResolution.x * 2.0 * PI);
    }
    _CameraDir *= camera_distance;

    fragColor = vec4(0);

    for (int m = 0; m < AA; m++) {
        for (int n = 0; n < AA; n++) {
            vec2 uv = -1.0 + 2.0*(fragCoord.xy/iResolution.xy) + vec2(float(n) * (1.0/(iResolution.x * float(AA))), float(m) * (1.0/(iResolution.y * float(AA))));
            uv.x *= iResolution.x/iResolution.y;

            vec3 rd = camera(_CameraDir, vec3(0.0, 200.0, 0.0))*normalize(vec3(uv, 2.0));

            fragColor += raymarch_terrain(_CameraDir, rd);

            if (lens_flare > 0) {
                vec3 sunpos = inverse(camera(_CameraDir, vec3(0)))*kSunDir;
                if (sunpos.z > 0.0) fragColor += vec4(flare(uv, sunpos.xy), 1);
            }
        }
    }

    fragColor /= float(AA*AA);
    if (gamma > 0) fragColor = pow(fragColor, vec4(1.0 / 2.2));
}    
