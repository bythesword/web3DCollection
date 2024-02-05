// Jacquemet Matthieu

// Contants --------------------------------------------------------------------------------------------
const float PI   = 3.141592653589793238462643383279502884197169;
const float PI_2 = 1.570796326794896619231321691639751442098585;
const float PI_4 = 0.785398163397448309615660845819875721049292;

const int MAX_STEPS = 100;      // Number of steps
const int WAVE_OCTAVES = 12;
const int WAVE_NORMAL_OCTAVES = 32;
const float EPSILON = 0.001; // Marching epsilon
const float K = 1.0;
const float WATER_LEVEL = 0.0;

// Structure for objects
// v : Field value
// i : Texture index
struct TraceData {
    float v; 
    int i;
};

// Structure for texture
// c : Color
// s : Specular
struct Material {
    vec3 albedo;
    vec3 emissive;
    float roughness;
    float metallic;
};


#define TONEMAP_LINEAR 0 
#define TONEMAP_FILMIC 1
#define TONEMAP_REINHARD 2
#define TONEMAP_FILMIC_REINHARD 3
#define TONEMAP_UNCHARTED2 4
#define TONEMAP_ACES 5

#define MATERIAL_PLASTIC 0
#define MATERIAL_BUOY 1
#define MATERIAL_WATER 2
#define MATERIAL_BEACON 3
#define MATERIAL_METAL 4
#define MATERIAL_BLACK_PLASTIC 5


#define SPECULAR_GGX 0
#define SPECULAR_BLINN 1
#define SPECULAR_BECKMANN 2

#define SPECULAR_MODE SPECULAR_GGX
#define TONEMAP_MODE TONEMAP_ACES // Tone mapping mode
#define _DEBUG 0               // set 1 to enable debuging



#if _DEBUG
    vec3 _debug_color;
    bool _is_debug = false;
    #define DEBUG(color) if (_is_debug) _debug_color = color; // use this to debug a color
    #define CATCH_DEBUG(expr) _is_debug = true; expr ; _is_debug = false;
#else
    #define DEBUG(color)
    #define CATCH_DEBUG(expr) expr // any call to DEBUG(color) in this macro will output de color
#endif

bool _trace_ocean = true;
mat4 buoy_trans;

#define NO_TRACE_OCEAN(expr) _trace_ocean = false; expr ; _trace_ocean = true;

// Primitive functions -----------------------------------------------------------------------------------


// Hashing function
// Returns a random number in [-1,1]
float Hash(float seed)
{
    return fract(sin(seed)*43758.5453 );
}


// Cosine direction 
// seed : Random seed
//    n : Normal
vec3 Cosine( in float seed, in vec3 n)
{
    float u = Hash( 78.233 + seed);
    float v = Hash( 10.873 + seed);

    // Method by fizzer: http://www.amietia.com/lambertnotangent.html
    float a = 6.2831853 * v;
    u = 2.0 * u - 1.0;
    return normalize( n + vec3(sqrt(1.0-u*u)* vec2(cos(a), sin(a)), u) );
}


// saturate
float sat(float x) {

    return clamp(x, 0.0, 1.0);
}

// saturate
vec3 sat(vec3 x) {

    return clamp(x, 0.0, 1.0);
}

// Map value from [Imin, Imax] to [Omin, Omax]
float map(float value, float Imin, float Imax, float Omin, float Omax) {
  
  return Omin + (value - Imin) * (Omax - Omin) / (Imax - Imin);
}

float dot2(in vec2 v) {
    
    return dot(v,v);
}

float dot2(in vec3 v) {
    
    return dot(v,v);
}

// Tranforms --------------------------------------------------------------------

// Translate point p
vec3 Translate(vec3 pos, vec3 p) {

    return p - pos;
}

// Scale point p
vec3 Scale(vec3 scale, vec3 p) {

    return vec3(p.x/scale.x, p.y/scale.y, p.z/scale.z);
}

// Rotate point p around X axis (radians)
vec3 RotateX(float theta, vec3 p)
{   
    float _sin = sin(theta);
    float _cos = cos(theta);

    mat3 M = mat3(  1,     0,     0,
                    0,  _cos, -_sin,
                    0,  _sin,  _cos);
    
    return M*p;
}

// Rotate point p around Y axis (radians)
vec3 RotateY(float theta, vec3 p)
{   
    float _sin = sin(theta);
    float _cos = cos(theta);

    mat3 M = mat3( _cos,    0,  -_sin,
                    0,      1,      0,
                   _sin,    0,   _cos);
    
    return M*p;
}

// Rotate point p around Z axis (radians)
vec3 RotateZ(float theta, vec3 p)
{   
    float _sin = sin(theta);
    float _cos = cos(theta);

    mat3 M = mat3(  _cos,  -_sin,   0,
                    _sin,   _cos,   0,
                    0,      0,      1);

    return M*p;
}

// Rotate point p
vec3 Rotate(vec3 rot, vec3 p)
{
    p = RotateX(rot.x, p);
    p = RotateY(rot.y, p);
    p = RotateZ(rot.z, p);

    return p;
}


// Create scaling matrix
mat3 Scaling(vec3 scale) {

    return mat3(1.0/scale.x,0,          0,
                0,          1.0/scale.y,0,
                0,          0,          1.0/scale.z);
}

// Create rotation matrix for theta angle around X axis (radians)
mat3 RotationX(float theta)
{   
    float _sin = sin(theta);
    float _cos = cos(theta);

    return mat3(    1,     0,     0,
                    0,  _cos, -_sin,
                    0,  _sin,  _cos);
}

// Create rotation matrix for theta angle around Y axis (radians)
mat3 RotationY(float theta)
{   
    float _sin = sin(theta);
    float _cos = cos(theta);

    return mat3( _cos,     0, -_sin,
                    0,     1,     0,
                 _sin,     0,  _cos);
}

// Create rotation matrix for theta angle around Z axis (radians)
mat3 RotationZ(float theta)
{   
    float _sin = sin(theta);
    float _cos = cos(theta);

    return mat3(_cos, -_sin,  0,
                _sin,  _cos,  0,
                0,     0,     1);
}

// Create rotation matrix for the 3 axes (radians)
mat3 Rotation(float x, float y, float z) 
{
    return RotationZ(z) * RotationY(y) * RotationX(x);
}

// Create rotation matrix for the 3 axes (radians)
mat3 Rotation(vec3 rot)
{
    return Rotation(rot.x, rot.y, rot.y);
}

// Create translation matrix
mat4 Translation(vec3 trans) {

    mat4 M = mat4(1.0);
    M[3] = vec4(-trans, 1.0);

    return M;
}

// Create transform matrix 
mat4 Transform(vec3 scale, vec3 rot, vec3 trans) 
{
    return mat4(Scaling(scale) * Rotation(rot)) * Translation(trans);
}

// Transform of point p
vec3 Transform(vec3 scale, vec3 rot, vec3 trans, vec3 p)
{
    p = Scale(scale, p);
    p = Rotate(rot, p);
    p = Translate(trans, p);

    return p;
}

// Texturing and noise ---------------------------------------------------------

// Hashing function
// Returns a random number in [-1,1]
// p : Vector in space
float Hash(in vec3 p)  
{
    p  = fract( p*0.3199+0.152 );
	p *= 17.0;
    return fract( p.x*p.y*p.z*(p.x+p.y+p.z) );
}


// Hashing function
// Returns a random number in [-1,1]
// p : Vector in the plane
float Hash(in vec2 p)  
{
    p  = fract( p*0.3199+0.152 );
	p *= 17.0;
    return fract( p.x*p.y*(p.x+p.y) );
}


// Procedural value noise with cubic interpolation
// x : Point 
float Noise(in vec3 p)
{
    vec3 i = floor(p);
    vec3 f = fract(p);
  
    f = f*f*(3.0-2.0*f);
    // Could use quintic interpolation instead of cubic
    // f = f*f*f*(f*(f*6.0-15.0)+10.0);

    return mix(mix(mix( Hash(i+vec3(0,0,0)), 
                        Hash(i+vec3(1,0,0)),f.x),
                   mix( Hash(i+vec3(0,1,0)), 
                        Hash(i+vec3(1,1,0)),f.x),f.y),
               mix(mix( Hash(i+vec3(0,0,1)), 
                        Hash(i+vec3(1,0,1)),f.x),
                   mix( Hash(i+vec3(0,1,1)), 
                        Hash(i+vec3(1,1,1)),f.x),f.y),f.z);
}

// Procedural value noise with cubic interpolation
// x : Point 
float Noise(in vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
  
    f = f*f*(3.0-2.0*f);
    // Could use quintic interpolation instead of cubic
    // f = f*f*f*(f*(f*6.0-15.0)+10.0);

    return mix(mix( Hash(i+vec2(0,0)), 
                        Hash(i+vec2(1,0)),f.x),
                   mix( Hash(i+vec2(0,1)), 
                        Hash(i+vec2(1,1)),f.x),f.y);
}


// Compute the distance to the Voronoi boundary
// x : Point
// Return (closest distance, second closest, cell id)
vec3 Voronoi( in vec3 x )
{
    vec3 p = floor( x );
    vec3 f = fract( x );

	float id = 0.0;
    vec2 res = vec2( 100.0 );
    for( int k=-1; k<=1; k++ )
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec3 b = vec3( float(i), float(j), float(k) );
        vec3 r = vec3( b ) - f + Hash( p + b );
        float d = dot( r, r );

        if( d < res.x )
        {
			id = dot( p+b, vec3(1.0,57.0,113.0 ) );
            res = vec2( d, res.x );			
        }
        else if( d < res.y )
        {
            res.y = d;
        }
    }

    return vec3( sqrt( res ), abs(id) );
}

// adated from https://iquilezles.org/articles/smoothvoronoi
float SmoothVoronoi( in vec2 x )
{
    vec2 p = floor( x );
    vec2  f = fract( x );

    float res = 0.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 b = vec2( i, j );
        vec2  r = vec2( b ) - f + Hash( p + b );
        float d = dot( r, r );

        res += 1.0/pow( d, 8.0 );
    }
    return pow( 1.0/res, 1.0/16.0 );
}

// based on https://www.shadertoy.com/view/llG3zy
vec3 VoronoiE( in vec3 x )
{
    vec3 n = floor(x);
    vec3 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec3 mr;

    float md = 8.0;
    for( int k=-1; k<=1; k++ )
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec3 g = vec3(float(i),float(j), float(k));
		vec3 o = vec3( g ) - f + Hash( n + g );
        vec3 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
        }
    }

    //----------------------------------
    // second pass: distance to borders,
    // visits only neighbouring cells
    //----------------------------------
    md = 8.0;
    for( int k=-1; k<=1; k++ )
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec3 g = vec3(float(i),float(j), float(k));
		vec3 o = vec3( g ) - f + Hash( n + g );
		vec3 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON ) // skip the same cell
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return vec3( md, mr );
}


// Fractal brownian motion
float Fbm(vec3 p, int octaves) {

    float v = 0.0,  a = 0.5;
    mat3 R = RotationX(.37);

    for (int i = 0; i < octaves; ++i, p*=2.0, a/=2.0) { 
        p *= R;
        v += a * Noise(p);
    }

    return v;
}

// Voronoi fractal brownian motion
vec3 FbmVoronoi(vec3 p, int octave) {

    vec3 v = vec3(0.0);
    float a = 0.5;
    mat3 R = RotationX(.37);

    for (int i = 0; i < octave; ++i, p*=2.0, a/=2.0) {
        p *= R;
        v += a * Voronoi(p);
    }

    return v;
}


// Camera -----------------------------------------------------------------------


// Compute the ray
//      m : Mouse position
//      p : Pixel
// ro, rd : Ray origin and direction
mat3 Camera(in vec2 m, out vec3 ro)
{
    // position camera
    ro=vec3(-3.0,0.0,0.0);
    
    // reset camera position
    // shadertoy initialize mouse position at (0,0)
    if (m == vec2(0))
        m = vec2(0.9, -0.3);
    else
        m = (m*2.0 - vec2(1.0))*3.0;

    m.y = clamp(-m.y, -0.05, PI_2 - 0.1); // clamp camera's y rotation

    ro = RotateY(m.y, ro);
    ro = RotateZ(m.x, ro);

    vec3 ww = normalize(-ro);
    vec3 uu = normalize( cross(ww,vec3(0.0,0.0,1.0) ) );
    vec3 vv = normalize( cross(uu,ww));

	return mat3(uu, vv, ww);
}




// Sphere 
// p : point
// c : center 
// r : radius
TraceData Sphere(vec3 p, vec3 c,float r,int index)
{
    return TraceData(length(p-c)-r,index);
}


TraceData Box(vec3 point, vec3 box, int tex) {

  vec3 q = abs(point) - box;
  float v = length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
  return TraceData(v, tex);
}


// Plane 
// p : point
// n : Normal of plane
// o : Point on plane
TraceData Plane(vec3 p, vec3 n, vec3 o, int index)
{
    return TraceData(dot((p-o),n),index);
}

// Union
// a : field function of left sub-tree
// b : field function of right sub-tree
TraceData Union(TraceData a,TraceData b)
{
    if (a.v < b.v)
        return a;
    else 
        return b;
}

// Intersection, preserve the index of first object
// a, b : field function of left and right sub-trees
TraceData Inter(TraceData a,TraceData b)
{
    if (a.v > b.v)
        return a;
    else 
        return b;
}


// Difference, preserve the index of first object
// a, b : field function of left and right sub-trees
TraceData Diff(TraceData a, TraceData b) {

    return Inter(a, TraceData(-b.v, b.i));
}



// https://iquilezles.org/articles/distfunctionsl

// Union with smoothing
// a : Field function of left sub-tree, 
// b : Field function of right sub-tree
TraceData SmoothUnion( TraceData a, TraceData b, float k ) 
{
    float h = clamp( 0.5 + 0.5*(b.v-a.v)/k, 0.0, 1.0 );
    return TraceData(mix( b.v, a.v, h ) - k*h*(1.0-h), a.i);
}

// Difference with smoothing
// a : Field function of left sub-tree, 
// b : Field function of right sub-tree
TraceData SmoothDiff( TraceData a, TraceData b, float k ) 
{
    float h = clamp( 0.5 - 0.5*(a.v+b.v)/k, 0.0, 1.0 );
    return TraceData(mix( a.v, -b.v, h ) + k*h*(1.0-h), a.i); 
}

// Intersection with smoothing
// a : field function of left sub-tree, 
// b : field function of right sub-tree
TraceData SmoothInter( TraceData a, TraceData b, float k )
{
    float h = clamp( 0.5 - 0.5*(a.v-b.v)/k, 0.0, 1.0 );
    return TraceData(mix( a.v, b.v, h ) + k*h*(1.0-h), a.i);
}


float almostIdentity( float x, float n )
{
    return sqrt(x*x+n);
}

float Overlay(in float a, in float b) {

    if (a  < 0.5)
        return 2.0*a*b;
    else
        return 1.0 - 2.0*(1.0-a)*(1.0-b);
}



TraceData Segment(vec3 point, vec3 a, vec3 b, int tex) {

    vec3 ba = b-a;
    vec3 pa = point-a;
    float t = dot(pa, ba) / dot(ba, ba);
    vec3 c = ba*clamp(t, 0.0, 1.0);
    
    return TraceData(length(pa - c), tex);
}


TraceData Capsule(vec3 point, vec3 a, vec3 b, float radius, int tex) {

    return TraceData(Segment(point, a, b, tex).v - radius, tex);
}


TraceData Cylinder(vec3 point, vec3 a, vec3 b, float radius, int tex) {

    TraceData caps = Capsule(point, a, b, radius, tex);
    TraceData plane = Plane(point, normalize(a-b), b, tex);

    TraceData d = Diff(caps, plane);
    
    return Diff(d, Plane(point, normalize(b-a), a, tex));
}


// based on https://iquilezles.org/articles/distfunctions/distfunctions.html
TraceData RoundedCylinder(vec3 point, float radius, float k, float h, int tex)
{
    vec2 d = vec2( length(point.xy) - 2.0 * radius + k, abs(point.z) - h);
    float v = min(max(d.x,d.y),0.0) + length(max(d,0.0)) - k;

    return TraceData(v, tex);
}


// based on https://iquilezles.org/articles/distfunctions/distfunctions.html
TraceData CappedCone(vec3 p, float h, float r1, float r2, int tex)
{
    vec2 q = vec2( length(p.xy), p.z );
    vec2 k1 = vec2(r2,h);
    vec2 k2 = vec2(r2-r1,2.0*h);
    vec2 ca = vec2(q.x-min(q.x,(q.y<0.0)?r1:r2), abs(q.y)-h);
    vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot2(k2), 0.0, 1.0 );
    float s = (cb.x<0.0 && ca.y<0.0) ? -1.0 : 1.0;
    s *= sqrt(min(dot2(ca),dot2(cb)));

    return TraceData(s, tex);
}

TraceData Circle(vec3 point, float radius, int tex)
{
    vec2 p;
    p.x = length(point.xy) - radius;
    p.y = point.z;

    return TraceData(length(p), tex);
}


TraceData Torus(vec3 point, float radius, float r, int tex)
{
    TraceData vp = Circle(point, radius, tex);
    vp.v -= r;
    return vp;
}


// based on https://iquilezles.org/articles/distfunctions/distfunctions.html
TraceData TriPrism(vec3 p, vec2 h, int tex)
{
    vec3 q = abs(p);
    float v = max(q.x-h.y,max(q.z*0.866025+p.y*0.5,-p.y)-h.x*0.5);
    return TraceData(v, tex);
}


// based on https://iquilezles.org/articles/distfunctions/distfunctions.html
TraceData Ellipsoid(vec3 p, vec3 r, int tex)
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    float v = k0*(k0-1.0)/k1;
    return TraceData(v, tex);
}


float Troichoid(in float x, in float t) {

    float A = 0.5 + 0.4 * sin(0.1 * t * 2.0 * PI);
    return A - 2.0 * A * pow(1.0 - pow(0.5 + 0.5 * sin(x), A + 1.0), 1.0 / (A + 1.0));
}



vec2 Wave(vec2 p, vec2 d, float speed, float frequency, float phase) {

    float x = dot(d, p) * frequency + phase * speed;
    float wave = exp(sin(x) - 1.0);
    float dx = wave * cos(x);
    return vec2(wave, -dx);
}


float Ocean(in vec2 p, in int octaves) {
    
    p *= 0.7;
    
    mat2 R = mat2(RotationZ(12.0));
    vec2 dir = vec2(0,1);

    float freq = 6.0;
    float speed = 2.0;
    float w = 1.0;
    float s = 0.0;
    float h = 0.0;

    for(int i=0; i<octaves; i++){
        vec2 res = Wave(p, dir, speed, freq, iTime);
        p += dir * w * res.y * 0.038;
        h += res.x * w;
        s += w;
        w = mix(w, 0.0, 0.2);
        freq *= 1.18;
        speed *= 1.07;
        dir *= R;
    }
    return h / s * 0.2;
}


TraceData ImplicitOcean(in vec3 p)
{
    float z = Ocean(p.xy, WAVE_OCTAVES);
    float h = p.z - z;

    return TraceData(h, MATERIAL_WATER);
}


TraceData RoundCone(vec3 p, float r1, float r2, float h, int tex)
{
    vec2 q = vec2( length(p.xy), p.z );

    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);
    float k = dot(q,vec2(-b,a));

    float v;

    if( k < 0.0 ) 
        v = length(q) - r1;
    else if( k > a*h )
        v = length(q-vec2(0.0,h)) - r2;
    else
        v = dot(q, vec2(a,b)) - r1;
        
    return TraceData(v, tex);
}


TraceData SolidAngle(vec3 p, vec2 c, float ra, int tex)
{
    // c is the sin/cos of the angle
    vec2 q = vec2( length(p.xy), p.z );
    float l = length(q) - ra;
    float m = length(q - c*clamp(dot(q,c),0.0,ra) );
    float v = max(l,m*sign(c.y*q.x-c.x*q.y));
    return TraceData(v, tex);
}


TraceData Frame(in vec3 p, in vec2 size) {

    TraceData frame = Box(p, vec3(vec2(size),0.02), MATERIAL_BUOY);
    return Diff(frame, Box(p, vec3(vec2(size)-vec2(0.02),0.03), MATERIAL_BUOY));
}


TraceData Propeller(in vec3 p, in int tex) {

    p = RotateY(iTime*10.0, p);
    vec3 pp = p;
    pp.y -= 0.02;
    pp.xz = abs(p.xz);
    pp = RotateY(-PI_4, pp);
    pp = RotateZ(0.4, pp);
    TraceData d = SolidAngle(pp, vec2(0.2955, 0.9553), 0.07, tex);
    d = Inter(d, Box(pp, vec3(0.1,0.002,0.1), tex));

    TraceData tip = Ellipsoid(p+vec3(0,0.05,0.0), vec3(0.02,0.1,0.02), tex);
    tip = Inter(tip, Plane(p, vec3(0,-1,0), vec3(0,-0.005,0), tex));
    d = Union(d, tip);

    return d;  
}


TraceData Anemometer(in vec3 p) {

    TraceData d = Capsule(p, vec3(0,0.05,0), vec3(0,0.05,-0.1), 0.01, MATERIAL_METAL);
    d = Union(d, Ellipsoid(p, vec3(0.02,0.15,0.02), MATERIAL_METAL));
    d = Inter(d, Plane(p, vec3(0,1,0), vec3(0,0.07,0), MATERIAL_METAL));
    d = Union(d, Propeller(p-vec3(0,0.08,0), MATERIAL_BLACK_PLASTIC));

    vec3 tail_p = (p+vec3(0,0.11,0))*vec3(1,2.0,1);
    TraceData tail = TriPrism(tail_p, vec2(0.1,0.002), MATERIAL_METAL);
    tail = Inter(tail, Box(p, vec3(0.1,0.3,0.05), MATERIAL_METAL));

    d = Union(d, tail);

    return d;
}


TraceData Buoy(in vec3 p) {

    p = (buoy_trans * vec4(p,1.0)).xyz;

    // base;
    TraceData base = RoundedCylinder(p, 0.3, 0.05, 0.3, MATERIAL_BUOY);
    base = Union(base, Cylinder(p, vec3(0), vec3(0,0,0.45), 0.17, MATERIAL_BUOY));
    base = Union(base, Cylinder(p, vec3(0,0,0.45), vec3(0,0,0.48), 0.2, MATERIAL_BUOY));

    // rig
    vec3 rig_p = p;
    rig_p.xy = abs(rig_p.xy);
    rig_p = RotateZ(PI_4, rig_p);
    rig_p -= vec3(0.25,0,0.9);
    rig_p = RotateY(0.15, rig_p);
    rig_p = RotateZ(PI_4, rig_p);

    TraceData rig = Box(rig_p, vec3(0.02,0.02,0.6), MATERIAL_BUOY);
    rig = Union(rig, Box(p-vec3(0,0,1.47), vec3(0.13,0.13,0.02), MATERIAL_BUOY));
    rig = Union(rig, Frame(p-vec3(0,0,0.38), vec2(0.3)));
    rig = Union(rig, Frame(p-vec3(0,0,1.1), vec2(0.16)));
    rig.v -= 0.005;

    // beacon
    TraceData beacon = CappedCone(p-vec3(0,0,1.6), 0.05, 0.040, 0.02, MATERIAL_BEACON);
    beacon.v -= 0.02;
    beacon = Union(beacon, Cylinder(p, vec3(0,0,1.45), vec3(0,0,1.55), 0.067, MATERIAL_BLACK_PLASTIC));

    // top part
    vec3 top_part_p = p;
    top_part_p.xy = abs(top_part_p.xy);
    TraceData top_part = Torus(p-vec3(0,0,1.65), 0.25, 0.007, MATERIAL_METAL);
    
    top_part = Union(top_part, Capsule(top_part_p, vec3(0.11,0.13,1.45), 
                            vec3(0.11,0.22,1.65), 0.007, MATERIAL_METAL));

    top_part = Union(top_part, Capsule(top_part_p, vec3(0.13,0.11,1.45), 
                            vec3(0.22,0.11,1.65), 0.007, MATERIAL_METAL));
    
    top_part = Union(top_part, Anemometer(p-vec3(0.25,-0.05,1.75)));
    
    top_part = Union(top_part, Capsule(p, vec3(0.05,0.05,1.2), 
                        vec3(0.05,0.05,1.37), 0.07, MATERIAL_PLASTIC));

    return Union(Union(Union(base, rig), beacon), top_part);
}


// Potential field of the object
// p : point
TraceData Object(vec3 p)
{
    // p.xy += vec2(-2,12);
    // p.xy += vec2(0,5);
    TraceData d = Buoy(p);

    if (_trace_ocean)
        d = Union(d, ImplicitOcean(p));

    return d;
}

// Analysis of the scalar field --------------------------------------------------------------------------

vec3 OceanNormal(in vec2 p, in int octaves, out float h) {

    // vec2 ex = vec2(e, 0);
    const float epsilon = 0.01;

    h = Ocean(p, octaves);

    float dx = h - Ocean(p + vec2(epsilon, 0), octaves);
    float dy = h - Ocean(p + vec2(0, epsilon), octaves);

    return normalize(vec3(dx, dy, epsilon));
}



// Calculate object normal
vec3 ComputeNormal(in vec3 p, inout int matID) {

    float h;
    TraceData vp;
    
    if (matID == 0) {
        vp = Object(p);
        matID = vp.i;
    }

    if (matID == MATERIAL_WATER)
        return OceanNormal(p.xy, WAVE_NORMAL_OCTAVES, h);
    
    float eps = 0.001;
    vec3 n;

    NO_TRACE_OCEAN(

    float v = vp.v;
    n.x = Object( vec3(p.x+eps, p.y, p.z) ).v - v;
    n.y = Object( vec3(p.x, p.y+eps, p.z) ).v - v;
    n.z = Object( vec3(p.x, p.y, p.z+eps) ).v - v;

    )

    return normalize(n);
}


float DistToPlane(in vec3 ro, in vec3 rd, in float h) {

    return -(ro.z-h) / rd.z;
}


// Trace ray using ray marching
// o : ray origin
// u : ray direction
// e : Maximum distance 
// h : hit
// s : Number of steps
float SphereTrace(vec3 o, vec3 u, float e,out bool h,out int s)
{
    h = false;

    // Start at the origin
    float t=0.0;

    for(int i=0; i<MAX_STEPS; i++)
    {
        s=i;
        vec3 p = o+t*u;
        float v = Object(p).v;
        // Hit object
        if (v < 0.0)
        {
            s=i;
            h = true;
            break;
        }
        // Move along ray
        t += max(EPSILON, v / K);
        // Escape marched too far away
        if (t>e)
        {
            break;
        }
    }

    return t;
}

// Lighting ----------------------------------------------------------------------------------------------

struct DirectionalLight {
    vec3 direction;
    vec3 color;
    float energy;
    float shadow_dist;
};

DirectionalLight sun;

// Compute ambient occlusion based on https://www.shadertoy.com/view/3lXyWs
// p : Point
// n : Normal at point
float AmbientOcclusion(vec3 p,vec3 n) {

    const int AO_STEPS = 4;
    const float AO_MAX_DIST = 3.0;

    const float SCALE = AO_MAX_DIST / pow(2.0, float(AO_STEPS))*2.0;
    float ocl = 0.0;

    for(int i = 1; i <= AO_STEPS; ++i) {
        float dist = pow(2.0, float(i)) * SCALE;
        ocl += 1.0 - (max(0.0, Object(p + n * dist).v) / dist);
    }
    
    return min(1.0-(ocl / float(AO_STEPS)),1.0);
    // return pow(abs(Object(p + 2.0 * n)),0.9);
}


// Cast soft shadow based on https://www.shadertoy.com/view/tlBcDK
// p : Point
// l : Point to light vector
// d : Max tracing distance
// r : Softness radius
float Shadow(vec3 p,vec3 l, float d, float r)
{

    float res = 1.0;
    float t = 0.1;

    for (int i = 0; i < MAX_STEPS; ++i) {

        if (res < 0.0 || t > d)
            break;
    
        float h = Object(p+t*l).v;

        res = min(res, r * h / t);
        t += h;    
    }    

    return clamp(res, 0.0, 1.0);
}


// Shading and lighting ---------------------------------------------------------------------------


Material MixMaterial(Material a, Material b, float t) {

    Material mat;

    mat.albedo = mix(a.albedo, b.albedo, t);
    mat.roughness = mix(a.roughness, b.roughness, t);
    mat.emissive = mix(a.emissive, b.emissive, t);
    mat.metallic = mix(a.metallic, b.metallic, t);

    return mat;
}


Material MatWater(in vec3 p, in vec3 n, in vec3 ro, in float v) {

    Material mat;

    const vec3 underwater = vec3(0.1745, 0.3537, 0.7254);
    const vec3 deepwater = vec3(0.0135, 0.0451, 0.1294);
    const vec3 sss_color = vec3(0.0, 0.6, 0.3) * 5.0; // vec3(0.01, 0.33, 0.55);

    vec3 rd = normalize(ro - p);
    float fac = dot(rd, n);

    float sss_fact = max(0.0, acos(dot(rd, sun.direction)));
    sss_fact = smoothstep(1.5, PI, sss_fact);
    sss_fact = pow(sss_fact*fac, 3.0);

    vec3 sss = sss_color * p.z * sss_fact * 100.0;
    vec3 water = mix(underwater, deepwater, sat(fac));

    mat.albedo = vec3(0);
    mat.metallic = 1.0;
    mat.roughness = 0.0;
    mat.emissive = (water + sss*v) * sun.energy * 0.002;

    return mat;
}


Material MatPlastic(vec3 p, vec3 n, in vec3 ro, in float v) {

    Material mat;
    mat.albedo = vec3(1);
    mat.metallic = 0.0;
    mat.roughness = 0.2;
    return mat;
}


Material MatBlackPlastic(vec3 p, vec3 n, in vec3 ro, in float v) {

    Material mat;
    mat.albedo = vec3(0.01);
    mat.metallic = 0.0;
    mat.roughness = 0.7;
    return mat;
}


Material MatBeacon(vec3 p, vec3 n, in vec3 ro, in float v) {

    vec3 flare_pos = vec3(0,0,1.6);
    flare_pos = (inverse(buoy_trans) * vec4(flare_pos, 1.0)).xyz;

    vec3 rd = normalize(p-ro);
    float flare_fact = acos(dot(normalize(flare_pos-ro), rd));
    flare_fact = smoothstep(0.038,0.0, flare_fact) * 20.0;
    flare_fact *= mod(ceil(iTime*0.5), 2.0);
    Material mat;
    mat.albedo = vec3(1,0.3,0);
    mat.metallic = 0.0;
    mat.roughness = 0.5;
    mat.emissive = vec3(1,0,0)*flare_fact;
    return mat;
}


Material MatMetal(vec3 p, vec3 n, in vec3 ro, in float v) {

    Material mat;
    mat.albedo = vec3(1.0);
    mat.metallic = 1.0;
    mat.roughness = 0.3;
    return mat;
}


Material MatBuoy(vec3 p, vec3 n, in vec3 ro, in float v) {

    Material mat;
    
    float var = Fbm(p*8.0, 3)*0.5 + 0.2;
    DEBUG(vec3(var))
    mat.albedo = vec3(1,1,0);
    mat.metallic = 0.0;
    mat.roughness = var;
    return mat;
}


// Compute texture 
// p : Point
// ro : Ray origin
// v : visibility
// n : Normal
Material ComputeMaterial(vec3 p, vec3 n, int matID, vec3 ro, float v)
{
    switch (matID) {
        case MATERIAL_WATER: return MatWater(p, n, ro, v);
        case MATERIAL_BUOY: return MatBuoy(p, n, ro, v);
        case MATERIAL_BEACON: return MatBeacon(p, n, ro, v);
        case MATERIAL_METAL: return MatMetal(p, n, ro, v);
        case MATERIAL_BLACK_PLASTIC: return MatBlackPlastic(p, n, ro, v);
        case MATERIAL_PLASTIC:
        default: return MatPlastic(p, n, ro, v);
    }
}

// Sky --------------------------------------------------------------------------------

// Atmospheric scattering based on preetham's analytical model
// https://github.com/vorg/pragmatic-pbr/blob/master/local_modules/glsl-sky/index.glsl


const float turbidity = 10.0;
const float reileighCoefficient = 2.0;
const float mieCoefficient = 0.005;
const float mieDirectionalG = 0.65;

// constants for atmospheric scattering

const float n = 1.0003; // refractive index of air
const float N = 2.545E25; // number of molecules per unit volume for air at
// 288.15K and 1013mb (sea level -45 celsius)
const float pn = 0.035; // depolatization factor for standard air

// wavelength of used primaries, according to preetham
const vec3 lambda = vec3(680E-9, 550E-9, 450E-9);

// mie stuff
// K coefficient for the primaries
const vec3 K2 = vec3(0.686, 0.678, 0.666);
const float V = 4.0;

// optical length at zenith for molecules
const float rayleighZenithLength = 8.4E3;
const float mieZenithLength = 1.25E3;
const vec3 up = vec3(0.0, 0.0, 1.0);

const float EE = 1000.0;
const float AngularDiameterCos = 0.999956676946448443553574619906976478926848692873900859324;
// 66 arc seconds -> degrees, and the cosine of that

// earth shadow hack
const float cutoffAngle = PI/1.95;
const float steepness = 1.5;


vec3 totalRayleigh(vec3 lambda)
{
    return (8.0 * pow(PI, 3.0) * pow(pow(n, 2.0) - 1.0, 2.0) * (6.0 + 3.0 * pn)) / (3.0 * N * pow(lambda, vec3(4.0)) * (6.0 - 7.0 * pn));
}

float rayleighPhase(float cosTheta)
{
    return (3.0 / (16.0*PI)) * (1.0 + pow(cosTheta, 2.0));
    // return (1.0 / (3.0*PI)) * (1.0 + pow(cosTheta, 2.0));
    // return (3.0 / 4.0) * (1.0 + pow(cosTheta, 2.0));
}

vec3 totalMie(vec3 lambda, vec3 K, float T)
{
    float c = (0.2 * T ) * 10E-18;
    return 0.434 * c * PI * pow((2.0 * PI) / lambda, vec3(V - 2.0)) * K;
}

float hgPhase(float cosTheta, float g)
{
    return (1.0 / (4.0*PI)) * ((1.0 - pow(g, 2.0)) / pow(1.0 - 2.0*g*cosTheta + pow(g, 2.0), 1.5));
}

float sunIntensity(float zenithAngleCos)
{
    return max(0.0, 1.0 - exp(-((cutoffAngle - acos(zenithAngleCos))/steepness)));
}



void AtmosphericScattering(DirectionalLight light, vec3 worldNormal, 
    out float cosTheta, out vec3 Lin, out vec3 Fex) 
{

    vec3 lightDirection = light.direction;
    float lightEnergy = light.energy;

    float sunfade = 1.0-clamp(1.0- exp(light.direction.z / 450000.0) ,0.0,1.0);

    float reileigh = reileighCoefficient - (1.0-sunfade);

    // extinction (absorbtion + out scattering)
    // rayleigh coefficients
    vec3 betaR = totalRayleigh(lambda) * reileigh;

    // mie coefficients
    vec3 betaM = totalMie(lambda, K2, turbidity) * mieCoefficient;

    // optical length
    // cutoff angle at 90 to avoid singularity in next formula.
    //float zenithAngle = acos(max(0.0, dot(up, normalize(vWorldPosition - cameraPos))));
    float zenithAngle = acos(max(0.0, dot(up, worldNormal)));
    float sR = rayleighZenithLength / (cos(zenithAngle) + 0.15 * pow(93.885 - ((zenithAngle * 180.0) / PI), -1.253));
    float sM = mieZenithLength / (cos(zenithAngle) + 0.15 * pow(93.885 - ((zenithAngle * 180.0) / PI), -1.253));


    // combined extinction factor
    Fex = exp(-(betaR * sR + betaM * sM));

    // in scattering
    cosTheta = dot(worldNormal, lightDirection);

    float rPhase = rayleighPhase(cosTheta*0.5+0.5);
    vec3 betaRTheta = betaR * rPhase;

    float mPhase = hgPhase(cosTheta, mieDirectionalG);
    vec3 betaMTheta = betaM * mPhase;


    Lin = pow(lightEnergy * ((betaRTheta + betaMTheta) / (betaR + betaM)) * (1.0 - Fex),vec3(1.5));
    Lin *= mix(vec3(1.0),pow(lightEnergy * ((betaRTheta + betaMTheta) / (betaR + betaM)) * Fex,vec3(1.0/2.0)),clamp(pow(1.0-dot(up, lightDirection),5.0),0.0,1.0));
}


vec3 AtmosphericScattering(DirectionalLight sun, vec3 viewDir) {

    float cosTheta;
    vec3 Lin;
    vec3 Fex;

    AtmosphericScattering(sun, viewDir, cosTheta, Lin, Fex);

    vec3 texColor = Lin*0.04;
    texColor += vec3(0.0,0.001,0.0025)*0.3;

    return texColor;
}


// Get sky color
vec3 Sky(DirectionalLight sun, vec3 viewDir) {

    float CosTheta;
    vec3 Lin;
    vec3 Fex;

    AtmosphericScattering(sun, viewDir, CosTheta, Lin, Fex);

    float sundisk = smoothstep(AngularDiameterCos,AngularDiameterCos+0.00002,CosTheta);
    vec3 L0 = sun.energy * 19000.0 * sundisk * Fex;

    vec3 texColor = (Lin + L0) * 0.04;
    texColor += vec3(0.0,0.001,0.0025)*0.3;

    return texColor;
}


vec3 SkyExtinxion(DirectionalLight light) 
{

    float sunfade = 1.0-clamp(1.0-exp(light.direction.z),0.0,1.0);

    float reileigh = reileighCoefficient - (1.0-sunfade);

    // rayleigh coefficients
    vec3 betaR = totalRayleigh(lambda) * reileigh;

    // mie coefficients
    vec3 betaM = totalMie(lambda, K2, turbidity) * mieCoefficient;

    // sun optical length
    float zenithAngle = acos(max(0.0, dot(up, light.direction)));
    float sR = rayleighZenithLength / (cos(zenithAngle) + 0.15 * pow(93.885 - ((zenithAngle * 180.0) / PI), -1.253));
    float sM = mieZenithLength / (cos(zenithAngle) + 0.15 * pow(93.885 - ((zenithAngle * 180.0) / PI), -1.253));

    // combined extinction factor
    return exp(-(betaR * sR + betaM * sM));
}



vec3 Env(vec3 view, DirectionalLight sun) {

    float cosTheta;
    vec3 Lin;
    vec3 Fex;

    AtmosphericScattering(sun, view, cosTheta, Lin, Fex);

    vec3 L0 = Fex * 0.1;

    vec3 texColor = (Lin+L0) * 0.04;
    texColor += vec3(0.0,0.001,0.0025)*0.3;


    return texColor;
}

// PBR -------------------------------------------------------------------------

//https://gist.github.com/galek/53557375251e1a942dfa

// Get sky ambient color
// sunDirection : Sun direction
// worldNormal : Ray direction
vec3 SkyAmbient(DirectionalLight sun) {

    return Env(normalize(sun.direction*1.8 + vec3(0,0,1)), sun);
}


// phong (lambertian) diffuse term
float phong_diffuse()
{
    return (1.0 / PI);
}


// compute fresnel specular factor for given base specular and product
// product could be NdV or VdH depending on used technique
vec3 fresnel_factor(in vec3 f0, in float product)
{
    return mix(f0, vec3(1.0), pow(1.01 - product, 5.0));
}


// following functions are copies of UE4
// for computing cook-torrance specular lighting terms

float D_blinn(in float roughness, in float NdH)
{
    float m = roughness * roughness;
    float m2 = m * m;
    float n = 2.0 / m2 - 2.0;
    return (n + 2.0) / (2.0 * PI) * pow(NdH, n);
}

float D_beckmann(in float roughness, in float NdH)
{
    float m = roughness * roughness;
    float m2 = m * m;
    float NdH2 = NdH * NdH;
    return exp((NdH2 - 1.0) / (m2 * NdH2)) / (PI * m2 * NdH2 * NdH2);
}

float D_GGX(in float roughness, in float NdH)
{
    float m = roughness * roughness;
    float m2 = m * m;
    float d = (NdH * m2 - NdH) * NdH + 1.0;
    return m2 / (PI * d * d);
}

float G_schlick(in float roughness, in float NdV, in float NdL)
{
    float k = roughness * roughness * 0.5;
    float V = NdV * (1.0 - k) + k;
    float L = NdL * (1.0 - k) + k;
    return 0.25 / (V * L);
}


// cook-torrance specular calculation                      
vec3 cooktorrance_specular(in float NdL, in float NdV, in float NdH, in vec3 specular, in float roughness)
{
#if SPECULAR_MODE == SPECULAR_BLINN
    float D = D_blinn(roughness, NdH);
#elif SPECULAR_MODE == SPECULAR_BECKMANN
    float D = D_beckmann(roughness, NdH);
#elif SPECULAR_MODE == SPECULAR_GGX
    float D = D_GGX(roughness, NdH);
#endif

    float G = G_schlick(roughness, NdV, NdL);

    float rim = mix(1.0 - roughness * 0.9, 1.0, NdV);

    return max((1.0 / rim) * specular * G * D, 0.0);
}


// Picture in picture ------------------------------------------------------------------------------

// Shading according to the number of steps in sphere tracing
// n : Number of steps
vec3 ShadeSteps(int n)
{
   float t=float(n)/(float(MAX_STEPS-1));
   return 0.5+mix(vec3(0.05,0.05,0.5),vec3(0.65,0.39,0.65),t);
}

// Picture in picture
// pixel : Pixel
// pip : Boolean, true if pixel was in sub-picture zone
vec2 Pip(in vec2 pixel, out bool pip)
{
    // Pixel coordinates
    vec2 p = (-iResolution.xy + 2.0*pixel)/iResolution.y;
   if (pip==true)
   {    
    const float fraction=1.0/4.0;
    // Recompute pixel coordinates in sub-picture
    if ((pixel.x<iResolution.x*fraction) && (pixel.y<iResolution.y*fraction))
    {
        p=(-iResolution.xy*fraction + 2.0*pixel)/(iResolution.y*fraction);
        pip=true;
    }
       else
       {
           pip=false;
       }
   }
   return p;
}

// Background color
vec3 background(vec3 r, DirectionalLight sun)
{
    return Sky(sun, r);
    // return mix(vec3(0.452,0.551,0.995),vec3(0.652,0.697,0.995), d.z*0.5+0.5);
}


// Tone mappin -------------------------------------------------------------------

// based on https://www.shadertoy.com/view/ldcSRN

const float W =11.2; // white scale

// filmic (John Hable)


const float A = 0.22; // shoulder strength
const float B = 0.3; // linear strength
const float C = 0.1; // linear angle
const float D = 0.20; // toe strength
const float E = 0.01; // toe numerator
const float F = 0.30; // toe denominator

vec3 LinearToSRGB(vec3 x) 
{
    vec3 t = step(x,vec3(0.0031308));
    return mix(1.055*pow(x, vec3(1./2.4)) - 0.055, 12.92*x, t);
}


vec3 Gamma(vec3 color, float gamma) 
{
    return pow(color, vec3(gamma));
}


vec3 Uncharted2Curve(vec3 x)
{
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;

    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 Uncharted2(vec3 color)
{
    vec3 white_scale = Uncharted2Curve(vec3(W));
    return Uncharted2Curve(color) / white_scale;
}


vec3 ReinhardCurve (vec3 x)
{
	return x / (1.0 + x);
}

vec3 Reinhard(vec3 color) 
{
    vec3 white_scale = ReinhardCurve(vec3(W));
    return ReinhardCurve(color) / white_scale;
}


vec3 FilmicReinhardCurve (vec3 x) 
{
    const float T = 0.01;
    vec3 q = (T + 1.0)*x*x;
	return q / (q + x + T);
}


vec3 FilmicReinhard(vec3 color) 
{
    vec3 white_scale = FilmicReinhardCurve(vec3(W));
    return FilmicReinhardCurve(color) / white_scale;
}


vec3 FilmicCurve(vec3 x)
{
	return ((x*(0.22*x+0.1*0.3)+0.2*0.01)/(x*(0.22*x+0.3)+0.2*0.3))-0.01/0.3;
}

vec3 Filmic(vec3 color)
{
    vec3 white_scale = FilmicCurve(vec3(W));
    return FilmicCurve(color) / white_scale;
}


vec3 ACESFitted(vec3 color) {

    color = pow(color, vec3(0.833));
    color *= 1.07;

    const mat3 ACESInput = mat3(
        0.59719, 0.35458, 0.04823,
        0.07600, 0.90834, 0.01566,
        0.02840, 0.13383, 0.83777
    );

    const mat3 ACESOutput = mat3(
        1.60475, -0.53108, -0.07367,
        -0.10208,  1.10813, -0.00605,
        -0.00327, -0.07276,  1.07602
    );


    color = color * ACESInput;

    // Apply RRT and ODT
    vec3 a = color * (color + 0.0245786) - 0.000090537;
    vec3 b = color * (0.983729 * color + 0.4329510) + 0.38081;
    color = a/b;

    return color * ACESOutput;
}


vec3 ToneMapping(vec3 color) {

    color = color*0.2;

    #if TONEMAP_MODE == TONEMAP_FILMIC
        color = Filmic(color);
    #elif TONEMAP_MODE == TONEMAP_REINHARD
        color = Reinhard(color);
    #elif TONEMAP_MODE == TONEMAP_FILMIC_REINHARD
        color = FilmicReinhard(color);
    #elif TONEMAP_MODE == TONEMAP_UNCHARTED2
        color = Uncharted2(color);
    #elif TONEMAP_MODE == TONEMAP_ACES
        color = ACESFitted(color);
    #endif

    color = clamp(LinearToSRGB(color), 0.0, 1.0);

    return color;
}

// Compute Blinn-Phong specular
// l : Vector to light
// n : Normal at point
// r : View ray direction
// k : glossyness
float Specular(vec3 l, vec3 n, vec3 r, float k)
{
    vec3 half_dir = normalize(l + r);
    float spec_angle = max(dot(half_dir, n), 0.0);
    return pow(spec_angle, k);

    // Phong
//     vec3 ref = reflect(r, n);
//     float c = max(dot(ref, r), 0.0);
//     return pow(c, k/4.0);
}


// Compute lighting
// sun : Sun data
// mat : Material data
// p : Point on surface
// rd : View ray direction
// n : Normal at Point
// reflection : Computed reflection
// clearcoat : Computed clearcoat reflection
 vec3 Shade(DirectionalLight sun, Material mat, vec3 p, vec3 rd, vec3 n, 
    vec3 reflection, float v) 
{

    // Ambient color
    vec3 ambient = SkyAmbient(sun) * 0.7;

    // Ambient occlusion
    NO_TRACE_OCEAN(ambient *= AmbientOcclusion(p, n))

    // vec3 diffuse = ambient;
    vec3 specular = mix(vec3(0.02), mat.albedo, mat.metallic);


    vec3 L = sun.direction;
    vec3 N = n;
    vec3 V = -rd;
    vec3 H = normalize(V+L);

    float NdL = max(0.001, dot(N, L));
    float NdV = max(0.001, dot(N, V));
    float NdH = max(0.001, dot(N, H));
    float HdV = max(0.001, dot(H, V));


    // specular reflectance with COOK-TORRANCE
    vec3 specfresnel = fresnel_factor(specular, HdV);
    vec3 specref = cooktorrance_specular(NdL, NdV, NdH, specfresnel, mat.roughness);

    specref *= vec3(NdL) * 10.0;

    // diffuse is common for any model
    vec3 diffref = (vec3(1.0) - specfresnel) * phong_diffuse() * NdL;
    
    // compute lighting
    vec3 reflected_light = vec3(0);
    vec3 diffuse_light = vec3(0);

    // point light
    vec3 light_color = sun.color * sun.energy * 0.01;
    reflected_light += specref * light_color * v;
    diffuse_light += diffref * light_color * v;

    reflected_light += min(vec3(0.99), fresnel_factor(specular, NdV)) * reflection;

    diffuse_light += ambient * (1.0 / PI);

    // final result
    vec3 result = diffuse_light * mix(mat.albedo, vec3(0.0), mat.metallic);
    result += reflected_light;
    result += mat.emissive;

    return result;
}


// Sample color from ray
// sun : Sun light
// ro : Ray origin
// rd : Ray direction
// steps : Number of trace steps
vec3 Render(DirectionalLight sun, vec3 ro, vec3 rd, out int steps) {

    // Hit and number of steps
    bool hit;
    int s;
    
    // primary ray
    CATCH_DEBUG(float t = SphereTrace(ro, rd, 1000.0, hit, s));
    steps += s;

    if (!hit && rd.z > 0.0)
        return background(rd, sun);
    
    vec3 pt, n;
    int matID = 0;
    Material mat;
    float h;
    
    if (!hit && rd.z < 0.0) {
        t = DistToPlane(ro, rd, WATER_LEVEL);
        matID = MATERIAL_WATER;
    }
    
    pt = ro + t * rd;
    n = ComputeNormal(pt, matID);
    
    NO_TRACE_OCEAN(float v = Shadow(pt+n*0.001, sun.direction, 
                                    sun.shadow_dist, 20.0))

    CATCH_DEBUG(mat = ComputeMaterial(pt, n, matID, ro, v));

    vec3 reflect_dir = reflect(rd, n);
    vec3 reflection = vec3(0);

    // reflection
    if (mat.roughness == 0.0) {

        // secondary ray
        vec3 start = pt+n*0.01;
        bool hit;
        NO_TRACE_OCEAN(float t = SphereTrace(start, reflect_dir, 1000.0, hit, s))
        steps += s;

        if (hit) {
            vec3 rpt = pt + t * reflect_dir;
            
            int matID;
            vec3 rn = ComputeNormal(rpt, matID);

            Material rmat = ComputeMaterial(rpt, rn, matID, pt, 1.0);

            vec3 sec_reflection = Env(reflect(reflect_dir, rn), sun);
            
            reflection = Shade(sun, rmat, rpt, reflect_dir, 
                                rn, sec_reflection, 1.0);
        }
        else
            reflection = Env(reflect_dir, sun);
    }
    else if (hit) {
        float r = 1.0/max(mat.roughness, 0.00001);
        NO_TRACE_OCEAN(float v = Shadow(pt+n*0.01, reflect_dir, 1000.0, r));
        reflection = mix(SkyAmbient(sun)*0.01, Env(reflect_dir, sun), v);
    } else 
        reflection = Env(reflect_dir, sun);


    vec3 color = Shade(sun, mat, pt, rd, n, reflection, v);

    color = mix(color, Env(rd, sun), sat(t*t*0.0002));

    return color;
}


vec2 RayDirection(in vec2 pixel, inout bool pip)
{
    // Pixel coordinates
    vec2 p = (-iResolution.xy + 2.0*pixel)/iResolution.y;
   if (pip==true)
   {
    const float fraction=1.0/4.0;
    // Picture in picture
    if ((pixel.x<iResolution.x*fraction) && (pixel.y<iResolution.y*fraction))
    {
        p=(-iResolution.xy*fraction + 2.0*pixel)/(iResolution.y*fraction);
        pip=true;
    }
       else
       {
           pip=false;
       }
   }
   return p;
}


// Image
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Picture in picture on
    bool pip=false;

    // Mouse
    vec2 m=iMouse.xy/iResolution.xy;

    // Camera
    vec3 ro;
    mat3 cam = Camera(m, ro);

    vec2 p = RayDirection(fragCoord, pip);
   
    // Camera ray    
    vec3 rd = cam * normalize(vec3(p,1.0));


    // Shade background
    sun.direction = normalize(vec3(0.2, -0.5, 0.1));
    sun.color = SkyExtinxion(sun)* 19.0;
    sun.energy = sunIntensity(sun.direction.z) * EE;
    sun.shadow_dist = 100.0;

    
    // compute buoy transform
    float h;
    vec3 buoy_nor = OceanNormal(vec2(0), 4, h);
    buoy_nor.xy *= -0.2;

    buoy_nor = normalize(buoy_nor);

    vec3 buoy_tan = cross(buoy_nor, vec3(1,0,0));
    vec3 buoy_bit = cross(buoy_tan, buoy_nor);

    buoy_trans = mat4(mat3(buoy_bit, buoy_tan, buoy_nor));
    buoy_trans[3].z = h * -0.4 + 0.1;

    int s;

    vec3 rgb = Render(sun, ro, rd, s);

    rgb = ToneMapping(rgb);

    // Uncomment this line to shade image with false colors representing the number of steps
    if (pip==true)
        rgb = ShadeSteps(s); 


#if _DEBUG
    rgb = _debug_color;
#endif
    fragColor = vec4(rgb, 1.0);
}

