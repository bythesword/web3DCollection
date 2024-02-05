#iChannel0 "file://noise.rock.jpg"
#define MAX_RAYMARCHING_COUNT 96
#define PRECISION 0.00003
#define FAR 100.
#define mouse (iMouse.xy / iResolution.xy)
#define time iTime

const mat2 m2 = mat2( 0.60, -0.80, 0.80, 0.60 );
vec3 hash( vec3 p ) // replace this by something better
{
	p = vec3( dot(p,vec3(127.1,311.7, 74.7)),
			  dot(p,vec3(269.5,183.3,246.1)),
			  dot(p,vec3(113.5,271.9,124.6)));

	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}
float hash( float n ) { return fract(sin(n)*753.5453123); }
float noise( in vec3 p )
{
    vec3 i = floor( p );
    vec3 f = fract( p );
	
	vec3 u = f*f*(3.0-2.0*f);

    return mix( mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                          dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                          dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                          dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                          dot( hash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

vec4 noised( in vec3 x )
{
    vec3 p = floor(x);
    vec3 w = fract(x);
	vec3 u = w*w*(3.0-2.0*w);
    vec3 du = 6.0*w*(1.0-w);
    
    float n = p.x + p.y*157.0 + 113.0*p.z;
    
    float a = hash(n+  0.0);
    float b = hash(n+  1.0);
    float c = hash(n+157.0);
    float d = hash(n+158.0);
    float e = hash(n+113.0);
	float f = hash(n+114.0);
    float g = hash(n+270.0);
    float h = hash(n+271.0);
	
    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return vec4( k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z, 
                 du * (vec3(k1,k2,k3) + u.yzx*vec3(k4,k5,k6) + u.zxy*vec3(k6,k4,k5) + k7*u.yzx*u.zxy ));
}
float tri(in float x){return abs(fract(x)-.5);}
vec3 tri3(in vec3 p){return vec3( tri(p.z+tri(p.y*1.)), tri(p.z+tri(p.x*1.)), tri(p.y+tri(p.x*1.)));}
float triNoise3d(in vec3 p, in float spd)
{
    float z=1.4;
	float rz = 0.;
    vec3 bp = p;
	for (float i=0.; i<=3.; i++ )
	{
        vec3 dg = tri3(bp*2.);
        p += (dg+iTime*spd);

        bp *= 1.8;
		z *= 1.5;
		p *= 1.2;
        //p.xz*= m2;
        
        rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
        bp += 0.14;
	}
	return rz;
}

float fogmap(in vec3 p, in float d)
{
    p.x += iTime*1.1;
    p.z += sin(p.x*.5);
    return triNoise3d(p*2.2/(d+20.),0.2)*(1.-smoothstep(0.,1.8,p.y));
}

vec3 fog(in vec3 col, in vec3 ro, in vec3 rd, in float mt)
{
    float d = .1;
    for(int i=0; i<30; i++)
    {
        vec3  pos = ro + rd*d;
        float rz = fogmap(pos, d);
		float grd =  clamp((rz - fogmap(pos+.8-float(i)*0.1,d))*3., 0.1, 1. );
        vec3 col2 = (vec3(1.)*.5 + .5*vec3(1.)*(1.7-grd))*0.55;
        col = mix(col,col2,clamp(rz*smoothstep(d-0.4,d+2.+d*.75,mt),0.,1.) );
        d *= 1.5+0.3;
        if (d>mt)break;
    }
    return col;
}

//------------------------------------------------------------------

// Tri-Planar blending function. Based on an old Nvidia writeup:
// GPU Gems 3 - Ryan Geiss: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch01.html
vec3 tex3D( sampler2D tex, in vec3 p, in vec3 n ){
   
    n = max(n*n, 0.001);
    n /= (n.x + n.y + n.z );  
    
	return (texture(tex, p.yz)*n.x + texture(tex, p.zx)*n.y + texture(tex, p.xy)*n.z).xyz;
}

mat3 setCamera(vec3 ro, vec3 lookAt) {
	vec3 cw = normalize(lookAt-ro);
	vec3 cp = vec3(0.0, 1.0, 0.0);
	vec3 cu = normalize( cross(cw,cp) );
	vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

float sminP(float a, float b , float s){
    
    float h = clamp( 0.5 + 0.5*(b-a)/s, 0. , 1.);
    return mix(b, a, h) - h*(1.0-h)*s;
}

vec2 map(vec3 p, float concernWater) {
    float d = 0.0;
    vec2 q = p.xz*0.02;
    float h = 0.;
    float s = 1.0;
    /*for (int i = 0 ; i < 10 ; i++) {
        h += s*noise(vec3(q, 1.));
        q = m2 * q * 2.01;
        s *= 0.49;
    }
    h *= 23.;
	*/
    for (int i = 0 ; i < 9 ; i++) {
        h += s*noise(vec3(q, 1.));
        q = m2 * q * 3.01;
        s *= 0.334;
    }
    h *= 25.;
    
    if (concernWater == 1.0) {
        q = m2*p.xz*0.2;
        float o = 0.;
        float t = iTime * 0.3;
        s = 0.3;
        for (int i = 0 ; i < 5 ; i++) {
            o += s*noise(vec3(q + t, 1.));
            q = m2 * q * 1.98;
            s *= 0.51;
            t *= 1.5;
        }
        o += 4.;

        float d = p.y - max(h, o);
    	return vec2(d, smoothstep(0., abs(noise(p*2.))*1.+0.03, h-o));
    	//return vec2(d, step(0.0, h-o));
    
    } else {
    
        float d = p.y - h;
        return vec2(d, 1.0);
    }
    
    
}


vec2 rayMarching(vec3 ro, vec3 rd, float concernWater) {
	
	float t = 0.01, told = 0., mid, dn;
    vec2 res = map(rd*t + ro, concernWater);
    float d = res.x;
    float m = res.y;
    float sgn = sign(d);
    
    for (int i = 0 ; i < MAX_RAYMARCHING_COUNT ; i++) {
    	if (sign(d) != sgn || d < PRECISION || t > FAR) break;
        
        told = t;
        t += max(d/2.0, t*0.02);
        res = map(rd*t + ro, concernWater);
        d = res.x;
        m = res.y;
    }
    
    if (sign(d) != sgn) {
        res = map(rd*told + ro, 1.0);
    	dn = sign(res.x);
        vec2 iv = vec2(told, t);
        
        for (int j = 0 ; j < 8 ; j++) {
        	mid = dot(iv, vec2(.5));
            res = map(rd*mid + ro, concernWater);
            d = res.x;
        	m = res.y;
            if (abs(d) < PRECISION) break;
            iv = mix(vec2(iv.x, mid), vec2(mid, iv.y),step(0.0, d*dn));
        }
        t = mid;
    }
    
    return vec2(min(t, FAR), res.y);
    
}

// Tetrahedral normal, courtesy of IQ.
vec3 calcuNormal(in vec3 p, float concernWater)
{  
    vec2 e = vec2(-1., 1.)*0.03;   
	return normalize(e.yxx*map(p + e.yxx, concernWater).x + e.xxy*map(p + e.xxy, concernWater).x + 
					 e.xyx*map(p + e.xyx, concernWater).x + e.yyy*map(p + e.yyy, concernWater).x );   
}

float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax, float concernWater)
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<32; i++ )
    {
		float h = map( ro + rd*t , concernWater).x;
        res = min( res, 0.4*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<PRECISION || t>tmax ) break;
    }
    return clamp( res, 0.2, 1.0 );
}

float calcAO( in vec3 pos, in vec3 nor )
{
	float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float hr = 0.01 + 0.12*float(i)/4.0;
        vec3 aopos =  nor * hr + pos;
        float dd = map( aopos , 1.0).x;
        occ += -(dd-hr)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 );    
}

vec3 cloud(in vec3 bgCol, in vec3 ro, in vec3 rd, float spd) {
    float t = iTime*spd;
    vec2 sc = ro.xz + rd.xz*(120000.)/rd.y;
    vec2 p = sc * 0.000007;
    float f = 0.;
    float s = 1.;
    float sum = 0.;
    for (int i = 0 ; i < 6 ; i++) {
    	p += t;
        t *= 1.5;
        f += s*abs(noise(vec3(p, 0.)));
        p = m2 * p * 2.01;
        sum += s;
        s *= 0.6;
    }
    vec3 col = mix(bgCol, vec3(1.), smoothstep(0.0, 1.0, pow(f/sum, 0.5)) * pow(max(rd.y, 0.), 0.5));
    //col = vec3(f/sum);
    return col;
}

vec3 sun(vec3 lightPos, in vec3 ro, in vec3 rd) {
	float sunStength = clamp(dot(rd, normalize(lightPos)), 0., 1.);
    vec3 col = 0.2*vec3(1.0, 0.5, 0.4) * pow(sunStength, 256.0);
    col += 1.8*vec3(1.0, 0.5, 0.4) * pow(sunStength, 512.0);
    //col += 0.4*vec3(1.0, 0.7, 0.4) * pow(sunStength, 1024.0);
    return col;
}
/*
vec3 bgCol1 = vec3(1.0);
vec3 bgCol2 = vec3(0.9137, 0.9176, 0.898);
vec3 bgCol3 = vec3(0.9137, 0.9176, 0.898);
*/
//vec3 mountainCol = vec3(0.6, 0.49, 0.313)*0.5;
vec3 bgCol1 = vec3(0.98, 0.95, 0.93);
vec3 bgCol2 = vec3(0.125, 0.550, 0.60);
vec3 bgCol3 = vec3(0.0, 0.40, 0.41);

vec3 sky(in vec3 ro, in vec3 rd) {
	
    vec3 skyCol = mix(bgCol2, bgCol3, smoothstep(0., 0.3, rd.y));
    skyCol = mix(bgCol1, skyCol, smoothstep(-0.3, 0.12, rd.y));
    skyCol = cloud(skyCol, ro, rd, .3);
    
    skyCol += sun(vec3(-0.7, 0.3, 1.0), ro, rd);
    
    return skyCol;
}


vec3 render(vec3 ro, vec3 rd) {
    vec2 res = rayMarching(ro, rd, 1.0);
    float t = res.x;
    float m = res.y;
    vec3 col = vec3(0.);
    vec3 shdCol = vec3(0.1294, 0.1882, 0.207);
    
    vec3 bgCol = sky(ro, rd);
    
    if (t < FAR) {
        vec3 sp = ro + rd*t;
    	vec3 nor = calcuNormal(sp, 1.0); 
        
        vec3 lp =vec3(10.0*sin(time/1.), 7.0, 10.0*cos(time/1.)+time*3.-4.);
        vec3 ld = normalize(lp - sp);
        ld = vec3(-1.0, 0.8, -1.0);
        lp = normalize(ld + sp);   

        //float shd = softShadow(sp, ld, 0.5, FAR, 3.);
        float shd = calcSoftshadow( sp, ld, 0.5, FAR, 1.0);

        float occ = calcAO( sp, nor );

        vec3 hal = normalize( ld - rd );
        float amb = clamp( 0.3+ 0.7*nor.y, 0.0, 1.0 );
        float dif = max( dot( ld, nor ), 0.0); // Diffuse term.
        //dif += 0.5*max( dot( ld*vec3(-1., 1., -1.), nor ), 0.0); // Diffuse term.
        float speWater = pow( clamp( dot( nor, hal ), 0.0, 1.0 ), 128.0)*dif ; // Specular term.
        float speMoun = pow( clamp( dot( nor, hal ), 0.0, 1.0 ), 32.0)*dif ; // Specular term.
        float bac = clamp( dot( nor, normalize(vec3(-lp.x,0.0,-lp.z))), 0.0, 1.0 )*clamp( 1.0-sp.y,0.0,1.0);
        float fre = clamp(1.0 + dot(rd, nor), 0.0, 1.0); // Fresnel reflection term.

        
        if(res.y == 0.0) {
            float fre = clamp(dot(-rd, nor), 0.0, 0.2);
            fre = smoothstep(0., 0.6, pow(fre, 1.6));
            fre = fre * 0.7;
            vec3 reflectCol = sky(sp, reflect(rd, nor));
            
        	col = mix(reflectCol, vec3(0.2, 0.3, 0.4), fre);
            
        	//col = vec3(lin);
            
            vec3 nrd = refract(rd, nor, 1./1.333);
            vec2 nres = rayMarching(sp, nrd, 0.0);
            float nt = nres.x;
            float nm = nres.y;
            vec3 nsp = sp + nrd*nt;
            vec3 nnor = calcuNormal(nsp, 0.0);
            
        	float nshd = calcSoftshadow( sp, ld, 0.5, FAR, 0.0 );
            
            vec3 nhal = normalize( lp - nrd );
            float namb = clamp( 0.3+ 0.7*nnor.y, 0.0, 1.0 );
            float ndif = max( dot( ld, nnor ), 0.0); // Diffuse term.
            float nspeMoun = pow( clamp( dot( nnor, nhal ), 0.0, 1.0 ), 32.0)*ndif ; // Specular term.
            vec3 ncol = tex3D( iChannel0,sp/10., nnor );
            
            vec3 nlin = vec3(1.) * ndif * nshd;
            nlin += 0.5*namb*bgCol2;
            nlin += 0.1*nspeMoun*vec3(1.0,1.0,1.0);

            ncol *= nlin;
            
            col = mix(ncol, col, smoothstep(0.0, 1., (1.-fre*0.7 )*smoothstep(0.0, 1.7, 0.5+nt)));
            
            col += speWater*vec3(1.0,1.0,1.0)*occ;
            col *= smoothstep(0.0, 0.4, shd);
            //col = ncol;
        } else {
			vec3 mountainCol = tex3D( iChannel0, sp/10., nor );
            col = mountainCol;
            col = mix(col, vec3(0.9), smoothstep(0.45, 0.6, dot(nor, normalize(vec3(-0.3, 1.0, 0.0))	)));
            col = mix(col, mountainCol, 1.-smoothstep(0., 1., m	));

            vec3 lin = vec3(1.) * dif * shd;
            lin += 0.5*amb*bgCol2*occ;
            //lin += 0.3*bac*vec3(0.4)*occ;
            lin += 0.1*speMoun*vec3(1.0,1.0,1.0)*occ;

            col *= lin;


            //col = vec3(shd);
            //col = vec3(smoothstep(0.4, 0.52, nor.y));
            //col = vec3(speMoun);

        }
        

    } else {
        col = bgCol;
    }
    float e = clamp(dot(rd, normalize(vec3(-0.7, 0.3, 1.0))), 0.0, 1.0);
    vec3 fogCol = mix(bgCol, vec3(1.0, 1.0, 0.9)*1.2, pow(e, 32.));
    col = mix(col, fogCol, smoothstep(5., FAR, t));
    
    col = mix(col, fog(col, ro, rd, t*1.2), 1.);
    //col = pow(col, vec3(0.9));
    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 p = (-iResolution.xy + 2.0*fragCoord)/iResolution.y;
    
    vec3 ro = vec3(0., 12., time*4.);
    vec3 lookAt = vec3(sin(mouse.x*3.1415926*2.), ro.y, ro.z+cos(mouse.x*3.1415926*2.));
    mat3 viewMat = setCamera(ro, lookAt);
    vec3 rd = viewMat * normalize(vec3(p, 1.5));
    
    vec3 col = render( ro, rd );
    

    // Output to screen
    fragColor = vec4(col,1.0);
}