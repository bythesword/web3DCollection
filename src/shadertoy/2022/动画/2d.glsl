
#define R iResolution.xy

float sdBox(vec2 c, vec2 s){
    c = abs(c) - s; return max(c.x,c.y);
}
#define rot(a) mat2(cos(a),-sin(a),sin(a),cos(a))
#define pmod(p,a) mod(p,a) - 0.5*a
// Dave hoskins hash without sine
float r21(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// cyclic noise by nimitz. i have a tutorial on it on shadertoy

float noise(vec3 p_){
    float n = 0.;
    float amp = 1.;
    vec4 p = vec4(p_,11.);
    p.xy *= rot(1.4);
    p.x *= 3.;
    for(float i = 0.; i < 12.; i++){
        p.yz *= rot(.5);
        p.xz *= rot(2.5 + i);
        p.wy *= rot(1.5-i);
        p += cos(p*1. + vec4(3,2,1,1.) )*amp*.5;
        n += dot(sin(p),cos(p))*amp;
    
        amp *= 0.7;
        p *= 1.5;
    }
    
    //n = n * 0.9;
    n = sin(n*2.);
    return n;
}
float noiseGrid(vec3 p_){
    float n = 0.;
    float amp = 1.;
    vec4 p = vec4(p_,11.);
    for(float i = 0.; i <2.; i++){
        p.yz *= rot(.5);
        p.xz *= rot(2.5 + i);
        p.wy *= rot(2.5-i);
        p += cos(p*1. + vec4(3,2,1,1.) )*amp*.5;
        n += dot(sin(p),cos(p))*amp;
    
        amp *= 0.5;
        p *= 1.5;
    }
    
    //n = sin(n*1.);
    return n;
}
vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
// from iq
vec3 hsv2rgbSmooth( in vec3 hsv )
{
    vec3 rgb = clamp( abs(mod(hsv.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0 );

    rgb = rgb*rgb*(3.0-2.0*rgb); // cubic smoothing

    return hsv.z * mix( vec3(1.0), rgb, hsv.y);
}


#define iTime (iTime + 4.*iMouse.x/iResolution.x)
void mainImage( out vec4 C, in vec2 U ){
    vec2 uv = (U - 0.5*R.xy)/max(R.x,R.y);

    vec3 col = vec3(0);
    
    float res =3./max(R.x,R.y);
    
    
    vec2 p = uv;
    p = uv;

    float scroll = iTime*0.1;
    float pan = sin(iTime*0.6 + sin(iTime*0.5))*0.05 - 0.1;
    pan += (iMouse.y/iResolution.y - 0.5)*0.;
    
    p.x += scroll;
    p.y += pan;
    
    vec2 op = p;

    vec2 pid = floor(p/res);
    p = floor(p/res)*res;       
    float dith = texelFetch(iChannel0,ivec2(pid)%8,0).x;
        
    // sky
    {
 
        
        vec3 skyCol = vec3(1.,0.7,0.4)*0.9;
        vec3 skyColb = vec3(0.1,0.4,0.5)*1.2;
        
        //float dith = texture(iChannel0,);
        
        col = mix(skyCol,vec3(1.4,0.7,0.5)*0.7,step(0.7-p.y*15. + 0.5*noise(vec3(p.x,4,4))*0.1,dith));
        col = mix(col,vec3(1.,0.4,0.3)*0.8,step(1.4-p.y*10. + 0.5*noise(vec3(p.x,4,4))*0.1,dith));
        col = rgb2hsv(col);
        
        col = hsv2rgbSmooth(col + vec3(0,-0.1,0));
        //col = mix(col,skyColb,step(1.2-p.y*15. + 0.5*noise(vec3(p.x,4,4))*0.1,dith));
        
        //col = dith*vec3(1);
        //col = mix(skyCol,skyColb,p.y*2. + 0.5*noise(vec3(p.x,4,4))*0.1);
    }
    float sdl = p.y + noise(vec3(p.x,4. + p.x,4))*0. + 0.1;
    if(sdl < 0. + mod(pid.x*0.1 + r21(vec2(pid.x*0.1,40.))*0.,2.)*0.003){
        col = vec3(0.45,0.42,0.5) + 0.;
    }     
    // Clouds
    {
        
        for(float i = 0.; i < 56.; i++){
            vec2 cpos = vec2(
                    - mod(
                        i*12.6
                        + .05*iTime*max(abs(sin(i + sin(i*20.)*4. + 200.)),0.8)*1.05
                        + scroll*0. 
                        ,2.
                    )*2.
                    + 2. + scroll
                ,cos(i*20.)*0.1 + 0.07);
            vec2 lcpos = cpos - p;
            float d = length(lcpos.y) + abs(lcpos.y)*0.9;
            d += smoothstep(0.,4.,abs(lcpos.x));
            d += noise(vec3(lcpos.x*2. + i*120.,4. + i,4. + p.y))*0.05;
            //d -= noise(vec3(lcpos.x*4. + i*120.,4. + i,4. + p.y))*0.01;
            //if(d < 0.00)
                //col = mix(col,col*3.*vec3(0.2,0.15,0.25)*(1. + sin(i*3.)*0.1),0.4 + sin(i*20.)*0.1);
            if(d < 0.00)
                col = mix(col,vec3(1.2,0.7,0.6)*0.8,0.6);
        
        }
    
    }
    
    vec3 rocks = vec3(0);
    float minDrocks = 10e6;
    // props
    
    {
            
        for(float i = 0.; i < 50.; i++){
            vec2 lp = p + vec2(0,pan*0.4);
            lp.x -= 2. + sin(i)*4.;
            lp.x = mod(lp.x,4.) - 2.;
            lp.y += 0.45 + sin(i*200.)*0.05;
            float sdplank = length(lp);
            sdplank = max(sdplank,-lp.y + 0.02);
            
            sdplank -= abs(noiseGrid(vec3(lp*120. + i*20.,15.)))*0.004 + abs(sin(i*15.))*0.02;
            minDrocks = min(minDrocks,sdplank + max(lp.y*1. -0.01,0.));
            if(sdplank < 0.004 ){
                rocks = mix(
                    vec3(0.34,0.35,0.4),
                    vec3(0.1)*2.,
                    1.-step(
                        clamp(dot(lp,normalize(vec2(1,-1)))*44.+0.4,0.,1.),
                        dith
                    )
                );
            }
        }
        
        
        
    
    }
        
    
    // ground
    
    {


            float griters = 40.;
            vec2 lastp = p;
            
            for(float i = 0.; i <= griters; i++){
                
                float sc = pow(i/griters,4.);
                p = lastp;
                p.y += sc*pan*0.4;
                p.x += (1.-sc)*iTime*0.004;
                
                //vec2 offs = vec2(0,pan);
                
                //vec2 lp = p;
                //vec2 lpid = floor(p/res);
                //lp = floor(lp/res)*res;    
                
                
                //p.x += iTime*0.2*0.003;
                //pid = floor(p/res);
                //p = floor(p/res)*res;
                
                float iterscale = smoothstep(1.,0.7,i/griters);
    
                float n = noise(vec3(p.x*(1. + 1.*abs(sin(i*20.))),1. + i * 20.,4. + i))*0.1*sin(i*10.+iTime*float(iMouse.z>0.))*iterscale;
                float lsd = p.y*1. + n + i/griters*0.2 + 0.14;
                float idx = i/griters;
                float grass = mod(pid.x + r21(vec2(i*20. + pid.x,40.))*4.,2.)*0.01*sin(i)*smoothstep(0.4,1.,idx);
                float fade = (1. - idx);
                if(lsd < 0. + grass){
                    vec3 grcol = (vec3(0.4,0.34,0.1) - 0.1*sin(vec3(0.2,0.4,41.3) + i));
                    grcol = mix(grcol,vec3(0.3,0.3,0.5), pow(fade,2.4));
                    float grass = mod(pid.x + r21(vec2(pid.x))*1., 2.)*0.3*(idx);
                    
                    float dn = noise(vec3(p.x*2.,1. + p.y*6. + n*4. + i*20.,4. + i*20.)) + grass;
                    
                    grcol *= 1.
                        - vec3(0.6,0.8,0.6)*0.2*step(dn,0.5) 
                        - vec3(0.6,0.4,0.5)*0.3*step(dn + 0.9 - grass*0.7*(1.-iterscale) + n*20.,0.1)*(sc*0.6 + 0.4)
                        - smoothstep(0.02,0.,minDrocks)*0.4
                        //- step(minDrocks,0.007)*0.4
                        
                        ;
                    
                    col = grcol;
                }
            }
    }
    
    col = mix(col,rocks,float(rocks.x!=0.));
    col = rgb2hsv(col);
    
    
    float t = iTime + 10.;
    float tt = 10.;
    float tid = floor(t/tt);
    float T = mod(t,tt);
    float tenv = clamp(T,0.,1.);
    
    
    float a = -0.4;
    float b = 0.;
    float v = mod(tid,2.) == 0. ? mix(a,b,tenv) : mix(b,a,tenv);
        
    col = hsv2rgbSmooth(col + vec3(0,v,-0.04));
    
    
    
    col = pow(abs(col),vec3(0.8));

    C = vec4(col,1);
}