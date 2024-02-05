#iChannel0"file://valley.1.jpg"
#iChannel1"file://valley.2.jpg"
#iChannel2"file://valley.3.jpg"
#iChannel3"file://valley.4.png"

mat3 rotx(float a){mat3 rot;rot[0]=vec3(1.,0.,0.);rot[1]=vec3(0.,cos(a),-sin(a));rot[2]=vec3(0.,sin(a),cos(a));return rot;}
mat3 roty(float a){mat3 rot;rot[0]=vec3(cos(a),0.,sin(a));rot[1]=vec3(0.,1.,0.);rot[2]=vec3(-sin(a),0.,cos(a));return rot;}
mat3 rotz(float a){mat3 rot;rot[0]=vec3(cos(a),-sin(a),0.);rot[1]=vec3(sin(a),cos(a),0.);rot[2]=vec3(0.,0.,1.);return rot;}

const float waterY=.0;
// light
const vec3 ld=normalize(vec3(0.,2.1,14.));

float hash(vec2 p){
    float h=dot(p,vec2(127.1,311.7));
    return fract(sin(h)*43758.5453123);
}

float noise(in vec2 p){
    vec2 i=floor(p);
    vec2 f=fract(p);
    vec2 u=f*f*(3.-2.*f);
    return-1.+2.*mix(mix(hash(i+vec2(0.,0.)),
    hash(i+vec2(1.,0.)),u.x),
    mix(hash(i+vec2(0.,1.)),
    hash(i+vec2(1.,1.)),u.x),u.y);
}

float water(vec2 uv){
    uv+=noise(uv*.8);
    vec2 wv=1.-abs(sin(uv));
    return(wv.x+wv.y)*.5;
}
float getCurve(in vec3 rp)
{
    float x=sin(rp.z*.1)*2.4+sin(rp.z*.22)*1.;
    x-=sin(rp.z*.02-.0)*21.5;
    return x;
}

float shoreLine(vec3 rp)
{
    return abs(getCurve(rp)+rp.x);
}

float river(vec2 uv)
{
    float s=0.;
    const float levels=4.;
    mat2 r;
    r[0]=vec2(.4,.4);
    r[1]=vec2(-.24,.27);
    for(float i=1.;i<(levels+1.);i=i+1.)
    {
        uv*=r;
        s+=water(uv*i*2.);
    }
    s/=(levels+1.);
    return s;
}

float mapEdge(in vec3 rp)
{
    rp.x+=getCurve(rp);
    float edgeL=-1.;
    float difx=1.-abs(rp.x);
    return difx;
}

float mapBottom(in vec3 rp)
{
    rp.x+=getCurve(rp);
    float bottom=-.6;
    float ax=abs(rp.x);
    bottom+=smoothstep(1.,10.,ax);
    bottom+=(.5+.5*noise(rp.xz*.3))*.5;
    float hill=smoothstep(65.,80.,ax);
    bottom+=hill*15.;
    bottom+=hill*sin(rp.z*.1)*2.;
    bottom+=hill*sin(rp.z*.15)*1.;
    return bottom;
}

float mapHeightHQ(in vec3 rp)
{
    float bottom=mapBottom(rp);
    float limit=smoothstep(45.,80.,abs(rp.x))*3.;
    bottom-=(.4*smoothstep(.2,.5,noise(rp.xz*.13)))*limit;
    bottom+=(.3*smoothstep(.1,.5,noise(rp.xz*.33)))*limit;
    return rp.y-bottom;
}

float mapHeightLQ(in vec3 rp)
{
    return rp.y-mapBottom(rp);
}

vec3 grad(in vec3 rp,float bump)
{
    float hc=mapHeightHQ(rp);
    vec2 off=vec2(1.1,0.);
    float hh=mapHeightHQ(rp+off.xyy);
    float hv=mapHeightHQ(rp+off.yyx);
    
    vec3 h=normalize(vec3(bump,hh-hc,0.));
    vec3 v=normalize(vec3(0.,hv-hc,bump));
    return-normalize(cross(h,v));
}

vec3 seagrad(in vec2 uv,float bump,float t)
{
    uv*=14.;
    float hc=river(uv);
    vec2 off=vec2(3./t,0.);
    float hh=river(uv+off);
    float hv=river(uv+off.yx);
    
    vec3 h=normalize(vec3(bump,hh-hc,0.));
    vec3 v=normalize(vec3(0.,hv-hc,bump));
    return-normalize(cross(h,v));
}

vec3 texgrad(in sampler2D tex,in vec2 uv,float bump)
{
    float hc=dot(texture(tex,uv).rgb,vec3(.33));
    vec2 off=vec2(.002,0.);
    float hh=dot(texture(tex,uv+off).rgb,vec3(.33));
    float hv=dot(texture(tex,uv+off.yx).rgb,vec3(.33));
    
    vec3 h=normalize(vec3(bump,hh-hc,0.));
    vec3 v=normalize(vec3(0.,hv-hc,bump));
    return-normalize(cross(h,v));
}

bool trace(inout vec3 rp,in vec3 rd)
{
    
    bool hit=false;
    vec3 ro=rp;
    vec3 prevp=rp;
    
    // rough height map tracing
    for(int i=0;i<350;++i)
    {
        float dify=mapHeightLQ(rp);
        if(dify<0.)
        {
            hit=true;
            break;
        }
        prevp=rp;
        rp+=rd*max(.15,dify);
        if(dot(ro-rp,ro-rp)>150000.)break;
    }
    
    // detailed height map tracing
    if(hit)
    {
        hit=false;
        for(int i=0;i<25;++i)
        {
            float dify=mapHeightHQ(rp);
            if(dify<0.)
            {
                hit=true;
                break;
            }
            prevp=rp;
            rp+=rd*max(.1,dify*(1.+log2(1.+length(ro-rp)*1.5)));
            if(dot(ro-rp,ro-rp)>150000.)break;
        }
    }
    
    // binary search the surface
    if(hit)
    {
        vec3 insidep=rp;
        vec3 outsidep=prevp;
        float cursor=.5;
        
        for(int i=1;i<20;++i)
        {
            rp=mix(outsidep,insidep,cursor);
            float dist=mapHeightHQ(rp);
            cursor+=pow(.5,float(i+1))*sign(dist);
        }
        
    }
    
    return hit;
    
}

mat3 lookat(vec3 from,vec3 to)
{
    vec3 f=normalize(to-from);
    vec3 _tmpr=normalize(cross(f,vec3(0.,1.,0.)));
    vec3 u=normalize(cross(_tmpr,f));
    vec3 r=normalize(cross(u,f));
    return mat3(r,u,f);
}

vec4 colorSurface(in vec3 rp,in vec3 rd)
{
    vec4 color=vec4(0.);
    
    vec2 uv=rp.xz*.05;
    color=pow(texture(iChannel1,uv),vec4(2.2));
    vec3 g=texgrad(iChannel1,uv,.2);
    color.rgb*=.4+1.7*max(0.,dot(g,ld));
    
    float toEdge=smoothstep(-9.,-2.,mapEdge(rp*1.1)+noise(rp.zx*2.5));
    color*=mix(1.,toEdge,.7);
    vec3 worldg=grad(rp,1.2);
    
    float limit=smoothstep(25.,50.,abs(mapEdge(rp)));
    color+=vec4(.01,.02,.0,1.)*worldg.y*limit;
    color+=vec4(.01,.02,.0,1.)*abs(worldg.x)*limit;
    color.a=1.;
    return color;
}

vec2 flowGrad(in vec3 rp)
{
    vec3 off=vec3(.02,0.,0.);
    
    float dc=mapEdge(rp);
    float dh=mapEdge(rp+off);
    float dv=mapEdge(rp+off.yyx);
    
    return-vec2(dh-dc,dv-dc);
    
}

float getMixValue(float cycle,inout float offset1,inout float offset2)
{
    // mixval 0..1..0 over full cycle
    float mixval=cycle*2.;
    if(mixval>1.)mixval=2.-mixval;
    
    // texture phase 1
    offset1=cycle;
    // texture phase 2, phase 1 offset by .5
    offset2=mod(offset1+.5,1.);
    return mixval;
}

vec4 getTrees(in vec3 rd)
{
    float an=atan(rd.z,rd.x);
    
    vec4 trees=vec4(.0);
    const float layers=5.;
    float alpha=0.;
    
    for(float i=1.;i<=layers;i=i+1.)
    {
        float offset=an*2.+i*.5;
        float colA=texture(iChannel0,vec2(offset,offset)).r;
        
        float yp=(.5+.5*sin(i+an*.5))*.005;
        yp+=(.5+.5*sin(i*2.+an*5.))*.025;
        yp-=.1*cos(an*.1);
        yp+=rd.y;
        
        float layerH=max(0.,.2-(i*.05));
        colA*=smoothstep(layerH,layerH-.1,yp);
        colA=smoothstep(.33,.37,colA);
        float a=an*.05+i*.01;
        mat2 rm;rm[0]=vec2(cos(a),-sin(a));rm[1]=vec2(sin(a),cos(a));
        vec4 texCol=texture(iChannel2,(vec2(offset,yp*.4)*rm)*4.)*colA;
        texCol=smoothstep(-.6,1.,texCol);
        texCol.rgb*=pow((1./layers)*i,1.);
        texCol.rgb*=vec3(.25,.3,.15)*.5;
        
        trees.rgb=texCol.rgb*colA+(1.-colA)*trees.rgb;
        trees.a=clamp(trees.a+colA,0.,1.);
    }
    return trees;
}

void mainImage(out vec4 fragColor,in vec2 fragCoord)
{
    fragColor=vec4(0.);
    vec2 uv=((fragCoord.xy/iResolution.xy)-vec2(.5))*vec2(1.,iResolution.y/iResolution.x);
    vec2 im=1.5*((iMouse.xy/iResolution.xy)-vec2(.5));
    
    vec3 rp=vec3(-.0,17.5,-10.);
    if(iMouse.z<0.||iMouse.xy==vec2(0.))
    {
        im.xy=vec2(.11,.551);
        rp.z+=sin(iTime*.2);
        rp.y+=sin(iTime*.5);
        rp.x+=cos(iTime*.5);
    }
    
    // camera
    vec3 rd=normalize(vec3(uv,.4));
    vec3 _rp=rp;
    rp=roty(im.x*5.)*rp;
    rp.y=(rotx(im.y*1.5)*_rp).y;
    
    vec3 ro=rp;
    rd=lookat(rp,vec3(.0,4.,5.))*rd;
    
    // ground
    bool hit=trace(rp,rd);
    _rp=rp;
    
    if(hit)
    {
        fragColor=colorSurface(rp,rd);
    }
    
    // water
    if(rp.y<waterY&&mapEdge(rp)>-30.)
    {
        vec3 n=vec3(0.,1.,0.);
        float t=(waterY-dot(ro,n))/dot(n,rd);
        if(t>0.)
        {
            vec3 p=ro+rd*t;
            
            float T=4.;
            
            // texture offsets for advection
            float cycle=mod(iTime,T)/T;
            float o1,o2=0.;
            float mv=getMixValue(cycle,o1,o2);
            float dist=smoothstep(6.,1.,shoreLine(p));
            
            // flow vec
            vec2 sideFlow=(flowGrad(p))*3.;
            float h=mapHeightLQ(p);
            
            vec2 flowFwd=flowGrad(vec3(shoreLine(vec3(0.,0.,rp.z)),0.,rp.z)).yx*3.;
            float mixval=dist;//smoothstep(-5., 0., dist);
            vec2 flow=mix(sideFlow,flowFwd,mixval);
            
            // normal
            float speed=50.;
            vec2 scale=vec2(.35,.4);
            float bmp=.1;
            vec3 g1=seagrad(scale*p.xz+flow*o1*speed,bmp,t);
            vec3 g2=seagrad(scale*p.xz+flow*o2*speed,bmp,t);
            
            vec3 g3=seagrad(scale*p.xz+vec2(.1,.2)+flow*o1*speed*.4,bmp,t);
            vec3 g4=seagrad(scale*p.xz+vec2(.3,.2)+flow*o2*speed*.4,bmp,t);
            
            vec3 gm=mix(g2,g1,mv);
            gm+=mix(g4,g3,mv);
            gm=normalize(gm);
            
            // diffuse
            vec4 blue=vec4(0.,69.,129.,0.)/255.;
            float wd=dot(gm,ld);
            wd=max(0.,wd);
            float wrp=.5;
            wd=(wd+wrp)/(1.+wrp);
            vec4 difcol=blue;
            
            // spec
            vec3 H=normalize(-rd+ld);
            float specd=dot(H,gm);
            specd=max(0.,specd);
            float sd=pow(max(0.,specd),58.)*1.;
            
            // fres
            float fres=1.-max(0.,dot(-rd,gm));
            vec4 lightblue=vec4(151./255.,203./255.,220./255.,0.);
            vec4 orange=vec4(.7,.3,.1,0.);
            fres=pow(fres,2.)*1.;
            
            // combined
            vec4 frescol=fres*orange;
            vec4 surfcol=frescol*.4+difcol*wd*.2;
            
            vec3 refr=normalize(refract(rd,gm,.7));
            
            bool hit=trace(p,refr);
            vec4 bottomColor=colorSurface(p,refr);
            float mx=1.-smoothstep(-.1,2.,-refr.y);
            fragColor=mix(bottomColor,surfcol,mx);
            fragColor+=vec4(sd);
            
            float foam=smoothstep(.1,-.5,h+noise(rp.xz*.15)*.2);//smoothstep(0.5, -1.4, mixval);
            fragColor+=foam*texture(iChannel2,.5*p.xz+sideFlow*o1*speed).rrrr;
            fragColor+=foam*texture(iChannel2,.5*p.xz+sideFlow*o2*speed).rrrr;
            
            fragColor.a=1.;
        }
    }
    else if(hit)
    {
        // grass
        float toEdge=smoothstep(10.,7.,shoreLine(_rp)+noise(_rp.xz)*1.5);
        toEdge+=smoothstep(17.,87.,shoreLine(_rp));
        float H=.1;
        vec3 rstp=rd/-rd.y;
        rp-=rstp*H;
        float STP=.002;
        
        bool hitGrass=false;
        float i=0.;
        for(i=0.;i<H;i=i+STP)
        {
            vec4 tcl=texture(iChannel3,rp.xz*.6);
            float D=.4-i;
            D+=toEdge;
            D+=abs(noise(rp.xz*.015))*.5;
            if(D<tcl.g)
            {
                
                hitGrass=true;
                break;
            }
            rp+=rstp*STP;
        }
        
        if(hitGrass)
        {
            vec4 grassLow=vec4(.08,.05,0.,1.);
            vec4 grassHi=vec4(.1,.1,0.,1.);
            float depth=clamp(pow(1.-(i/H),4.),0.,1.);
            vec4 grassCol=mix(grassLow,grassHi,depth);
            grassCol+=texture(iChannel1,rp.xz*.05)*.25*mix(1.,depth,.4);
            grassCol*=texture(iChannel1,rp.xz*.01).g;
            fragColor=grassCol;
        }
        
        fragColor.a=1.;
        
        vec3 g=grad(_rp,1.2);
        float d=max(0.,dot(g,normalize(vec3(0.,1.,1.))))*2.;
        float wrap=.4;
        d=(d+wrap)/(1.+wrap);
        fragColor.rgb*=d;
        
    }
    
    vec3 skyLow=vec3(.6,.7,.8);
    vec3 skyHi=vec3(.6,.3,.5);
    vec3 skyRed=vec3(6.,.3,0.);
    vec3 sky=mix(skyLow,skyHi,rd.y);
    vec3 skyYellow=vec3(1.,1.,.5);
    
    float a=sin((atan(rd.z,rd.x)+3.14159265)*.5);
    vec2 sun=vec2(a,rd.y*1.5);
    sky=mix(skyRed,sky,smoothstep(0.,1.5,length(sun)));
    sky=mix(skyYellow,sky,smoothstep(0.,.7,length(sun)));
    
    if(length(_rp)>400.)fragColor=vec4(.0);
    
    fragColor.rgb*=.2+.8*smoothstep(-22.,0.,_rp.x+sin(_rp.z*.5));
    fragColor.rgb*=1.+.7*smoothstep(4.,50.,_rp.x+sin(_rp.z*.2));
    
    fragColor=mix(getTrees(rd),fragColor,fragColor.a);
    fragColor.rgb=mix(fragColor.rgb,sky,1.-fragColor.a);
    fragColor.rgb=pow(fragColor.rgb,vec3(1./2.2));
    
}