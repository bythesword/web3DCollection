//created by manoloide 2018
//@manoloidee


#define TAU 6.28318530718

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    //vec2 uv = fragCoord/iResolution.xy;

    float tt = iTime*0.11;
    float dt = mod(-tt, 0.5);
    
    vec2 st = fragCoord/iResolution.xy;
    st -= 0.5;
    float asp = iResolution.x/iResolution.y;
    st.x *= asp;
    
    vec3 color = mix(vec3(0.965,0.837,0.692), vec3(0.795,0.577,0.270), rand(st+iTime*0.0000000001));
    color *= 0.95+min(1.0, pow(rand(iTime*0.00001+st*10.0+20.0), 200.0)*20.0);
    color *= 0.5+color*(1.1+cos(iTime*.1)*0.1);
    
    float mov = pow(cos(tt*TAU), 1.0);
    float m1 = st.y+cos(+cos(st.x*3.0+iTime*0.4)+cos(st.x*2.0+iTime*.013))*0.2+1.0;
    float sw = smoothstep(0.4, 0.1, m1-1.0-mov*0.02)*0.8;
    float lm = smoothstep(0.5, 0.1, m1-mov*1.1);
    float wave = pow(lm*1.2, 2.4)+pow(lm*1.2, 1.4)*0.001;
    float wo2 = 120.0;
    float light = wave*cos((((st.y+wave*0.1+m1)+lm*0.1)+cos(iTime)*0.2-mov+iTime*0.01)*400.0+cos((st.x)*wo2))*(cos(((st.y+wave*0.0001+m1)-mov*2.0)*20.0)*1.0);
    
    color = mix(color*(1.-sw*0.1), vec3(0.0), sw*pow(dt, 0.8));
    color = mix(color, vec3(1.000,0.714,0.832), wave*1.0);
    color = mix(color, vec3(0.109,0.910,0.893), min(wave, 0.98));
    float li = light*pow(1.-st.y*0.02, 0.02)*0.5+pow(light, 5.0)*0.05;
    color = mix(color, vec3(1.0), li-(0.5-st.y)*0.5);
    
    float dd = distance(st, vec2(0.0))/asp;
    color = mix(color, vec3(0.0), pow(dd, 2.2));
    color += color*(0.24+cos(iTime*0.2)*0.1);

    // Output to screen
    fragColor = vec4(color,1.0);
}