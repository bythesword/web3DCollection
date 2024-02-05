#iChannel0 "file://noise.BigColor.png"
vec4 renderSnowLayer(vec2 uv, int layer, float time, float dx) 
{ 
    vec4 col = vec4(0);

    float fi = float(layer)*.25;
    float tileSize = fi;
    dx *= tileSize;
    
    float fallRate = .7 / fi;
    float windRate = .7 / fi;
    
    // establish a grid, try to remove any hint of correlation between layers
    vec2 tileSpace = (uv + hash21(fi) + vec2(windRate, fallRate)*time) * tileSize;
    vec2 tileId = floor(tileSpace);
    vec2 tileUV = fract(tileSpace);
    
    // 4 random numbers that are constant for each snowflake & unique per layer
    vec4 rnd = hash42(tileId + fi*vec2(-3.3, 5.7));
    
    // give the snowflake a pendulum swaying motion as it falls
    float flakeT = 2.*time*(.25 + .75*rnd.x) + rnd.y*tau;
    flakeT = sin(flakeT)*tau/4. + tau/2.;
    vec2 flakePos = .25 + .5*rnd.zw + .15*vec2(sin(flakeT),cos(flakeT));
    
    float flakeRad = .1;
    float d = distance(tileUV, flakePos) - flakeRad;
    
    // bail early if there's nothing to do. 2*dx because fwidth needs to be defined
    // up to dx, so we need an extra dx on top
    if (d > dx+dx) return col;
    
    // put the streetlight just out of view
    vec2 lightPos = vec2(1.1, 1.1);
    float falloff = smoothstep(1.3, .0, sqr(distance(uv, lightPos)));
    float focus = pow(dot(normalize(uv - lightPos), vec2(0,-1)),9.);
    float light = mix(0.02, 1., focus * falloff);

    // use a texture for the snowflake
    // make sure the texture space is constant
    vec2 txuv = tileUV - flakePos;
    // convert to polar
    float txuvr = length(txuv);
    float txuva = atan(txuv.y, txuv.x);
    // give it a hexagonal shape + rotating + unique angle
    txuva = saw(6.*(txuva/tau+time*(rnd.x-.5)))*.5 + rnd.y*tau;
    // convert back to cartesian & add an offset
    txuv = txuvr * vec2(cos(txuva),sin(txuva)) + rnd.zw*1.;
    // keep the tex lookup independent of texture res
    float tres = float( textureSize(iChannel0, 0).x );
    // falloff radially; aim for some hexagonal edges, 
    // blurry foreground flakes, at least a little subtlety
    float tx = smoothstep(0., sqr(txuvr*30.), texture(iChannel0, txuv*(50./tres)).x);
    
    // accumulate
    col += smoothstep(+dx, -dx, d) * vec4(light) * tx;
    
    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.yy;
    float dx = 1.*length(fwidth(uv));
    float time = iTime + iMouse.x/100.;

    // time of day
    float t = cos(tau*time/120.)*.5+.5;
    
    // cool night, warm day
    fragColor = vec4(.1+.7*t, .1+.65*t, .15+.55*t, 1.);
    
    // add snow
    for (int i = 1; i < 64; i++) {
        fragColor += renderSnowLayer(uv, i, time, dx);
    }
    
    // fade in
    if (iTime < 10.) fragColor *= smoothstep(1., 10., iTime);
}
