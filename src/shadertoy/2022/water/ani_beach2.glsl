// 3D simplex noise adapted from https://www.shadertoy.com/view/Ws23RD
// * Removed gradient normalization

vec4 mod289(vec4 x)
{
    return x - floor(x / 289.0) * 289.0;
}

vec4 permute(vec4 x)
{
    return mod289((x * 34.0 + 1.0) * x);
}

vec4 snoise(vec3 v)
{
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);

    // First corner
    vec3 i  = floor(v + dot(v, vec3(C.y)));
    vec3 x0 = v   - i + dot(i, vec3(C.x));

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + C.x;
    vec3 x2 = x0 - i2 + C.y;
    vec3 x3 = x0 - 0.5;

    // Permutations
    vec4 p =
      permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0))
                            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
                            + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    vec4 j = p - 49.0 * floor(p / 49.0);  // mod(p,7*7)

    vec4 x_ = floor(j / 7.0);
    vec4 y_ = floor(j - 7.0 * x_); 

    vec4 x = (x_ * 2.0 + 0.5) / 7.0 - 1.0;
    vec4 y = (y_ * 2.0 + 0.5) / 7.0 - 1.0;

    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 g0 = vec3(a0.xy, h.x);
    vec3 g1 = vec3(a0.zw, h.y);
    vec3 g2 = vec3(a1.xy, h.z);
    vec3 g3 = vec3(a1.zw, h.w);

    // Compute noise and gradient at P
    vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    vec4 m2 = m * m;
    vec4 m3 = m2 * m;
    vec4 m4 = m2 * m2;
    vec3 grad =
      -6.0 * m3.x * x0 * dot(x0, g0) + m4.x * g0 +
      -6.0 * m3.y * x1 * dot(x1, g1) + m4.y * g1 +
      -6.0 * m3.z * x2 * dot(x2, g2) + m4.z * g2 +
      -6.0 * m3.w * x3 * dot(x3, g3) + m4.w * g3;
    vec4 px = vec4(dot(x0, g0), dot(x1, g1), dot(x2, g2), dot(x3, g3));
    return 42.0 * vec4(grad, dot(m4, px));
}

// Jaybird's Simple Water Caustic Pattern
// https://www.shadertoy.com/view/3d3yRj
// 
// Adapted from https://www.shadertoy.com/view/Ws23RD

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    
	vec2 p = (-iResolution.xy + 2.0*fragCoord) / iResolution.y;
    
    // Uncomment, and then set any settings value to either prevX/Y to preview it with the mouse
    float prevX  = (iMouse.x/iResolution.x)*4.0-2.0;
    float prevY  = (iMouse.y/iResolution.y)*4.0-2.0;
    
    // Settings
    float invertY    = -1.0; // 1.0 = invert y axis (application), -1.0= don't invert (shadertoy)
    float yaw        = -0.03;  // Rotate camera on z axis (like saying no with your head).
    float pitch      = 0.6;  // Rotate camera like saying yes with your head
    float roll       = 0.0;  // Rotate camera like putting your head to your shoulder
    float height     = 2.0;  // Height of the room, BUT also changes the pitch downwards.
    float fov        = 1.0;  // Basically zoom, comes with perspective distortion too. 
    float scale      = 8.0; // Size of the rays (also changes the speed)
    float speed      = 0.16; // How quickly the rays dance
    float brightness = 1.7;  // Smaller = brighter, more intense rays
    float contrast   = 2.0;  // Difference between ray and darkness. Smaller = more grey.
    float multiply   = 0.2;  // Alpha/transparency and colour intensity of final result
    vec3  rayColour  = vec3(1.0,0.964,0.690); // rgb colour of rays
    
    // Move the camera
    float offsetX    = -prevX*15.0;
    float offsetY    = prevY*15.0;
    
    // Camera matrix complicated maths stuff
    vec3 ww = normalize(invertY*vec3(yaw, height, pitch));
    vec3 uu = normalize(cross(ww, vec3(roll, 1.0, 0.0)));
    vec3 vv = normalize(cross(uu,ww));
	vec3 rd = p.x*uu + p.y*vv + fov*ww;	// view ray
    vec3 pos = -ww + rd*(ww.y/rd.y);	// raytrace plane
    pos.y = iTime*speed;				// animate noise slice
    pos *= scale;				        // tiling frequency
    
    // Apply the offsets to camera position
    pos.x += offsetX;
    pos.z += offsetY;
    
    // Generate some noise
    vec4 noise = snoise( pos );
    
    // Offset it and regenerate x2
    pos -= 0.07*noise.xyz;
    noise = snoise( pos );

    pos -= 0.07*noise.xyz;
    noise = snoise( pos );

    // Calculate intensity of this pixel
    float intensity = exp(noise.w*contrast - brightness);
    
    // Generate a lovely warm oceany gradient
    vec4 c = vec4(234.0/255.0-(fragCoord.y/iResolution.y)*0.7,235.0/255.0-(fragCoord.y/iResolution.y)*0.4,166.0/255.0-(fragCoord.y/iResolution.y)*0.1,1.0);
    
    // Generate final rgba of this pixel
	fragColor = c+vec4(rayColour*multiply*intensity, intensity);
}


