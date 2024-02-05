#define MAX_STEPS 100
#define MAX_DIST 500.0
#define MIN_DIST 0.01
#define pi acos(-1.0)
#define sat(t) clamp(t, 0.0, 1.0)

// 2D rotation
vec2 rot(vec2 p, float a) {
	return p*mat2(cos(a), sin(a), -sin(a), cos(a));
}

// random [0,1]
float rand(vec2 p) {
	return fract(sin(dot(p, vec2(12.543,514.123)))*4732.12);
}

// value noise
float noise(vec2 p) {
	vec2 f = smoothstep(0.0, 1.0, fract(p));
	vec2 i = floor(p);
	float a = rand(i);
	float b = rand(i+vec2(1.0,0.0));
	float c = rand(i+vec2(0.0,1.0));
	float d = rand(i+vec2(1.0,1.0));
	return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
	
}

// fractal noise
float fbm(vec2 p) {
    float a = 0.5;
    float r = 0.0;
    for (int i = 0; i < 8; i++) {
        r += a*noise(p);
        a *= 0.5;
        p *= 2.0;
    }
    return r;
}

// sky SDF
float sky(vec3 p) {
    vec2 puv = p.xz;
    // move clouds
    puv += vec2(-2, 4)*iTime;
    // plane with distortion
	return 0.4*(-p.y+25.0+noise(puv/20.0)*1.5*fbm(puv/7.0));
}

// mountains SDF
float mountains(vec3 p) {
    // add slope so it forms a valley
    float addSlope = -clamp(abs(p.x/20.0), 0.0, 7.0);
    // increase intensity of distortion as it moves away from the river
    float rockDist = clamp(2.0*abs(p.x/3.0), 0.0, 30.0);
    // rock formations
    float rocks = fbm(vec2(0, iTime/5.0)+p.xz/15.0);
    // plane with distortion
    return p.y-rockDist*rocks+addSlope+10.0;
}

// river SDF
float river(vec3 p) {
    // underwater rocks that disturb the flow of the river
    // with a modification by pyBlob that adds a pressure hole after each rock
    float rocks = pow(noise(p.xz/6.0+vec2(0, iTime/1.5)),4.0)
                  - pow(noise((p.xz-vec2(0,1.5))/6.0+vec2(0, iTime/1.5)),4.0);
    // surface waves
    float waves = 0.75*fbm(noise(p.xz/4.0)+p.xz/2.0-vec2(0,iTime/1.75-pow(p.x/7.0,2.0)))
                  + 0.75*fbm(noise(p.xz/2.0)+p.xz/2.0-vec2(0, iTime*1.5));
    // Plane with distortion
    return p.y+4.0-rocks+waves;
}

// scene
float dist(vec3 p) {
	return min(river(p), min(mountains(p), sky(p)));
}

// classic ray marcher that returns both the distance and the number of steps
vec2 rayMarch(vec3 cameraOrigin, vec3 rayDir) {
	float minDist = 0.0;
	int steps = 0;
	while (steps < MAX_STEPS) {
		vec3 point = cameraOrigin + rayDir * minDist;
		float d = dist(point);
		minDist += d;
		if(d < MIN_DIST || minDist > MAX_DIST) {
			break;
		}
		steps++;
	}
	return vec2(minDist, steps);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
	vec2 uv = fragCoord.xy/(iResolution.y);
	uv -= iResolution.xy/iResolution.y/2.0;
    // camera setup
	vec3 cameraOrigin = vec3(0, noise(vec2(0, iTime/4.0))-1.5, 0);
	vec3 ray = normalize(vec3(uv.x, uv.y, 0.4));
    // camera sway
    ray.yz = rot(ray.yz, mix(-0.5 , 0.5, 0.25*noise(vec2(0, 0.5+iTime/4.0))+noise(vec2(3.0-iTime/9.0))));
    ray.xz = rot(ray.xz, mix(-1.0 , 1.0, noise(vec2(5.0+iTime/10.0))));
    // ray march
    vec2 rm = rayMarch(cameraOrigin, ray);
    // color is based on the number of steps and distance
	vec4 col = pow(vec4(rm.y/100.0),vec4(3.0))+pow(rm.x/MAX_DIST,2.5);
    // gamma correction
	fragColor = pow(col, vec4(1.0/2.2));
}