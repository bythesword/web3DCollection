// Processed by 'GLSL Shader Shrinker' (2,112 to 1,707 characters)
// (https://github.com/deanthecoder/GLSLShaderShrinker)

mat2 rot(float a) {
	float c = cos(a),
	      s = sin(a);
	return mat2(c, s, -s, c);
}

vec3 getRayDir(vec3 ro, vec3 lookAt, vec2 uv) {
	vec3 forward = normalize(lookAt - ro),
	     right = normalize(cross(vec3(0, 1, 0), forward));
	return normalize(forward + right * uv.x + cross(forward, right) * uv.y);
}

float sdWaves(vec3 p) {
	float t,
	      h = 0.,
	      a = 1.,
	      f = 1.1;
	mat2 r = rot(3.7692);
	t = iTime / -3.;
	for (float i = 0.; i < 6.; i++) {
		p.xz *= r;
		h += 1. - abs(sin(f * ((p.x + sin(p.z * a)) + t))) * a;
		a /= 1.97;
		f *= 2.02;
		t *= -.95;
	}

	return p.y - h / 7.;
}

vec3 calcNormal(vec3 p) {
	const vec2 e = vec2(1, -1) * .0005;
	return normalize(e.xyy * sdWaves(p + e.xyy) + e.yyx * sdWaves(p + e.yyx) + e.yxy * sdWaves(p + e.yxy) + e.xxx * sdWaves(p + e.xxx));
}

float calcShadow(vec3 p, vec3 lightPos, float sharpness) {
	vec3 rd = normalize(lightPos - p);
	float h,
	      minH = 1.,
	      d = .1;
	for (int i = 0; i < 16; i++) {
		h = sdWaves(p + rd * d);
		minH = abs(h / d);
		if (minH < .01) return 0.;
		d += h;
	}

	return minH * sharpness;
}

float glow = 0.;
vec3 vignette(vec3 col, vec2 fragCoord) {
	vec2 q = fragCoord.xy / iResolution.xy;
	col *= .5 + .5 * pow(16. * q.x * q.y * (1. - q.x) * (1. - q.y), .4);
	return col;
}

void mainImage(out vec4 fragColor, vec2 fragCoord) {
	vec2 uv = (fragCoord - .5 * iResolution.xy) / iResolution.y;
	vec3 col;
	{
		vec3 p,
		     rd = getRayDir(vec3(0, 1.5, -10), vec3(0), uv);
		bool hit = false;
		float d = .01;
		for (float steps = 0.; steps < 128.; steps++) {
			p = vec3(0, 1.5, -10) + rd * d;
			float h = sdWaves(p),
			      dd = length(p - vec3(.1, 2, -2));
			glow += .1 / (.1 + dd * dd * 5.);
			if (dd < h) h = dd;
			if (h < .005 * d) {
				hit = true;
				break;
			}

			d += h;
		}

		if (hit) {
			vec3 n = calcNormal(p),
			     lightToPoint = normalize(vec3(4, 20, 10) - p);
			float sha = calcShadow(p, vec3(4, 20, 10), 5.),
			      mainLight = max(0., dot(n, lightToPoint)),
			      fog = 1. - exp(-d * .02);
			col = mix((mainLight * sha + 2.) * vec3(1, .9, .8), vec3(.15, .2, .25), (1. - .98 * max(0., dot(rd, n))));
			col *= vec3(.2, .32, .41);
			col = mix(col, vec3(.15, .2, .25), fog);
		}
		else col = vec3(.15, .2, .25);
	}
	fragColor = vec4(pow(vignette(clamp(col + max(0., glow), 0., 1.), fragCoord), vec3(.4545)), 1);
}