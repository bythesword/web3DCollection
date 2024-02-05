// 贴图
#iChannel0 "file://images/erha.jpg"

// Wrap方式：Clamp Repeat  Mirror
#iChannel0::WrapMode "Repeat"

// 采样方式：Nearest Linear NearestMipMapNearest
#iChannel0::MinFilter "NearestMipMapNearest"
#iChannel0::MagFilter "Nearest"


// 坐标系缩放
#define PROJECTION_SCALE  1.

// 长方体的中心位置
#define RECT_POS vec3(0.5)
// 长方体的尺寸
#define RECT_SIZE vec3(0.5)
// 长方体的漫反射系数
#define RECT_KD vec3(1)

// 相机视点位
#define CAMERA_POS mat3(cos(iTime),0,sin(iTime),0,1,0,-sin(iTime),0,cos(iTime))*(vec3(1, 1.5, 2.5)-RECT_POS)+RECT_POS

// 相机目标点
#define CAMERA_TARGET vec3(0.5)
// 上方向
#define CAMERA_UP vec3(0, 1, 0)

// 光线推进的起始距离 
#define RAYMARCH_NEAR 0.1
// 光线推进的最远距离
#define RAYMARCH_FAR 128.
// 光线推进次数
#define RAYMARCH_TIME 40
// 当推进后的点位距离物体表面小于RAYMARCH_PRECISION时，默认此点为物体表面的点
#define RAYMARCH_PRECISION 0.001

// 相邻点的抗锯齿的行列数
#define AA 3

// 坐标系
vec2 Coord(in vec2 coord) {
  return PROJECTION_SCALE * 2. * (coord - 0.5 * iResolution.xy) / min(iResolution.x, iResolution.y);
}

// 长方体的的SDF模型
float SDFRect(vec3 coord) {
  vec3 d = abs(coord - RECT_POS) - RECT_SIZE;
  return length(max(d, 0.)) + min(max(d.x, max(d.y, d.z)), 0.);
}


// 视图旋转矩阵
mat3 RotateMatrix() {
  //基向量c，视线
  vec3 c = normalize(CAMERA_POS - CAMERA_TARGET);
  //基向量a，视线和上方向的垂线
  vec3 a = cross(CAMERA_UP, c);
  //基向量b，修正上方向
  vec3 b = cross(c, a);
  //正交旋转矩阵
  return mat3(a, b, c);
}

// 计算长方体的法线
vec3 SDFNormal(in vec3 p) {
  const float h = 0.0001;
  const vec2 k = vec2(1, -1);
  return normalize(k.xyy * SDFRect(p + k.xyy * h) +
    k.yyx * SDFRect(p + k.yyx * h) +
    k.yxy * SDFRect(p + k.yxy * h) +
    k.xxx * SDFRect(p + k.xxx * h));
}

// 获取纹理
//p.xyz 是坐标，为什么可以用作UV一样，还是没有明白;应该是自动归一化了
// vec3 getTexture(vec3 p,vec3 n){
//   vec3 absN=abs(n);
//   vec3 colorZ = texture(iChannel0, p.xy*2.).rgb*absN.z;
//   vec3 colorY = texture(iChannel0, p.xz).rgb*absN.y;
//   vec3 colorX = texture(iChannel0, p.zy).rgb*absN.x;
//   return colorZ+colorY+colorX;
// }
vec3 getTexture(vec3 p,vec3 n){
  return texture(iChannel0, p.xy).rgb;
  // return p.xyz;
}

// 光线推进
vec3 RayMarch(vec2 coord) {
  float d = RAYMARCH_NEAR;
  // 从相机视点到当前片元的射线
  vec3 rd = normalize(RotateMatrix() * vec3(coord, -1));
  // 片元颜色
  vec3 color = vec3(0);
  for(int i = 0; i < RAYMARCH_TIME && d < RAYMARCH_FAR; i++) {
    // 光线推进后的点位
    vec3 p = CAMERA_POS + d * rd;
    // 法线绝对值
    vec3 n=SDFNormal(p);
    // 光线推进后的点位到长方体的有向距离
    float curD = SDFRect(p);
    // 若有向距离小于一定的精度，默认此点在长方体表面
    if(curD < RAYMARCH_PRECISION) {
      
      color=getTexture(p,n);
      
      break;
    }
    // 距离累加
    d += curD;
  }
  return color;
}

// 抗锯齿 Anti-Aliasing
vec3 RayMarch_anti(vec2 fragCoord) {
  // 初始颜色
  vec3 color = vec3(0);
  // 行列的一半
  float aa2 = float(AA / 2);
  // 逐行列遍历
  for(int y = 0; y < AA; y++) {
    for(int x = 0; x < AA; x++) {
      // 基于像素的偏移距离
      vec2 offset = vec2(float(x), float(y)) / float(AA) - aa2;
      // 坐标位
      vec2 coord = Coord(fragCoord + offset);
      // 累加周围片元的颜色
      color += RayMarch(coord);
    }
  }
  // 返回周围颜色的均值
  return color / float(AA * AA);
}

/* 绘图函数，画布中的每个片元都会执行一次，执行方式是并行的。
fragColor 输出参数，用于定义当前片元的颜色。
fragCoord 输入参数，当前片元的位置，原点在画布左下角，右侧边界为画布的像素宽，顶部边界为画布的像素高
*/
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  // 光线推进
  vec3 color = RayMarch_anti(fragCoord);
  // 最终颜色
  fragColor = vec4(color, 1);
}
