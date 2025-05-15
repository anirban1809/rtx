#version 330 core

out vec4 FragColor;
in vec2 fragCoord;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;

#define PI 3.1415926535
#define SHADOWS

float FLOAT_MAX = 10e+10;
float FLOAT_MIN = -10e+10;

vec4 ambientLight = vec4(1, 1, 1, 1);
float ambientStrength = 0.1;
const int R = 3;
const float delta = 10e-5;
float shadowFactor = 0.1;
const int N = 6;
bool transform = true;
bool deg = false;
float fov = 0.8;
const int totalRays = int(pow(2.0, float(R)));

struct Material {
  vec4 color;
  float kd;
  float ks;
  float kr;
  float kt;
  float n;
};

struct Sphere {
  float radius;
  vec3 center;
  Material mat;
};

struct Plane {
  vec3 center;
  vec3 size;
  vec3 normal;
  Material mat;
};

struct Light {
  vec3 dir;
  float mag;
  vec4 color;
  vec3 ray;
};

struct Ray {
  vec3 dir;
  vec3 origin;
  float factor;
  float n;
};

struct Hit {
  float d;
  vec3 point;
  vec3 normal;
};

Ray reflectionRays[8];
Ray refractionRays[8];
Light light;
Plane ground;
Sphere spheres[N];

mat3 Rotation(vec3 euler, bool deg) {
  if (deg)
    euler *= PI / 180.0;
  float cx = cos(euler.x), sx = sin(euler.x);
  float cy = cos(euler.y), sy = sin(euler.y);
  float cz = cos(euler.z), sz = sin(euler.z);
  mat3 Rx = mat3(1, 0, 0, 0, cx, -sx, 0, sx, cx);
  mat3 Ry = mat3(cy, 0, sy, 0, 1, 0, -sy, 0, cy);
  mat3 Rz = mat3(cz, -sz, 0, sz, cz, 0, 0, 0, 1);
  return Rz * Ry * Rx;
}

vec4 CastRays(int iter) {
  return vec4(0.8, 0.2, 0.4, 1.0); // placeholder result
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec3 cameraPos = vec3(0, 0, -fov);
  Ray ray;
  ray.origin = cameraPos;

  vec3 camOffset = vec3(0, 2, 5);
  float camAngle = iTime * 0.6;
  float camRadius = 6.0;

  light.dir = vec3(sin(iTime * 0.7), -1, cos(iTime * 0.7));
  light.mag = 1.0;
  light.color = vec4(1, 1, 1, 1);

  ground.center = vec3(camOffset.x, 0, camOffset.z);
  ground.size = vec3(5, 0, 5);
  ground.normal = vec3(0, 1, 0);
  ground.mat = Material(vec4(0.3, 0.8, 0.2, 1.0), 1.0, 16.0, 0.2, 0.0, 1.0);

  spheres[0].radius = 0.8;
  spheres[0].center = vec3(0.0, 0.8, 2);
  spheres[0].mat = Material(vec4(0, 0, 0, 0.0), 1.0, 32.0, 0.1, 1.0, 1.25);
  spheres[1].radius = 1.1;
  spheres[1].center = vec3(1.0, 1.1, 6);
  spheres[1].mat = Material(vec4(0.3, 0.3, 1.0, 1.0), 1.0, 16.0, 0.1, 0.0, 2.0);
  spheres[2].radius = 0.5;
  spheres[2].center = vec3(-2.0, 0.5, 3.0);
  spheres[2].mat = Material(vec4(0.8, 0.8, 0.1, 1.0), 1.0, 32.0, 1.0, 0.0, 2.0);
  spheres[3].radius = 0.5;
  spheres[3].center = vec3(1.5, 0.8, 3);
  spheres[3].mat =
      Material(vec4(0.0, 1.0, 1.0, 1.0), 1.0, 0.0001, 0.0, 0.0, 2.0);
  spheres[4].radius = 1.0;
  spheres[4].center = vec3(-0.8, 1, 4);
  spheres[4].mat = Material(vec4(1.0, 0.1, 0.1, 1.0), 1.0, 16.0, 0.5, 0.0, 2.0);
  spheres[5].radius = 1.0;
  spheres[5].center = vec3(-2.0, 1.0, 7);
  spheres[5].mat = Material(vec4(0, 0, 0, 0.0), 1.0, 32.0, 0.1, 1.0, 1.5);

  light.dir = normalize(light.dir);
  light.ray = light.dir * light.mag;

  vec2 uv = (fragCoord - 0.5 * iResolution) / iResolution.y;
  ray.dir =
      normalize(vec3(cameraPos.x + uv.x, cameraPos.y + uv.y, 0) - cameraPos);
  ray.factor = 1.0;
  ray.n = 1.0;

  camAngle = mod(camAngle, 2.0 * PI);
  vec3 rotate = vec3(-0.2, camAngle, 0);
  vec3 translate = camOffset + vec3(camRadius * sin(camAngle), 0,
                                    -camRadius * cos(camAngle));
  if (!transform) {
    rotate = vec3(0);
    translate = vec3(0, 1, -1);
  }
  mat3 Rxyz = Rotation(rotate, deg);
  ray.dir = Rxyz * ray.dir;
  ray.origin = translate;

  for (int i = 0; i < totalRays + 1; i++) {
    reflectionRays[i] = Ray(vec3(0), vec3(0), 0.0, 0.0);
    refractionRays[i] = Ray(vec3(0), vec3(0), 0.0, 0.0);
  }
  reflectionRays[0] = ray;

  vec4 finalCol = vec4(0);
  for (int iter = 0; iter < R; iter++)
    finalCol += CastRays(iter);

  fragColor = finalCol;
}

void main() {
  vec4 color;

  mainImage(color, fragCoord * iResolution);
  FragColor = color;
}
