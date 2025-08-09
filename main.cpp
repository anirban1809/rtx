// main.cpp
// Cornell Box GPU path tracer with progressive HDR accumulation (GLFW + GLEW).
// Accumulate in linear space, tone-map in a separate display pass to avoid
// darkening.

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

static const int START_WIDTH = 1280;
static const int START_HEIGHT = 720;

struct float3 {
  float x, y, z;
};
static float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------- Shaders ----------------
static const char *kVS = R"(#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main(){
    vUV = aPos*0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

// Path tracer fragment shader (writes *linear HDR accumulation*; no
// tone-mapping here)
static const char *kFS_PT = R"(#version 330 core
out vec4 FragColor;
in vec2 vUV;

uniform sampler2D uPrev;      // previous *linear* accumulation
uniform int   uFrame;         // current frame index (starts at 1)
uniform vec2  uResolution;
uniform vec3  uCamPos;
uniform mat3  uCamBasis;      // columns: right, up, forward

// Light params
uniform vec3  uLightPos;      // center of rect light (top)
uniform vec2  uLightSize;     // half-extents in X,Z
uniform vec3  uLightEmit;     // emission (W/m^2-ish)

uint hash_uvec3(uvec3 x){
    x = (x ^ (x>>16u)) * 2246822519u;
    x ^= (x>>13u);
    x *= 3266489917u;
    x ^= (x>>16u);
    return x.x ^ x.y ^ x.z;
}
float rnd(inout uvec3 state){
    state = uvec3(hash_uvec3(state), state.y + 0x9e3779b9u, state.z + 0x85ebca6bu);
    return float(hash_uvec3(state)) / 4294967296.0;
}

struct Ray { vec3 o; vec3 d; };
struct Hit { float t; vec3 n; int matId; bool hit; vec3 pos; };

// 0: white, 1: red, 2: green, 3: box white
vec3 albedo(int id){
    if(id==1) return vec3(0.75,0.15,0.15);
    if(id==2) return vec3(0.15,0.75,0.15);
    return vec3(0.75);
}

bool intersectPlane(Ray r, vec3 p, vec3 n, out float t){
    float d = dot(n, r.d);
    if(abs(d) < 1e-4) return false;
    float tt = dot(n, p - r.o) / d;
    if(tt > 1e-4){ t = tt; return true; }
    return false;
}

bool aabbHit(Ray r, vec3 bmin, vec3 bmax, out float tNear, out vec3 n){
    vec3 inv = 1.0 / r.d;
    vec3 t0 = (bmin - r.o) * inv;
    vec3 t1 = (bmax - r.o) * inv;
    vec3 tsm = min(t0, t1);
    vec3 tbg = max(t0, t1);
    float tN = max(max(tsm.x, tsm.y), tsm.z);
    float tF = min(min(tbg.x, tbg.y), tbg.z);
    if(tF < 0.0 || tN > tF) return false;
    tNear = (tN>1e-4) ? tN : tF;
    vec3 hitp = r.o + r.d * tNear;
    const float eps=1e-3;
    if(abs(hitp.x - bmin.x) < eps) n = vec3(-1,0,0);
    else if(abs(hitp.x - bmax.x) < eps) n = vec3(1,0,0);
    else if(abs(hitp.y - bmin.y) < eps) n = vec3(0,-1,0);
    else if(abs(hitp.y - bmax.y) < eps) n = vec3(0,1,0);
    else if(abs(hitp.z - bmin.z) < eps) n = vec3(0,0,-1);
    else n = vec3(0,0,1);
    return true;
}

Hit sceneHit(Ray r){
    Hit h; h.t = 1e9; h.hit=false; h.matId=0;

    float t;
    // Floor y=0
    if(intersectPlane(r, vec3(0,0,0), vec3(0,1,0), t)){ if(t<h.t){ h.t=t; h.n=vec3(0,1,0); h.matId=0; h.hit=true; } }
    // Ceiling y=2
    if(intersectPlane(r, vec3(0,2,0), vec3(0,-1,0), t)){ if(t<h.t){ h.t=t; h.n=vec3(0,-1,0); h.matId=0; h.hit=true; } }
    // Back wall z= -1
    if(intersectPlane(r, vec3(0,0,-1), vec3(0,0,1), t)){ if(t<h.t){ h.t=t; h.n=vec3(0,0,1); h.matId=0; h.hit=true; } }
    // Left wall x = -1 (red)
    if(intersectPlane(r, vec3(-1,0,0), vec3(1,0,0), t)){ if(t<h.t){ h.t=t; h.n=vec3(1,0,0); h.matId=1; h.hit=true; } }
    // Right wall x = 1 (green)
    if(intersectPlane(r, vec3(1,0,0), vec3(-1,0,0), t)){ if(t<h.t){ h.t=t; h.n=vec3(-1,0,0); h.matId=2; h.hit=true; } }

    // Short box
    {
        float tNear; vec3 nn;
        if(aabbHit(r, vec3(-0.6,0.0,-0.2), vec3(-0.2,0.6,0.4), tNear, nn)){
            if(tNear < h.t){ h.t=tNear; h.n=nn; h.matId=3; h.hit=true; }
        }
    }
    // Tall box
    {
        float tNear; vec3 nn;
        if(aabbHit(r, vec3(0.2,0.0,-0.7), vec3(0.6,1.2,-0.2), tNear, nn)){
            if(tNear < h.t){ h.t=tNear; h.n=nn; h.matId=3; h.hit=true; }
        }
    }

    if(h.hit) h.pos = r.o + r.d*h.t;
    return h;
}

vec3 sampleCosHemisphere(vec3 n, float u1, float u2){
    float r = sqrt(u1);
    float theta = 6.2831853 * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u1));
    vec3 h = abs(n.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
    vec3 t = normalize(cross(h, n));
    vec3 b = cross(n, t);
    return normalize(t*x + b*y + n*z);
}

bool visibleToLight(vec3 p, vec3 q){
    Ray r; r.o = p + 1e-3*normalize(q-p); r.d = normalize(q-p);
    Hit h = sceneHit(r);
    float distL = length(q - p);
    return (!h.hit) || (h.t > distL - 1e-3);
}

vec3 sampleRect(vec2 halfSize, float u1, float u2){
    float sx = (u1*2.0 - 1.0) * halfSize.x;
    float sz = (u2*2.0 - 1.0) * halfSize.y;
    return vec3(sx, 0.0, sz);
}

void main(){
    uvec3 rng = uvec3(uint(gl_FragCoord.x), uint(gl_FragCoord.y), uint(uFrame*1664525));
    vec2 px = (gl_FragCoord.xy + vec2(rnd(rng), rnd(rng))) / uResolution;
    vec2 ndc = px*2.0 - 1.0;

    float fov = radians(45.0);
    vec3 rd_cam = normalize(vec3(ndc.x * tan(fov*0.5) * (uResolution.x/uResolution.y),
                                 -ndc.y * tan(fov*0.5),
                                 1.0));
    vec3 ro = uCamPos;
    vec3 rd = normalize(uCamBasis * rd_cam);

    vec3 L = vec3(0.0);
    vec3 T = vec3(1.0);

    const int MAX_BOUNCES = 4;
    for(int bounce=0; bounce<MAX_BOUNCES; ++bounce){
        Ray r; r.o = ro; r.d = rd;
        Hit h = sceneHit(r);
        if(!h.hit){
            // escaped front opening: black environment
            break;
        }

        // Next event estimation toward rectangular ceiling light
        vec3 lightCenter = uLightPos;
        vec2 halfSize = uLightSize;
        vec2 xi = vec2(rnd(rng), rnd(rng));
        vec3 lp = lightCenter + sampleRect(halfSize, xi.x, xi.y);
        vec3 wi = normalize(lp - h.pos);
        float dist2 = max(1e-6, dot(lp - h.pos, lp - h.pos));
        float cosL = max(0.0, dot(vec3(0,-1,0), -wi)); // light faces downward
        float cosH = max(0.0, dot(h.n, wi));
        if(cosL > 0.0 && cosH > 0.0){
            if(visibleToLight(h.pos + h.n*1e-3, lp)){
                float area = 4.0 * halfSize.x * halfSize.y;
                float G = (cosL * cosH) / dist2;
                vec3 f = albedo(h.matId) / 3.14159265;
                L += T * (uLightEmit * f * G * area);
            }
        }

        // Diffuse bounce
        vec3 newDir = sampleCosHemisphere(h.n, rnd(rng), rnd(rng));
        T *= albedo(h.matId);

        // Russian roulette
        float p = max(max(T.r, T.g), T.b);
        if(bounce > 1){
            float rr = float(rnd(rng));
            float keep = clamp(p, 0.05, 0.99);
            if(rr > keep) break;
            T /= keep;
        }

        ro = h.pos + h.n*1e-3;
        rd = newDir;
    }

    // Progressive *linear* accumulation
    vec3 prev = texture(uPrev, vUV).rgb;          // previous linear average
    float frameF = float(uFrame);
    vec3 color = (prev*(frameF-1.0) + L) / frameF; // still linear

    FragColor = vec4(color, 1.0); // no tone-map here
}
)";

// Display shader (tone-map + gamma to the default framebuffer)
static const char *kFS_Display = R"(#version 330 core
out vec4 FragColor;
in vec2 vUV;
uniform sampler2D uAccum; // linear HDR average
vec3 tonemapACES(vec3 x){
    return clamp((x*(2.51*x + 0.03)) / (x*(2.43*x + 0.59) + 0.14), 0.0, 1.0);
}
void main(){
    vec3 hdr = texture(uAccum, vUV).rgb;
    vec3 ldr = tonemapACES(hdr);
    // gamma 2.2
    ldr = pow(ldr, vec3(1.0/2.2));
    FragColor = vec4(ldr, 1.0);
}
)";

// ---------------- GL debug ----------------
static void GLAPIENTRY glDebugOutput(GLenum, GLenum, GLuint, GLenum severity,
                                     GLsizei, const GLchar *message,
                                     const void *) {
  if (severity == GL_DEBUG_SEVERITY_NOTIFICATION)
    return;
  fprintf(stderr, "GL Debug: %s\n", message);
}

// ---------------- App ----------------
struct App {
  GLFWwindow *win = nullptr;
  GLuint vao = 0, vbo = 0;
  GLuint ptProg = 0, dispProg = 0, vs = 0, fsPT = 0, fsDisp = 0;
  GLuint fboA = 0, texA = 0; // current write (accum)
  GLuint fboB = 0, texB = 0; // previous read (accum)
  int width = START_WIDTH, height = START_HEIGHT;
  int frame = 1;
  bool mouseCaptured = false;

  // Camera
  float3 camPos{0.0f, 1.0f, 3.0f};
  float yaw = 3.14159f; // look toward -Z
  float pitch = 0.0f;

  // Light
  float3 lightPos{0.0f, 1.95f, 0.0f};
  float lightHalfX = 0.25f, lightHalfZ = 0.18f;
  float lightIntensity = 7.0f;

  double lastX = width * 0.5, lastY = height * 0.5;
  bool firstMouse = true;
  bool mKeyLatch = false; // debounce for 'M'

  void resetAccum() { frame = 1; }

  // Create RGBA16F accumulation target and clear to zero
  void createAccumTarget(GLuint &fbo, GLuint &tex) {
    if (tex) {
      glDeleteTextures(1, &tex);
      tex = 0;
    }
    if (fbo) {
      glDeleteFramebuffers(1, &fbo);
      fbo = 0;
    }

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA,
                 GL_HALF_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           tex, 0);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "FBO incomplete\n");
      std::exit(1);
    }
    // Clear to 0 so first accumulation frame is correct
    glViewport(0, 0, width, height);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
      GLint len = 0;
      glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
      std::vector<char> log(len);
      glGetShaderInfoLog(s, len, nullptr, log.data());
      fprintf(stderr, "Shader compile error:\n%s\n", log.data());
      std::exit(1);
    }
    return s;
  }
  static GLuint link(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
      GLint len = 0;
      glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
      std::vector<char> log(len);
      glGetProgramInfoLog(p, len, nullptr, log.data());
      fprintf(stderr, "Program link error:\n%s\n", log.data());
      std::exit(1);
    }
    return p;
  }

  void initGL() {
    if (!glfwInit()) {
      fprintf(stderr, "glfwInit failed\n");
      std::exit(1);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef _DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif
    win = glfwCreateWindow(width, height,
                           "Cornell Box Path Tracer (HDR Accum + Display)",
                           nullptr, nullptr);
    if (!win) {
      fprintf(stderr, "Failed to create window\n");
      std::exit(1);
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    GLenum err = glewInit();
    if (err != GLEW_OK) {
      fprintf(stderr, "glewInit: %s\n", glewGetErrorString(err));
      std::exit(1);
    }

#ifdef _DEBUG
    if (GLEW_KHR_debug || GLEW_ARB_debug_output) {
      glEnable(GL_DEBUG_OUTPUT);
      glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
      glDebugMessageCallback(glDebugOutput, nullptr);
    }
#endif

    // Fullscreen quad
    float verts[8] = {-1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float),
                          (void *)0);

    // Shaders
    vs = compile(GL_VERTEX_SHADER, kVS);
    fsPT = compile(GL_FRAGMENT_SHADER, kFS_PT);
    fsDisp = compile(GL_FRAGMENT_SHADER, kFS_Display);
    ptProg = link(vs, fsPT);
    dispProg = link(vs, fsDisp);

    // Accum targets
    createAccumTarget(fboA, texA);
    createAccumTarget(fboB, texB);
  }

  void resize(int w, int h) {
    width = (w > 0 ? w : 1);
    height = (h > 0 ? h : 1);
    createAccumTarget(fboA, texA);
    createAccumTarget(fboB, texB);
    glViewport(0, 0, width, height);
    resetAccum();
  }

  void handleInput(float dt) {
    const float baseSpeed = 1.2f;
    float speed = (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
                      ? baseSpeed * 4.0f
                      : baseSpeed;

    float cy = cosf(yaw), sy = sinf(yaw);
    float cp = cosf(pitch), sp = sinf(pitch);
    float3 fwd = {sinf(yaw) * cp, sp, -cosf(yaw) * cp};
    float3 right = {cosf(yaw), 0.0f, sinf(yaw)};

    bool changed = false;

    if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) {
      camPos.x += fwd.x * speed * dt;
      camPos.y += fwd.y * speed * dt;
      camPos.z += fwd.z * speed * dt;
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) {
      camPos.x -= fwd.x * speed * dt;
      camPos.y -= fwd.y * speed * dt;
      camPos.z -= fwd.z * speed * dt;
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) {
      camPos.x -= right.x * speed * dt;
      camPos.z -= right.z * speed * dt;
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) {
      camPos.x += right.x * speed * dt;
      camPos.z += right.z * speed * dt;
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) {
      camPos.y -= speed * dt;
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) {
      camPos.y += speed * dt;
      changed = true;
    }

    // Light controls
    if (glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS) {
      lightPos.x -= 1.0f * dt;
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS) {
      lightPos.x += 1.0f * dt;
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS) {
      lightPos.z += 1.0f * dt;
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS) {
      lightPos.z -= 1.0f * dt;
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS) {
      lightPos.y = clampf(lightPos.y - 1.0f * dt, 1.2f, 1.99f);
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) {
      lightPos.y = clampf(lightPos.y + 1.0f * dt, 1.2f, 1.99f);
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_MINUS) == GLFW_PRESS) {
      lightHalfX = clampf(lightHalfX - 0.25f * dt, 0.05f, 0.6f);
      lightHalfZ = clampf(lightHalfZ - 0.25f * dt, 0.05f, 0.6f);
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_EQUAL) == GLFW_PRESS) {
      lightHalfX = clampf(lightHalfX + 0.25f * dt, 0.05f, 0.6f);
      lightHalfZ = clampf(lightHalfZ + 0.25f * dt, 0.05f, 0.6f);
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_COMMA) == GLFW_PRESS) {
      lightIntensity = clampf(lightIntensity - 2.0f * dt, 0.5f, 50.0f);
      changed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_PERIOD) == GLFW_PRESS) {
      lightIntensity = clampf(lightIntensity + 2.0f * dt, 0.5f, 50.0f);
      changed = true;
    }

    if (changed)
      resetAccum();
  }

  static void cursorPosCB(GLFWwindow *w, double xpos, double ypos) {
    App *app = (App *)glfwGetWindowUserPointer(w);
    if (!app->mouseCaptured) {
      app->lastX = xpos;
      app->lastY = ypos;
      app->firstMouse = false;
      return;
    }
    if (app->firstMouse) {
      app->lastX = xpos;
      app->lastY = ypos;
      app->firstMouse = false;
    }
    float dx = float(xpos - app->lastX);
    float dy = float(ypos - app->lastY);
    app->lastX = xpos;
    app->lastY = ypos;

    const float sens = 0.0025f;
    app->yaw += dx * sens;
    app->pitch += -dy * sens;
    const float lim = 1.2f;
    if (app->pitch > lim)
      app->pitch = lim;
    if (app->pitch < -lim)
      app->pitch = -lim;

    app->resetAccum();
  }

  static void scrollCB(GLFWwindow *w, double, double yoff) {
    App *app = (App *)glfwGetWindowUserPointer(w);
    app->lightIntensity =
        clampf(app->lightIntensity + float(yoff), 0.5f, 50.0f);
    app->resetAccum();
  }

  static void framebufferCB(GLFWwindow *w, int ww, int hh) {
    App *app = (App *)glfwGetWindowUserPointer(w);
    app->resize(ww, hh);
  }

  void run() {
    initGL();
    glfwSetWindowUserPointer(win, this);
    glfwSetCursorPosCallback(win, cursorPosCB);
    glfwSetScrollCallback(win, scrollCB);
    glfwSetFramebufferSizeCallback(win, framebufferCB);

    auto t0 = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(win)) {
      if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(win, true);

      // Toggle mouse capture (debounced)
      if (glfwGetKey(win, GLFW_KEY_M) == GLFW_PRESS) {
        if (!mKeyLatch) {
          mouseCaptured = !mouseCaptured;
          glfwSetInputMode(win, GLFW_CURSOR,
                           mouseCaptured ? GLFW_CURSOR_DISABLED
                                         : GLFW_CURSOR_NORMAL);
          resetAccum();
          mKeyLatch = true;
        }
      } else {
        mKeyLatch = false;
      }

      auto t1 = std::chrono::high_resolution_clock::now();
      float dt = std::chrono::duration<float>(t1 - t0).count();
      t0 = t1;
      if (dt > 0.1f)
        dt = 0.1f;

      handleInput(dt);

      // ---- Path trace pass: write linear accumulation to A, using B as
      // previous ----
      glBindFramebuffer(GL_FRAMEBUFFER, fboA);
      glViewport(0, 0, width, height);
      glDisable(GL_DEPTH_TEST);
      glUseProgram(ptProg);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, texB);
      glUniform1i(glGetUniformLocation(ptProg, "uPrev"), 0);
      glUniform1i(glGetUniformLocation(ptProg, "uFrame"), frame);
      glUniform2f(glGetUniformLocation(ptProg, "uResolution"), (float)width,
                  (float)height);

      // camera basis columns
      float cy = cosf(yaw), sy = sinf(yaw);
      float cp = cosf(pitch), sp = sinf(pitch);
      float3 right = {cy, 0.0f, sy};
      float3 up = {sy * sp, cp, -cy * sp};
      float3 fwd = {-sy * cp, sp, cy * cp};

      glUniform3f(glGetUniformLocation(ptProg, "uCamPos"), camPos.x, camPos.y,
                  camPos.z);
      glUniformMatrix3fv(glGetUniformLocation(ptProg, "uCamBasis"), 1, GL_FALSE,
                         (const float[9]){right.x, right.y, right.z, up.x, up.y,
                                          up.z, fwd.x, fwd.y, fwd.z});

      glUniform3f(glGetUniformLocation(ptProg, "uLightPos"), lightPos.x,
                  lightPos.y, lightPos.z);
      glUniform2f(glGetUniformLocation(ptProg, "uLightSize"), lightHalfX,
                  lightHalfZ);
      glUniform3f(glGetUniformLocation(ptProg, "uLightEmit"), lightIntensity,
                  lightIntensity, lightIntensity);

      glBindVertexArray(vao);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

      glBindFramebuffer(GL_FRAMEBUFFER, 0);

      // ---- Display pass: sample A (linear), tone-map to default framebuffer
      // ----
      glViewport(0, 0, width, height);
      glUseProgram(dispProg);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, texA);
      glUniform1i(glGetUniformLocation(dispProg, "uAccum"), 0);
      glBindVertexArray(vao);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

      glfwSwapBuffers(win);
      glfwPollEvents();

      // ---- Swap A/B for next frame ----
      std::swap(fboA, fboB);
      std::swap(texA, texB);

      frame++;
    }

    cleanup();
  }

  void cleanup() {
    if (texA)
      glDeleteTextures(1, &texA);
    if (texB)
      glDeleteTextures(1, &texB);
    if (fboA)
      glDeleteFramebuffers(1, &fboA);
    if (fboB)
      glDeleteFramebuffers(1, &fboB);
    if (vbo)
      glDeleteBuffers(1, &vbo);
    if (vao)
      glDeleteVertexArrays(1, &vao);
    if (ptProg)
      glDeleteProgram(ptProg);
    if (dispProg)
      glDeleteProgram(dispProg);
    if (vs)
      glDeleteShader(vs);
    if (fsPT)
      glDeleteShader(fsPT);
    if (fsDisp)
      glDeleteShader(fsDisp);
    if (win) {
      glfwDestroyWindow(win);
    }
    glfwTerminate();
  }
};

int main() {
  App app;
  app.run();
  return 0;
}
