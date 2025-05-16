#version 330

in vec3 position;
out vec4 color;

void main() {
  vec3 lightpos = vec3(-3.0f, 4.0f, -3.0f);
  vec3 lightcolor = vec3(1.0f, 0.0f, 0.0f);

  vec3 diffuse = vec3(0.4f);
  vec3 lightdir = normalize(lightpos - position);
  vec3 normal = normalize(vec3(0.0f, 1.0f, 0.0f));
  float intensity = dot(normal, lightdir);

  color = vec4((diffuse + lightcolor) * intensity, 1.0f);
}