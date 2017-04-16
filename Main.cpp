//=============================================================================================
// Framework for the ray tracing homework
// ---------------------------------------------------------------------------------------------
// Name    : 
// Neptun : 
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

struct vec3 {
	float x, y, z;

	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }

	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }

	vec3 operator/(float a) const { return vec3(x / a, y / a, z / a); }

	vec3 operator+(const vec3& v) const {
		return vec3(x + v.x, y + v.y, z + v.z);
	}

	vec3 operator+=(const vec3& vec) {
		this->x += vec.x;
		this->y += vec.y;
		this->z += vec.z;
		return *this;
	}

	vec3 operator-(const vec3& v) const {
		return vec3(x - v.x, y - v.y, z - v.z);
	}
	vec3 operator*(const vec3& v) const {
		return vec3(x * v.x, y * v.y, z * v.z);
	}
	vec3 operator-() const {
		return vec3(-x, -y, -z);
	}
	vec3 normalize() const {
		return (*this) * (1.0f / (Length() + 0.000001));
	}
	float Length() const { return sqrtf(x * x + y * y + z * z); }

	operator float*() { return &x; }
};

float dot(vec3 v1, vec3 v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(vec3 v1, vec3 v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}



void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

	out vec2 texcoord;

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates

	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

const int MAXDEPTH = 3;
const float EPSILON = 0.00001f;

class Material {
	public:
		vec3 kd, ks, F0;
		float n;
		boolean reflective, refractive, rough;

		void SetDiffuseColor(vec3 Kd) { kd = Kd / M_PI; }
		void SetSpecularColor(vec3 Ks) { ks = Ks * (n + 2) / M_PI / 2.0; }

		vec3 reflect(vec3 inDir, vec3 normal)
		{
			return inDir - normal * dot(normal, inDir) * 2.0f;
		};

		vec3 refract(vec3 inDir, vec3 normal) {
			float ior = n;
			float cosa = -dot(normal, inDir);
			if (cosa < 0) { cosa = -cosa; normal = -normal; ior = 1 / n; }
			float disc = 1 - (1 - cosa * cosa) / ior / ior;
			if (disc < 0) { return reflect(inDir, normal); }
			return inDir / ior + normal * (cosa / ior - sqrt(disc));
		}

		vec3 Fresnel(vec3 inDir, vec3 normal) {
			float cosa = fabs(dot(normal, inDir));
			return F0 + (vec3(1, 1, 1) - F0) * pow(1 - cosa, 5);
		}

		vec3 shade(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 inRad)
		{
			vec3 reflRad(0, 0, 0);
			float cosTheta = dot(normal, lightDir);
			if (cosTheta < 0) return reflRad;
			reflRad = inRad * kd * cosTheta;;
			vec3 halfway = (viewDir + lightDir).normalize();
			float cosDelta = dot(normal, halfway);
			if (cosDelta < 0) return reflRad;
			return reflRad + inRad * ks * pow(cosDelta, n);
		}
};

struct Hit {
	float t;
	vec3 position;
	vec3 normal;
	Material* material;
	Hit() { t = -1; };
};

class Ray {
public:
	vec3 start, dir;
	Ray(vec3 start0, vec3 dir0) {
		start = start0; dir = dir0; dir.normalize();
	}
};

struct Intersectable {
	Material material;
	virtual Hit intersect(const Ray ray) = 0;
};

class Sphere : public Intersectable {
	vec3 center;
	float radius;
public:
	Sphere(vec3 c, float r) {
		center = c;
		radius = r;
	}
	Hit intersect(Ray ray) {
		Hit retval;
		retval.material = &material;

		float a = dot(ray.dir, ray.dir);
		float b = dot((ray.start - center), (ray.dir * 2));
		float c = dot((ray.start - center), (ray.start - center) - (radius*radius));

		float d = b * b - 4 * a * c;
		if (d < 0)	//prevent getting rekt
			retval.t = -1.0;
		else {
			float t1 = (-1.0 * b - sqrt(b * b - 4 * a * c)) / 2.0 * a;
			float t2 = (-1.0 * b + sqrt(b * b - 4 * a * c)) / 2.0 * a;
			if (t1<t2)
				retval.t = t1;
			else
				retval.t = t2;
		}
		if (fabs(retval.t) < EPSILON)
			retval.t = -1;
		retval.position = ray.start + ray.dir * retval.t;
		retval.normal = getNormal(ray.start + ray.dir * retval.t);
		return retval;
	}
	vec3 getNormal(vec3 intersect) {
		vec3 retval = (intersect - center) * 2;
		return retval.normalize();
	}
};

class Camera {
public:
	vec3 position, lookat, up, right;
};

class Light {
public:
	vec3 color;
	vec3 position;
};

class Scene {
public:
	Camera camera;
	std::vector<Intersectable*> objects;
	std::vector<Light> lights;

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* obj : objects) {
			Hit hit = obj->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) {
				bestHit = hit;
			}
		}
		return bestHit;
	}

	Ray GetRay(int x, int y) {
		vec3 _lookat = camera.lookat;
		vec3 _right = camera.right*(x - windowWidth / 2.0) / (windowWidth / 2.0);
		vec3 _up = camera.up*(y - windowHeight / 2.0) / (windowHeight / 2.0);
		Ray ray(camera.position, (_lookat + _right + _up).normalize());
		return ray;

	}

	int sign(float f) {
		return f < 0 ? -1 : 1;
	}

	vec3 trace(Ray ray, int depth) {
		vec3 La;
		if (depth > MAXDEPTH) {
			return La;
		}
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) { return La; };
		vec3 outRadiance;
		outRadiance = La*hit.material->kd;
		if (hit.material->rough) {
			for (Light l : lights) {
				Ray shadowRay(hit.position + hit.normal*EPSILON*sign(dot(hit.normal, (ray.dir).normalize())), (l.position).normalize());
				Hit shadowHit = firstIntersect(shadowRay);
				if (shadowHit.t < 0 || shadowHit.t > 10000) {
					outRadiance += hit.material->shade(hit.normal, (ray.dir).normalize(), (l.position).normalize(), l.color);
				}
			}
		}
		if (hit.material->reflective) {
			vec3 reflectionDir = hit.material->reflect(ray.dir, hit.normal).normalize();
			Ray reflectedRay(hit.position + hit.normal*EPSILON*sign(dot(hit.normal, (ray.dir).normalize())), reflectionDir);
			outRadiance += trace(reflectedRay, depth + 1)*hit.material->Fresnel(ray.dir.normalize(), hit.normal);
		}
		if (hit.material->refractive) {
			vec3 refractionDir = hit.material->refract(ray.dir, hit.normal).normalize();
			Ray refractedRay(hit.position + hit.normal*EPSILON*sign(dot(hit.normal, (ray.dir).normalize())), refractionDir);
			outRadiance += trace(refractedRay, depth + 1)*(vec3(1, 1, 1) - hit.material->Fresnel(ray.dir.normalize(), hit.normal));
		}
		//printf("%f, %f, %f\n", outRadiance.x, outRadiance.y, outRadiance.z);
		return outRadiance;
		
	}
};

// handle of the shader program
unsigned int shaderProgram;

class FullScreenTexturedQuad {
	unsigned int vao, textureId;	// vertex array object id and texture id
public:
	void Create(vec3 image[windowWidth * windowHeight]) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

								// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		static float vertexCoords[] = { -1, -1, 1, -1, -1, 1,
			1, -1, 1, 1, -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
																							   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

																	  // Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
		}
		glDrawArrays(GL_TRIANGLES, 0, 6);	// draw two triangles forming a quad
	}
};

// The virtual world: single quad
FullScreenTexturedQuad fullScreenTexturedQuad;
Scene scene;

vec3 background[windowWidth * windowHeight];	// The image, which stores the ray tracing result


												// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	Material kek;
	kek.F0 = 0.5f;
	kek.kd = 0.5f;
	kek.ks = 0.5f;
	kek.n = 0.5f;
	kek.rough = true;

	Light l;
	l.color = vec3(0.5f, 0.5f, 0.5f);
	l.position = vec3(100.0f, 300.0f, 300.0f);

	Sphere s = Sphere(vec3(0,0,300.0f), 100.0f);
	s.material = kek;

	Camera c;
	c.position = vec3(0, 0, -500.0f);
	c.lookat = vec3(0, 0, 1.0f);
	c.up = vec3(0,1.0f,0);
	c.right = vec3(1.0f, 0, 0);

	scene.objects.push_back(&s);
	scene.lights.push_back(l);

	// Ray tracing fills the image called background
	for (int x = 0; x < windowWidth; x++) {
		for (int y = 0; y < windowHeight; y++) {
			Ray r = scene.GetRay(x, y);
			background[y * windowWidth + x] = scene.trace(r, 0); // vec3((float)x / windowWidth, (float)y / windowHeight, 0);
			if (background[y * windowWidth + x].x > 0.0f || background[y * windowWidth + x].y > 0.0f || background[y * windowWidth + x].z > 0.0f) {
				printf("%f, %f, %f\n", background[y * windowWidth + x].x, background[y * windowWidth + x].y, background[y * windowWidth + x].z);
			}
		}
	}

	fullScreenTexturedQuad.Create(background);

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
