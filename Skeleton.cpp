//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Nemeth Daniel
// Neptun : FTYYJR
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"
#include <iostream>

//================
// Okos Float osztály CPP11 labor megoldásaiból: https://cpp11.eet.bme.hu/lab03/#4
// Nagy részét magam is mefírtam, itt az összehasonlításokhoz fogom használni
namespace smartfloat{
	class Float {
	public:
		Float() = default;
		Float(float value) : value_(value) {}
		explicit operator float() const { return value_; }
	
		static constexpr float epsilon = 1e-4f;
	
	private:
		float value_;
	};
	
	
	Float operator+(Float f1, Float f2) {
		return float(f1) + float(f2);
	}
	
	Float & operator+=(Float &f1, Float f2) {
		return f1 = f1 + f2;
	}
	
	Float operator-(Float f1, Float f2) {
		return float(f1) - float(f2);
	}
	
	Float & operator-=(Float &f1, Float f2) {
		return f1 = f1 - f2;
	}
	
	/* egyoperandusú */
	Float operator-(Float f) {
		return -float(f);
	}
	
	/* kisebb */
	bool operator<(Float f1, Float f2) {
		return float(f1) < float(f2)-Float::epsilon;
	}
	
	/* nagyobb: "kisebb" fordítva */
	bool operator>(Float f1, Float f2) {
		return f2<f1;
	}
	
	/* nagyobb vagy egyenlő: nem kisebb */
	bool operator>=(Float f1, Float f2) {
		return !(f1<f2);
	}
	
	/* kisebb vagy egyenlő: nem nagyobb */
	bool operator<=(Float f1, Float f2) {
		return !(f1>f2);
	}
	
	/* nem egyenlő: kisebb vagy nagyobb */
	bool operator!=(Float f1, Float f2) {
		return f1 > f2 || f1 < f2;
	}
	
	/* egyenlő: nem nem egyenlő */
	bool operator==(Float f1, Float f2) {
		return !(f1 != f2);
	}
	
	/* kíirás */
	std::ostream & operator<< (std::ostream & os, Float f) {
		return os << float(f);
	}
	
	/* beolvasás */
	std::istream & operator>> (std::istream & is, Float & f) {
		float raw_f;
		is >> raw_f;
		f = raw_f;
		return is;
	}

}
// =====================
using namespace smartfloat;

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

class Primitive{
	virtual void genvertices(std::vector<float>& vertices)const =0;
	protected:
		unsigned int vao;	   // virtual world on the GPU
		unsigned int vbo;		// vertex buffer object
		unsigned int dimension=2;
		
		GLenum mode;
	public:
		Primitive(GLenum m):mode{m}{};
		void draw(){
			glGenVertexArrays(1, &vao);	// get 1 vao id
			glBindVertexArray(vao);		// make it active
			glGenBuffers(1, &vbo);	// Generate 1 buffer
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			std::vector<float> vertices;
			genvertices(vertices);
			glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
				sizeof(float)*vertices.size(),  // # bytes
				&*(vertices.begin()),	      	// address
				GL_STATIC_DRAW);	// we do not change later

			glEnableVertexAttribArray(0);  // AttribArray 0
			glVertexAttribPointer(0,       // vbo -> AttribArray 0
				2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
				0, NULL); 		     // stride, offset: tightly packed
			
			glBindVertexArray(vao);  // Draw call
			glDrawArrays(mode, 0 /*startIdx*/, vertices.size()/dimension /*# Elements*/);
			std::cout<<glGetError()<<std::endl;

			}
};

class Polygon: public Primitive{
	std::vector<vec2> vertices;
	void genvertices(std::vector<float>& temp)const override{
		for(auto v: vertices){
			temp.push_back(v.x);
			temp.push_back(v.y);
		}
	}
	public:
		Polygon(const std::vector<vec2>& v):Primitive{GL_TRIANGLE_FAN}, vertices{v}{}
};

class Triangle: public Primitive{
	std::vector<float> vertices;
	void genvertices(std::vector<float>& temp)const override final{
		temp=vertices;
	}
	public:
		Triangle(const std::vector<float>& v):Primitive{GL_TRIANGLES}, vertices{v}{};
};
class Hermite_interpolation_curve: public Primitive{
	
	
	
	float interpolation_increment=0.01;
	vec2 Hermite_value(vec2 leftpoint, vec2 leftspeed, float lefttime, vec2 rightpoint, vec2 rightspeed, float righttime, float t)const{
		vec2 a0=leftpoint;
		vec2 a1=leftspeed;
		float t1_t0=righttime-lefttime;
		float t1_t0_sq=t1_t0*t1_t0;
		float t1_t0_cub=t1_t0_sq*t1_t0;
		vec2 a2=3*(rightpoint-leftpoint)/t1_t0_sq-(rightspeed+2*leftspeed)/t1_t0;
		vec2 a3=2*(leftpoint-rightpoint)/t1_t0_cub+(leftspeed+rightspeed)/t1_t0_sq;
		float t_t0=t-lefttime;
		float t_t0_sq=t_t0*t_t0;
		float t_t0_cb=t_t0_sq*t_t0;
		return a3*t_t0_cb+a2*t_t0_sq+a1*t_t0+a0;

	}

	void genvertices(std::vector<float>& temp)const override final{
		size_t left_time_index=0;
		size_t right_time_index=1;
		float t=times[0];
		while(right_time_index<times.size()){
			vec2 vertex=Hermite_value(
							controlpoints[left_time_index],
							speeds[left_time_index],
							times[left_time_index],
							controlpoints[right_time_index],
							speeds[right_time_index],
							times[right_time_index],
							t
						);
			temp.push_back(vertex.x);
			temp.push_back(vertex.y);

			t+=interpolation_increment;
			if(Float(t)>=Float(times[right_time_index])){
				left_time_index++;
				right_time_index++;
			}
		}
	}
	protected:
		std::vector<vec2> speeds;
		std::vector<vec2> controlpoints;
		std::vector<float> times;

	public:
		Hermite_interpolation_curve(const std::vector<vec2>& cps, const std::vector<vec2>& sps=std::vector<vec2>() ):
			Primitive{GL_TRIANGLE_FAN}, controlpoints{cps}, speeds{sps} {
				for(float t=0;t<speeds.size();++t) times.push_back(t);
			};
		
};

class Catmull_Rom_spline: public Hermite_interpolation_curve{
	size_t numCtrPts=controlpoints.size();
	size_t index(size_t i)const{return i%numCtrPts;};
	public:
		Catmull_Rom_spline(std::vector<vec2>& v):Hermite_interpolation_curve(v){
			
			
			for(size_t i=0;i<numCtrPts;i<++i){
				
				vec2 a=(controlpoints[index(i+1)]-controlpoints[index(i)]);
				a=a/(times[index(i+1)]-times[index(i)]);
				vec2 b=(controlpoints[index(i)]-controlpoints[index(i-1)])/(times[index(i)]-times[index(i-1)]);
				speeds.push_back(0.5*(a+b));

			}
		}
};

std::vector points{vec2( -0.8f, -0.8f), vec2(-0.6f, 1.0f), vec2(0.8f, -0.2f)};
std::vector speeds{vec2( -0.8f, -0.8f),vec2( -0.6f, 1.0f), vec2(0.8f, -0.2f)};
//Polygon poly{points};
//Hermite_interpolation_curve tri{points, speeds};
//Triangle tri2{std::vector{ -0.8f, -0.8f, -0.6f, 1.0f, 0.8f, -0.2f }};
Catmull_Rom_spline crs{points};
GPUProgram gpuProgram; // vertex and fragment shaders


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	
	crs.draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
