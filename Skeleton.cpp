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
	Float operator/(Float f1, Float f2) {
		return float(f1) / float(f2);
	}
	
	Float & operator/=(Float &f1, Float f2) {
		return f1 = f1 / f2;
	}
	Float operator*(Float f1, Float f2) {
		return float(f1) * float(f2);
	}
	
	Float & operator*=(Float &f1, Float f2) {
		return f1 = f1 * f2;
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

bool operator==(const vec2& v1, const vec2& v2){
	return Float(v1.x)==Float(v2.x) && (Float(v1.y)==Float(v2.y));
}
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

GPUProgram gpuProgram; // vertex and fragment shaders

bool segmentIntersect(vec2 segment1_point1, vec2 segment1_point2, vec2 segment2_point1,vec2 segment2_point2){
	if(segment1_point1==segment2_point1
		|| segment1_point1==segment2_point2
		|| segment1_point2==segment2_point1
		|| segment1_point2==segment2_point2)

		return false;
	Float A=segment1_point1.x - segment1_point2.x;
	Float B=segment2_point2.x- segment2_point1.x;
	Float C=segment1_point1.y-segment1_point2.y;
	Float D= segment2_point2.y-segment1_point2.y;
	Float E=segment1_point2.x - segment2_point2.x;
	Float F= segment1_point2.y- segment2_point2.y;
	Float t2=(F-C*E/A)/(D-C*B/A);
	if(t2>1 || t2<0) return false;
	Float t1=E/A-B/A*((F-C*E/A)/(D-C*B/A));
	return (t1>0 && t1<1);
	
}



class Primitive{
	virtual void genvertices(std::vector<float>& vertices)const =0;
	protected:
		unsigned int vao;	   // virtual world on the GPU
		unsigned int vbo;		// vertex buffer object
		unsigned int dimension=2;
		vec3 color;
		
		GLenum mode;
	public:
		Primitive(GLenum m, vec3 color=vec3(0.0f, 1.0f, 0.0f) /*green*/):mode{m},color(color){};
		void setColor(vec3 color_){color=color_;}
		void draw(){
			// Set color to (0, 1, 0) = green
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, color.x, color.y, color.z); // 3 floats

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
			//std::cout<<glGetError()<<std::endl;

			}
};
class Points: public Primitive{
	std::vector<vec2> points;
	void genvertices(std::vector<float>& vertices)const{
		for(auto p:points){
			vertices.push_back(p.x);
			vertices.push_back(p.y);
		}
	}

	public:
		Points():Primitive(GL_POINTS){}
		Points(const vec3& color=vec3(0.0f, 1.0f, 0.0f), std::vector<vec2> points=std::vector<vec2>()):Primitive(GL_POINTS, color),points(points){}
		void add(const vec2& point){points.push_back(point);}
		

};

class Polygon: public Primitive{
	private:
		virtual size_t index(int i)const{
			int numcontrolpoints=vertices.size();
			if(i>=0){
				return i%numcontrolpoints;
			}
			else{
				int negmod=(-i)%numcontrolpoints;
				int ret=numcontrolpoints-negmod;
				return ret%numcontrolpoints;
			}
		}
		bool isDiagonal(const vec2& point1, const vec2& point2)const{
			if(segmentIntersect(point1, point2, vertices.back(), vertices.front())) return false;
			
			for(size_t i=0;i<vertices.size()-1;++i){
				if(segmentIntersect(point1, point2, vertices[i], vertices[i+1])) return false;
				
			}
			if(!doesContain((point1+point2)/2)) return false;
			return true;

		}
		void genvertices(std::vector<float>& temp)const override{
			if(vertices.size()>=3){
				std::vector<vec2> triangles;
				triangles=earclipping(triangles);
				for(auto v: triangles){
					temp.push_back(v.x);
					temp.push_back(v.y);
				}
			}
			
		}

		

		std::vector<vec2>& earclipping(std::vector<vec2>& triangles)const{
			size_t numberofclips=0;
			std::vector<bool> isClipped(vertices.size(),false);
			
			size_t i=0;
			size_t numofclips_at0;
			size_t numofclips_atLast;
			
			while(numberofclips<vertices.size()-3){
				
				if(i==0){numofclips_at0=numberofclips;}
				if(isClipped[i]==false){
					size_t left=index(i-1);
					while(isClipped[left]==true){left=index(left-1);}
					size_t right=index(i+1);
					while(isClipped[right]==true){right=index(right+1);}
					if(isDiagonal(vertices[left], vertices[right])){
						triangles.push_back(vertices[left]);
						triangles.push_back(vertices[i]);
						triangles.push_back(vertices[right]);
						isClipped[i]=true;
						++numberofclips;
					}
				}
				if(i==vertices.size()-1){
					numofclips_atLast=numberofclips;
					if(numofclips_at0 == numofclips_atLast){
					std::cerr<<"Nem egyszerű polinom , tesszalláció megáll"<<std::endl;
					return triangles;
					}
				}
				i=index(i+1);
			}
			while(numberofclips-vertices.size()>0){
				if(isClipped[i]==false){
					triangles.push_back(vertices[i]);
					++numberofclips;
					isClipped[i]==true;
					
				}
				i=index(i+1);
			}
			return triangles;
			
		}

		bool rayCastIntersect(vec2 point, vec2 segmentVertex1, vec2 segmentVertex2)const{
				if(point.y == segmentVertex2.y) return false;
				vec2 A=segmentVertex1;
				vec2 B=segmentVertex2;
				if(Float(A.y) > Float(B.y)) std::swap(A,B);
				if(Float(point.y)>Float(B.y) || Float(point.y)<Float(A.y)) return false;
				if(Float(point.x)> std::max(Float(A.x), Float(B.x))) return false;
				if(Float(point.x)< std::min(Float(A.x), Float(B.x))) return true;
				Float dx=Float(B.x)-Float(A.x);
				Float dy=Float(B.y)-Float(A.y);
				Float slopeAB=dy/dx;
				bool res;
				if(dy/dx >0) {
					res=((Float(point.y)-Float(A.y))/(Float(point.x)-Float(A.x)) > slopeAB);
					return res;
				}
				else {
					res= ((Float(A.y)-Float(point.y))/(Float(A.x)-Float(point.x)) > slopeAB);
					return res;
				};
		};
	protected:
		std::vector<vec2> vertices;
	public:
		Polygon(const std::vector<vec2>& v=std::vector<vec2>()):Primitive{GL_TRIANGLES}, vertices{v}{};
		bool doesContain(const vec2& point)const{
			bool contain=false;
			bool intersect;
			if(rayCastIntersect(point, vertices.back(),vertices.front())) contain=(!contain);
			for(size_t i=0;i<vertices.size()-1;++i){
				intersect=rayCastIntersect(point, vertices[i],vertices[i+1]);
				if(intersect){
					contain=(!contain);
				}
					
			}
			return contain;

		}
		virtual void add(const vec2& point){
			vertices.push_back(point);
		}
};

class Triangle: public Primitive{
	std::vector<float> vertices;
	void genvertices(std::vector<float>& temp)const override final{
		temp=vertices;
	}
	public:
		Triangle(const std::vector<float>& v):Primitive{GL_TRIANGLES}, vertices{v}{};
};
class Hermite_interpolation_curve: public Polygon{
	protected:
		size_t ngon;
		std::vector<vec2> speeds;
		std::vector<vec2> controlpoints;
		std::vector<float> times;

		size_t index(int i)const{
			int numcontrolpoints=controlpoints.size();
			if(i>=0){
				return i%numcontrolpoints;
			}
			else{
				int negmod=(-i)%numcontrolpoints;
				int ret=numcontrolpoints-negmod;
				return ret%numcontrolpoints;
			}
		}
		
		void genvertices_helper(std::vector<vec2>& temp)const{
			if(controlpoints.size()==0 || speeds.size()==0 || controlpoints.size()!=speeds.size()) return;
			size_t left_time_index=0;
			size_t right_time_index=1;
			float t=times[0];
			float t_step_size=1.0/(float(ngon)/float(times.size()));
			
			while(right_time_index<=times.size()){
				vec2 vertex=Hermite_value(
								controlpoints[index(left_time_index)],
								speeds[index(left_time_index)],
								times[index(left_time_index)],
								controlpoints[index(right_time_index)],
								speeds[index(right_time_index)],
								times[index(right_time_index)],
								t
							);
				temp.push_back(vertex);

				t+=t_step_size;
				if(Float(t)>=Float(right_time_index)){
					left_time_index++;
					right_time_index++;
				};
			};
			
		};
	private:
		
		

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

		

		// void genvertices(std::vector<float>& temp)const override{
		// 	genvertices_helper(temp);
		// };
	

	public:
		Hermite_interpolation_curve(size_t ngon, const std::vector<vec2>& cps, const std::vector<vec2>& sps=std::vector<vec2>() ):
			ngon(ngon), controlpoints{cps}, speeds{sps} {
				for(float t=0;t<controlpoints.size();++t) times.push_back(t);
				genvertices_helper(vertices);

			};
		
};

class Catmull_Rom_spline: public Hermite_interpolation_curve{
	size_t numCtrPts=controlpoints.size();
	
	public:
		Catmull_Rom_spline(size_t ngon_,std::vector<vec2>& ctrpts_):Hermite_interpolation_curve(ngon_,ctrpts_){
			
			
			for(size_t i=0;i<numCtrPts;i<++i){
				
				vec2 a=(controlpoints[index(i+1)]-controlpoints[index(i)]);
				a=a/fabs(times[index(i+1)]-times[index(i)]);
				vec2 b=controlpoints[index(i)];
				vec2 b_=controlpoints[index(i-1)];
				b=b-b_;
				b=b/fabs(times[index(i)]-times[index(i-1)]);
				
				speeds.push_back(0.5*(a+b));

			}
			genvertices_helper(vertices);
		}
};


// http://mathworld.wolfram.com/LagrangeInterpolatingPolynomial.html (5)ös képletből
vec2 Lagrange_acceleration(float t1, float t2, float t3, vec2 r1, vec2 r2, vec2 r3){
	vec2 a=r1/(t1-t2)/(t1-t3);
	vec2 b=r2/(t2-t1)/(t2-t3);
	vec2 c=r3/(t3-t1)/(t3-t2);
	return 2.0*(a+b+c);

}

std::vector<vec2> points{vec2( -0.5, -0.58), vec2(0.16, 0.31), vec2(0.583333, -0.806667), vec2(0.78, -0.15)};
std::vector<vec2> speeds{vec2( -0.8f, -0.8f),vec2( -0.6f, 1.0f), vec2(0.8f, -0.2f)};
std::vector<vec2> polypoints{vec2(20, 10),
                 vec2(50, 125),
                 vec2(125, 90),
                 vec2(150, 10)};
std::vector<vec2> points2{			 
	vec2(-0.54,-0.53),

	vec2(0.506667,0.0766667),

	vec2(-0.37,0.496667),

	vec2(0.18,-0.593333),

	vec2(-0.523333,-0.833333),

	vec2(-0.673333,-0.0366666),

	vec2(-0.133333,0.393333),

	vec2(0.34,0.533333),

	vec2(0.76,0.303333),

	vec2(0.68,-0.52),

	vec2(0.363333,-0.843333)
};

Polygon poly(points2);
Polygon poly_interactive{};

Points refpoints{vec3(1.0f, 0.0f, 1.0f)};

//Hermite_interpolation_curve tri{points, speeds};
Triangle tri2{std::vector{ -0.8f, -0.8f, -0.6f, 1.0f, 0.8f, -0.2f }};
Catmull_Rom_spline crs{100,points};



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

	

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	int location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	//tri2.draw();
	//crs.draw();
	//poly.draw();
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
	case GLUT_LEFT_BUTTON:{
		printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
		if(buttonStat=="pressed"){
			poly_interactive.add(vec2(cX, cY));
			refpoints.add(vec2(cX, cY));
			
			std::cout<<"vec2("<<cX<<","<<cY<<std::endl;
			glClear(GL_COLOR_BUFFER_BIT);
			poly_interactive.draw();
			refpoints.draw();
		}
		
		glutSwapBuffers();

		break;

	} 
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
