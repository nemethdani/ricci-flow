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
	
		static constexpr float epsilon = 1e-7f;
	
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
	Float D= segment2_point2.y-segment2_point1.y;
	Float E=segment2_point2.x - segment1_point2.x;
	Float F= segment2_point2.y - segment1_point2.y;
	Float f_ce_pa=F-C*E/A;
	Float d_cb_pa=D-C*B/A;
	Float t2=f_ce_pa/d_cb_pa;
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
		Primitive(GLenum m, vec3 color=vec3(0.0f, 1.0f, 1.0f) /*green*/):mode{m},color(color){};
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
class LineLoop: public Primitive{
	std::vector<vec2> vertices;
	public:
		LineLoop():Primitive(GL_LINE_LOOP){};
		LineLoop(std::vector<vec2> vertices, vec3 color=vec3(1,1,1)):Primitive(GL_LINE_LOOP, color), vertices(vertices){};
		void genvertices(std::vector<float>& targetvertices)const{
		for(auto v:vertices){
			targetvertices.push_back(v.x);
			targetvertices.push_back(v.y);
		}
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

size_t index(int i, size_t numcontrolpoints){
			
			if(i>=0){
				return i%numcontrolpoints;
			}
			else{
				int negmod=(-i)%numcontrolpoints;
				int ret=numcontrolpoints-negmod;
				return ret%numcontrolpoints;
			}
}

class Polygon: public Primitive{
	private:
		virtual size_t index(int i, size_t numcontrolpoints)const{
			
			return ::index(i, numcontrolpoints);
		}
		bool isDiagonal(const vec2& point1, const vec2& point2, const std::vector<vec2>& polivertices)const{
			if(segmentIntersect(point1, point2, polivertices.back(), polivertices.front())){
				return false;
			} 
			
			for(size_t i=0;i<polivertices.size()-1;++i){
				if(segmentIntersect(point1, point2, polivertices[i], polivertices[i+1])){
					return false;
				}
			}
			if(!doesContain((point1+point2)/2, polivertices)){
					return false;
			}
			
			return true;

		}
		void genvertices(std::vector<float>& temp)const override{
			if(vertices.size()>=3){
				std::vector<vec2> polyvertices=vertices;
				std::vector<vec2> triangles;
				earclipping(triangles, polyvertices);
				for(auto v: triangles){
					temp.push_back(v.x);
					temp.push_back(v.y);
				}
			}
			
		}

		

		void earclipping(std::vector<vec2>& triangles, std::vector<vec2>& polivertices)const{

			if(polivertices.size()<=3){
				
				for(auto v: polivertices){
					triangles.push_back(v);
				}
				
				return;
			}
			bool simplePolynom=false;
			
			for(size_t i=0; !simplePolynom && i<polivertices.size();++i){
				size_t left=index(i-1, polivertices.size());
				size_t right=index(i+1, polivertices.size());
				if(isDiagonal(polivertices[left], polivertices[right], polivertices)){
					triangles.push_back(polivertices[left]);
					triangles.push_back(polivertices[i]);
					triangles.push_back(polivertices[right]);
					polivertices.erase(polivertices.begin()+i);
					simplePolynom=true;
					
				}
			}
			if(!simplePolynom){
				std::cerr<<"Nem egyszerű polinom, tesszaláció megáll"<<std::endl;
				// for(auto v: polivertices){
				// 	std::cerr<<"vec2("<<v.x<<", "<<v.y<<"),"<<std::endl;
				// }
				return;
			}
				

			earclipping(triangles, polivertices); 
			return;


			
			
			
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
		bool doesContain(const vec2& point, const std::vector<vec2>& polivertices)const{
			bool contain=false;
			bool intersect;
			if(rayCastIntersect(point, polivertices.back(),polivertices.front()))
				contain=(!contain);
			for(size_t i=0;i<polivertices.size()-1;++i){
				

				intersect=rayCastIntersect(point, polivertices[i],polivertices[i+1]); //csak a maradék verticesből kell!
				if(intersect){
					contain=(!contain);
				}
					
			}
			return contain;

		}
		virtual void add(const vec2& point){
			vertices.push_back(point);
		}
		std::vector<vec2> getVertices()const{return vertices;}
		size_t getSize()const{return vertices.size();}
		const vec2& operator[](size_t i)const{
			return vertices[index(i, vertices.size())];
		}
		vec2& operator[](size_t i){
			return vertices[index(i, vertices.size())];
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
			
			float t_step_size=1/(float(ngon)/float(controlpoints.size()));
			Float t=0;
			size_t left_time_index=0;
			
			
			for(t; t<Float(controlpoints.size()) && temp.size()<ngon; t+=t_step_size){
					while(!(Float(left_time_index)<=t && t<Float(left_time_index+1))){
						++left_time_index;
					};
					size_t right_time_index=left_time_index+1;
					vec2 vertex=Hermite_value(
								controlpoints[left_time_index],
								speeds[left_time_index],
								left_time_index,
								controlpoints[index(right_time_index)],
								speeds[index(right_time_index)],
								right_time_index, 
								float(t)
							);
					temp.push_back(vertex);	
							
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
		Hermite_interpolation_curve(size_t ngon, const std::vector<vec2>& cps, const std::vector<vec2>& sps ):
			ngon(ngon), controlpoints{cps}, speeds{sps} {
				for(float t=0;t<controlpoints.size();++t) times.push_back(t);
				genvertices_helper(vertices);

			};
		
		
};

class Catmull_Rom_spline: public Hermite_interpolation_curve{
	size_t numCtrPts=controlpoints.size();

	private:
		void generateSpeeds(){
			numCtrPts=controlpoints.size();
			for(size_t i=0;i<numCtrPts;i<++i){
				
				vec2 a=(controlpoints[index(i+1)]-controlpoints[index(i)]);
				
				vec2 b=controlpoints[index(i)];
				vec2 b_=controlpoints[index(i-1)];
				b=b-b_;
				
				
				speeds.push_back(0.5*(a+b));

			}

		}
	
	public:
		Catmull_Rom_spline(size_t ngon_,const std::vector<vec2>& ctrpts_=std::vector<vec2>(), vec3 color=vec3(0,1,1))
			:Hermite_interpolation_curve(ngon_, ctrpts_, std::vector<vec2>()){
			
			
			generateSpeeds();
			genvertices_helper(vertices);
		}
		void addCtrPoint(vec2 ctrPoint){
			controlpoints.push_back(ctrPoint);
			speeds.clear();
			generateSpeeds();
			vertices.clear();
			genvertices_helper(vertices);
		}
};


// http://mathworld.wolfram.com/LagrangeInterpolatingPolynomial.html (5)ös képletből
vec2 Lagrange_acceleration(vec2 r1, vec2 r2, vec2 r3){

	float t1=-length(r1-r2);
	float t2=0;
	float t3=length(r1-r3);
	
	
	
	vec2 a=r1/(t1-t2)/(t1-t3);
	vec2 b=r2/(t2-t1)/(t2-t3);
	vec2 c=r3/(t3-t1)/(t3-t2);
	return 2.0*(a+b+c);

}

float constantAreaScalingFactor(const std::vector<vec2>& accelerations, const Polygon& polygon){
	float X, Y, Z;
	X= Y= Z=float();
	const std::vector<vec2> a=accelerations;
	const Polygon p=polygon;
	size_t polysize=polygon.getSize();
	//utolsó előtti ponting
	for(size_t i;i<polysize-1;++i){
		X+=a[i].x*a[i+1].y - a[i+1].x*a[i].y;
		Y+=a[i].x*p[i+1].y + a[i+1].y*p[i].x - a[i+1].x*p[i].y - a[i].y*p[i+1].x;
		//Z+=p[i].x*p[i+1].y - p[i+1].x*p[i].y;
	}
	//utolsó és első pont
	size_t i=polysize-1;
	X+=a[i].x*a[0].y - a[0].x*a[i].y;
	Y+=a[i].x*p[0].y + a[0].y*p[i].x - a[0].x*p[i].y - a[i].y*p[0].x;
	

	
	float scaling=-Y/X;
	
	return scaling;

}

float area(const Polygon& p){
	float sum=0;
	for(size_t i=0;i<p.getSize();++i){
		sum+=p[i].x*p[i+1].y - p[i+1].x*p[i].y;
	}
	return fabs(0.5*sum);
}

vec2 centroid(const Polygon& p){
	vec2 center(0,0);
	for(size_t i=0;i<p.getSize();++i){
		center=center+p[i];
		
	}
	return center/p.getSize();
}

vec2 constantAreaTranslationFactor(Polygon& polygon, std::vector<vec2>& accelerations, float deltaT=float(1.0/60), float s_y=1.0){
	float X, Y1, Y2;Y2=Y1=X=float();
	size_t polysize=polygon.getSize();
	std::vector<vec2>& a=accelerations;
	Polygon& p=polygon;
	//utolsó előtti ponting
	for(size_t i;i<polysize-1;++i){
		X+=a[i].x*a[i+1].y - a[i+1].x*a[i].y;
		Y1+=a[i].x*p[i+1].y - a[i+1].x*p[i].y ;
		Y2+=a[i+1].y*p[i].x - a[i].y*p[i+1].x;
		
	}
	//utolsó és első pont
	size_t i=polysize-1;
	X+=a[i].x*a[0].y - a[0].x*a[i].y;
	Y1+=a[i].x*p[0].y - a[0].x*p[i].y ;
	Y2+=a[0].y*p[i].x - a[i].y*p[0].x;

	float s_xt=(-s_y*deltaT*Y2)/(s_y*deltaT*X+Y1);
	return vec2(s_xt, s_y*deltaT);

}

float adjustmentScaler(float currentAcceleration, float referenceAcceleration, float tolerance, int remaining_time_ms){
	float adiff=currentAcceleration-referenceAcceleration;
	float divided=fabs(tolerance/adiff);
	float logarithm=fabs(logf(divided));
	float remtime=float(remaining_time_ms)/1000.0;
	float res=logarithm/remtime;
	return res;
}

void constantAreaScaling(Polygon& polygon, float targetArea, vec2 center, int timelimit_ms, int dt_ms){
	std::vector<vec2> accelerations;
	size_t polysize=polygon.getSize();
	for(size_t i=0;i<polysize;++i){
		vec2 a=Lagrange_acceleration(polygon[i-1],polygon[i], polygon[i+1]);
		accelerations.push_back(std::move(a));
	}
	if(accelerations.size()!=polysize){
		std::cerr<<"gyorsulások száma nem egyenlő a polygon pontjainak számával"<<std::endl;
	}
	
	//float scaling=constantAreaScalingFactor(accelerations, polygon);

	//vec2 centroid1=centroid(polygon);
	//Float area1=area(polygon);
	float pi=float(M_PI);
	Float targetAccelLength=sqrtf((pi)/float(targetArea));

	//vec2 trans_sclaing=constantAreaTranslationFactor(polygon, accelerations, float(1.0/60), 0.01);
	for(size_t i=0;i<polysize;++i){
		vec2 targetAccel=normalize(center-polygon[i])*float(targetAccelLength);
		float x_scaler=adjustmentScaler(accelerations[i].x, targetAccel.x, 0.0001, timelimit_ms);
		float y_scaler=adjustmentScaler(accelerations[i].y, targetAccel.y, 0.0001, timelimit_ms);
		x_scaler=x_scaler*float(dt_ms)*0.001;
		y_scaler=y_scaler*float(dt_ms)*0.001;
		std::cout<<"scaler: "<<x_scaler<<" "<<y_scaler<<std::endl;
		float translate_x=(accelerations[i].x)*dt_ms*10.846e-6 /* 0.0001 jo */  /* * trans_sclaing.x*/   ;
		float translate_y=(accelerations[i].y)*dt_ms*10.846e-6 /* * trans_sclaing.y*/   ;
		
		polygon[i]=polygon[i] + vec2(translate_x, translate_y);
	}

	vec2 centroid2=centroid(polygon);
	Float area2=area(polygon);
	if(Float(center.x)!=Float(centroid2.x) || Float(center.y)!=Float(centroid2.y))
		std::cerr<<"centroid mismatch: ("<<center.x<<" "<<center.y<<" "<<centroid2.x<<" "<<centroid2.y<<std::endl;
	if(targetArea!=area2)
		std::cerr<<"area error: "<<targetArea<<" != "<<area2<<std::endl;

}
LineLoop ll;
void ricciFlow(Polygon& polygon, int timelimit_ms=120000){
	float targetArea=area(polygon);
	vec2 center=centroid(polygon);
	int tref_ms=glutGet(GLUT_ELAPSED_TIME);
	std::cout<<"reftime: "<<tref_ms<<std::endl;
	int elasped_ms=0;
	int i=0;
	int newtime_ms=tref_ms;
	int oldtime_ms=tref_ms;
	
	while(true){
		std::cout<<i<<" "<<elasped_ms<< " dt: "<<newtime_ms-oldtime_ms<<" "<< tref_ms<<"\n"<<std::endl;
		constantAreaScaling(polygon, targetArea, center, timelimit_ms, newtime_ms-oldtime_ms);
		Points checkpoints{vec3(1,0,1),polygon.getVertices()};
		ll=LineLoop(polygon.getVertices());
		glClear(GL_COLOR_BUFFER_BIT);
		//checkpoints.draw();
		polygon.draw();
		ll.draw();
		glutSwapBuffers();

		oldtime_ms=newtime_ms;
		newtime_ms=glutGet(GLUT_ELAPSED_TIME);
		elasped_ms=newtime_ms-tref_ms;
		timelimit_ms-=oldtime_ms-newtime_ms;
		++i;
	}
}

std::vector<vec2> points{vec2( -0.5, -0.58), vec2(0.16, 0.31), vec2(0.583333, -0.806667), vec2(0.78, -0.15)};
std::vector<vec2> speeds{vec2( -0.8f, -0.8f),vec2( -0.6f, 1.0f), vec2(0.8f, -0.2f)};
std::vector<vec2> polypoints{vec2(20, 10),
                 vec2(50, 125),
                 vec2(125, 90),
                 vec2(150, 10)};

std::vector<vec2> crs_points{			 
	vec2(-0.656667,-0.39),

	vec2(-0.37,0.35),

	vec2(0.346667,0.34),

	vec2(0.0633334,-0.323333)

	

	
};
std::vector<vec2> debugpoints{			 
	vec2(0.0666668, 0.17),
	vec2(0.212604, -0.103047),
	vec2(0.253333, -0.166667),
	vec2(0.30914, -0.125235),
	vec2(0.368958, -0.0256252),
	vec2(0.43013, 0.0726301),
	vec2(0.49, 0.11),
	vec2(0.5825, 0.0566407),
	vec2(0.6975, -0.0510416),
	vec2(0.76625, -0.173203),
	vec2(0.72, -0.27),
	vec2(0.469792, -0.349323),
	vec2(0.0758336, -0.42625),
	vec2(-0.306875, -0.471719)
};

//Polygon poly(points2);
Polygon poly_interactive{};


std::vector<vec2> triangle={vec2(-0.4f, -0.4f), vec2(-0.3f, 0.5f), vec2(0.4f, -0.1f)};
//Hermite_interpolation_curve tri{points, speeds};
Triangle tri2{std::vector{ -0.8f, -0.8f, -0.6f, 1.0f, 0.8f, -0.2f }};
Catmull_Rom_spline crs{100,crs_points};
Catmull_Rom_spline interactive_crs{100, std::vector<vec2>(), };
//Points refpoints{vec3(1.0f, 0.0f, 1.0f), crs.getVertices()};
Points refpoints{vec3(1.0f, 0.0f, 1.0f)};

void cameraRight(){
	//everything left by 0.2
	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  -0.2, 0, 0, 1 };
	int location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
	glClear(GL_COLOR_BUFFER_BIT);
	crs.draw();
	refpoints.draw();
	glutSwapBuffers();
}


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

bool animation=false;
// Window has become invalid: Redraw
void onDisplay() {
	std::cout<<"onDisplay"<<std::endl;
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	int location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location


	glEnable(GL_POINT_SMOOTH);
	glPointSize(5);
	glLineWidth(4);

	//tri2.draw();
	//crs.draw();
	//poly.draw();
	//refpoints.draw();

	interactive_crs.draw();
	ll.draw();
	if(!animation) refpoints.draw();
	glutSwapBuffers(); // exchange buffers for double buffering

	
	
	

}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	else if (key == 'a') animation=true;
	else if (key == 'p') cameraRight();
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
			interactive_crs.addCtrPoint(vec2(cX, cY));
			refpoints.add(vec2(cX, cY));
			
			std::cout<<"vec2("<<cX<<","<<cY<<std::endl;
			ll=LineLoop(interactive_crs.getVertices());
			// glClear(GL_COLOR_BUFFER_BIT);
			// interactive_crs.draw();
			// ll.draw();
			// refpoints.draw();
			glutPostRedisplay();
		}
		
		//glutSwapBuffers();

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
