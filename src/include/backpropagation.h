#pragma once
#include <iostream>
#include <memory>
#include <vector>
#define debug true
using std::vector;
using std::cout;
using std::endl;

struct xw{
	float *x;
	float w;
};

class neuralunit{
 private:
	vector<std::shared_ptr<xw>> i_w;
	float z;
	float o;
	float bk;
 public:
	neuralunit(){};
	void init(int n);
	void set(int n,vector<float>& x);
	void set(int n,vector<std::shared_ptr<neuralunit>> &lay);
	void forward();
	void update(void (*optimizer)());
	~neuralunit();
};

enum class layertype{intput,hidden,output};
class layer{
 private:
	vector<std::shared_ptr<neuralunit>> lay;
	layer* lower;
	layer* upper;
	layertype t;
	vector<float*> y;
 public:
	layer(int m);
	void setinput(vector<float>& x);
	void sethidden(layer *lw);
	void setoutput(vector<float*>& y,layer *lw);
	void forward();
	void backward();
	void update();
	~layer();
};
