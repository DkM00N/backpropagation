#include "backpropagation.h"
#include <random>

//init neural unit with n wi and b in normal distribution
void neuralunit::init(int n){
	std::default_random_engine gen;
	std::normal_distribution<float> dis(0,1);
	float *b=new float(1.0);
	i_w.push_back(std::shared_ptr<xw> (new xw{b,dis(gen)}));
    for(int i=0;i<n;i++)
		i_w.push_back(std::shared_ptr<xw> (new xw{nullptr,dis(gen)}));
};

//set neural forward conection with input X
void neuralunit::set(int n,vector<float>& x){
	for(int i=1;i<=n;i++)
		i_w[i]->x=&x[i-1];
};

//set neural forward conection with lower layer output
void neuralunit::set(int n,vector<std::shared_ptr<neuralunit>> &lay){
	for(int i=1;i<=n;i++)
	  i_w[i]->x=&lay[i-1]->o;
};

//
void neuralunit::forward(){
	z=0;
	for(auto m:i_w)
	 z+=*(m->x)*m->w;
	o=z>0?z:0;
};

//
void neuralunit::update(void (*optimizer)()){
	
};

//
neuralunit::~neuralunit(){
	
};

//init layer with m  empty neural units
layer::layer(int m)
	:lay(m),lower{nullptr},upper{nullptr}
{};

//set inputlayer
void layer::setinput(vector<float>& x){
	t=layertype::intput;
	int n=x.size();
	for(auto &it : lay){
		it->init(n);
		it->set(n,x);
	}
};

//set hidden layer
void layer::sethidden(layer *lw){
	if(debug && lower!=nullptr){
		cout<<"err!re setlower. in "<<__FILE__<<":"<<__LINE__<<endl;
		return;
	};
	t=layertype::hidden;
	lower=lw;
	int n=lower->lay.size();
	for(auto &it: lay){
		it->init(n);
		it->set(n,lower->lay);
	};
	if(debug && upper!=nullptr){
		cout<<"err!re setupper. in "<<__FILE__<<":"<<__LINE__<<endl;
		return;
	};
	lw->upper=this;
}

//set output layer
void layer::setoutput(vector<float*>& y1,layer *lw){
	if(debug && upper!=nullptr){
		cout<<"err!layer has upper setted. in "<<__FILE__<<":"<<__LINE__<<endl;
		return;
	};
	t=layertype::output;
	sethidden(lw);
	for(auto it: y1){
		y.push_back(it);
	}
};

//forward pass
void layer::forward(){
	if(debug && t!=layertype::intput){
		cout<<"err!Not input layer. in "<<__FILE__<<":"<<__LINE__<<endl;
		return;
	}
	layer* itr=this;
	while(itr){
		for(auto x:itr->lay)
		 x->forward();
		itr=itr->upper;
	}
};

//backward pass
void layer::backward(){

};

//update parameters
void layer::update(){

};

layer::~layer(){};

