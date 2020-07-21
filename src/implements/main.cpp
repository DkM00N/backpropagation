#include "backpropagation.h"
int main(){
	cout<<"running"<<endl;
	vector<float> x(2);
	vector<float*> y(2);
	layer l1(3);
	l1.setinput(x);
	layer l2(4);
	l2.sethidden(&l1);
	layer l3(2);
	l3.setoutput(y,&l2);
	return 0;
}