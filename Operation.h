#pragma once
#include <memory>
#include <iostream>
#include "GradFunc.h"


/*
	For Unary GFunctions:
		Input: v
		Output: GradFilled v'
	For Binary GFunctions:
		Input: v1, v2
		Output: GradFilled v3 
*/


template <typename T>
class NGector;

template <typename T>
class Gector;

template<typename T>
Gector<T> Gsum(Gector<T>& v)
{
	Gector<T> res(v.data.sum(), v.requires_grad);

	if (v.requires_grad)
		res.add_dependency(new GradSum<T>(v));

	return res;
}

template<typename T>
Gector<T>& GaddAssign(Gector<T>& v1, Gector<T>& v2)
{
	assert(v1.size() == v2.size());

	v1.data = v1.data + v2.data;

	if (v2.requires_grad)
		v1.add_dependency(new GradSum(v2));

	return v1;
}

template<typename T>
Gector<T> Gadd(Gector<T>& v1, Gector<T>& v2)
{
	assert(v1.size() == v2.size());
	
	auto requires_grad = v1.requires_grad || v2.requires_grad;
	
	Gector<T> res(v1.data + v2.data, requires_grad);

	if (requires_grad)
		res.add_dependency(new GradAdd<T>(v1, v2));

	return res;
}


template<typename T>
Gector<T> Gsub(Gector<T>& v1, Gector<T>& v2)
{
	assert(v1.size() == v2.size());

	auto requires_grad = v1.requires_grad || v2.requires_grad;

	Gector<T> res(v1.data - v2.data, requires_grad);

	if (requires_grad)
		res.add_dependency(new GradSub<T>(v1, v2));

	return res;
}


template<typename T>
 Gector<T> Gmul(Gector<T>& v1, Gector<T>& v2)
{
	assert(v1.size() == v2.size());

	auto requires_grad = v1.requires_grad || v2.requires_grad;
	Gector<T> res(v1.data * v2.data, requires_grad);

	if (requires_grad)
		res.add_dependency(new GradMul<T>(v1, v2));

	return res;
}


 template<typename T>
 Gector<T> Gdiv(Gector<T>& v1, Gector<T>& v2)
 {
	 assert(v1.size() == v2.size());

	 auto requires_grad = v1.requires_grad || v2.requires_grad;
	 Gector<T> res(v1.data / v2.data, requires_grad);

	 if (requires_grad)
		 res.add_dependency(new GradDiv<T>(v1, v2));

	 return res;
 }

 template<class T>
 Gector<T> Gneg(Gector<T>& v)
 {
	 Gector<T> res(-v.data, v.requires_grad);

	 if (v.requires_grad)
		 res.add_dependency(new GradNeg<T>(v));

	 return res;
 }


 template<typename T>
 Gector<T> Gsin(Gector<T>& v)
 {
	 Gector<T> res(sin(v.data), v.requires_grad);
	 if (v.requires_grad)
		 res.add_dependency(new GradSin<T>(v));
	 return res;
 }

 template<typename T>
 Gector<T> Gcos(Gector<T>& v)
 {
	 Gector<T> res(cos(v.data), v.requires_grad);
	 if (v.requires_grad)
		 res.add_dependency(new GradSin<T>(v));
	 return res;
 }

 template<typename T>
 Gector<T> Gtan(Gector<T>& v)
 {
	 Gector<T> res(tan(v.data), v.requires_grad);
	 if (v.requires_grad)
		 res.add_dependency(new GradTan<T>(v));
	 return res;
 }

 template<typename T>
 Gector<T> Gexp(Gector<T>& v)
 {
	 Gector<T> res(exp(v.data), v.requires_grad);
	 if (v.requires_grad)
		 res.add_dependency(new GradExp<T>(v));
	 return res;
 }

 template<typename T>
 Gector<T> Glog(Gector<T>& v)
 {
	 Gector<T> res(log(v.data), v.requires_grad);
	 if (v.requires_grad)
		 res.add_dependency(new GradLog<T>(v));
	 return res;
 }