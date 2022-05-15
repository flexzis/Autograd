#pragma once
#include <memory>
#include <iostream>
#include "grad_func.h"

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

template<class T>
Gector<T> Gsum(Gector<T>& v)
{
	Gector<T> res(v.data.sum(), v.requires_grad);

	if (v.requires_grad)
		res.add_dependency(new GradSum<T>(v));

	return res;
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