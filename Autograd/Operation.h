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
class Gector;

template<class T>
Gector<T> Gsum(Gector<T>& v)
{
	Gector<T> res({T {}}, v.requires_grad);
	for (auto& el : v)
		res[0] += el;

	if (v.requires_grad)
		res.add_dependency(new GradSum<T>(v));

	return res;
}


template<typename T>
Gector<T> Gadd(Gector<T>& v1, Gector<T>& v2)
{
	assert(v1.size() == v2.size());
	
	auto requires_grad = v1.requires_grad || v2.requires_grad;
	Gector<T> res(vector<T>(v1.size()), requires_grad);

	for (auto i = 0; i < v1.size(); ++i)
		res[i] = v1[i] + v2[i];

	if (v1.requires_grad)
		res.add_dependency(new GradAdd<T>(v1, v2));

	return res;
}


template<typename T>
 Gector<T> Gmul(Gector<T>& v1, Gector<T>& v2)
{
	assert(v1.size() == v2.size());

	auto requires_grad = v1.requires_grad || v2.requires_grad;
	Gector<T> res(vector<T>(v1.size()), requires_grad);

	for (auto i = 0; i < v1.size(); ++i)
		res[i] = v1[i] * v2[i];

	if (v1.requires_grad)
		res.add_dependency(new GradMul<T>(v1, v2));

	return res;
}

 template<typename T>
 Gector<T> Gdiv(Gector<T>& v1, Gector<T>& v2)
 {
	 assert(v1.size() == v2.size());

	 auto requires_grad = v1.requires_grad || v2.requires_grad;
	 Gector<T> res(vector<T>(v1.size()), requires_grad);

	 for (auto i = 0; i < v1.size(); ++i)
		 res[i] = v1[i] / v2[i];

	 if (v1.requires_grad)
		 res.add_dependency(new GradDiv<T>(v1, v2));

	 return res;
 }

 template<class T>
 Gector<T> Gneg(Gector<T>& v)
 {
	 Gector<T> res(vector<T>(v.size()), v.requires_grad);
	 for (auto i = 0; i < res.size(); ++i)
		 res[i] = -v[i];

	 if (v.requires_grad)
		 res.add_dependency(new GradNeg<T>(v, v));

	 return res;
 }