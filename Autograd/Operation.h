#pragma once
#include <memory>
#include <iostream>
#include "GradFunc.h"

template <typename T>
class Gector;

template<class T>
// sums elems in a Vector and returns it as a 1-elemented Vector
Gector<T> Gsum(Gector<T>& v)
{
	T sum {};
	for (auto& el : v)
		sum += el;

	auto res = Gector<T>({ sum }, v.requires_grad);

	if (v.requires_grad)
	{
		res.add_dependency(new GradSum<T>(v));
	}

	return res;
}