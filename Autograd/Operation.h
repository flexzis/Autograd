#pragma once
#include <memory>
#include "GradFunc.h"

template <typename T>
class Gector;

template<class T>
// sums elems in a Vector and returns it as a 1-elemented Vector
Gector<T> Gsum(const Gector<T>& v)
{
	T sum {};
	for (auto& el : v)
		sum += el;

	auto res = Gector<T>({ sum }, v.requires_grad);

	if (v.requires_grad)
	{
		GradFunc<T>* grad_fn_ptr = new GradSum<T>(v);
		res.add_dependency(std::make_unique<GradFunc<T>>(grad_fn_ptr));
	}
	
	return res;
}