#pragma once
#include <cassert>
#include <vector>

using std::vector;

template <typename T>
class NGector;

template <typename T>
class Gector;


template<typename T>
class GradFunc
{
public:
	Gector<T>& gector;
	GradFunc() = default;

	GradFunc(Gector<T> & gector)
		: gector{ gector }
	{} 

	virtual Gector<T> operator()(const NGector<T>& grad) const = 0;
};

template<class T>
class GradSum: public GradFunc<T>
{
public:
	using GradFunc<T>::GradFunc;
	// grad should be 1-element Gector
	Gector<T> operator()(const NGector<T>& grad) const override
	{
		assert(grad.size() == 1);
		//std::cout << "gector size" << this->gector.size();
		return Gector<T>(vector<T>(this->gector.size(), 1));
	}
};
