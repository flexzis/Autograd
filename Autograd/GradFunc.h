#pragma once
#include <cassert>
#include <vector>

using std::vector;

template <typename T>
class Gector;


template<typename T>
class GradFunc
{
protected:
	Gector<T> inp;
public:
	GradFunc() = default;

	GradFunc(const Gector<T> & inp)
		: inp{ inp }
	{}

	virtual Gector<T> operator()(const Gector<T>& grad) = 0;
};

template<class T>
class GradSum: public GradFunc<T>
{
public:
	using GradFunc<T>::GradFunc;
	// grad should be 1-element Gector
	Gector<T> operator()(const Gector<T>& grad) override
	{
		assert(grad.size() == 1);
		return Gector<T>(vector<T>(this->inp.size(), grad[0]));
	}
};
