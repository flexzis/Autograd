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
	Gector<T>& parent;
	Gector<T>& uncle;
	GradFunc() = default;

	GradFunc(Gector<T>& parent, Gector<T>& uncle)
		: parent{ parent }
		, uncle{ uncle }
	{}

	virtual NGector<T> operator()(const NGector<T>& grad) const = 0;
};

template<typename T>
class GradSum: public GradFunc<T>
{
public:
	using GradFunc<T>::GradFunc;
	// grad should be 1-element Gector
	NGector<T> operator()(const NGector<T>& grad) const override
	{
		assert(grad.size() == 1);
		return NGector<T>(this->parent.size(), grad[0]);
	}
};

template<typename T>
class GradAdd : public GradFunc<T>
{
public:
	using GradFunc<T>::GradFunc;
	NGector<T> operator()(const NGector<T>& grad) const override
	{
		return grad;
	}
};

template<typename T>
class GradMul : public GradFunc<T>
{
public:
	using GradFunc<T>::GradFunc;
	NGector<T> operator()(const NGector<T>& grad) const override
	{
		NGector<T> res = grad;
		std::cout << "uncle = " << this->uncle;
		for (auto i = 0; i < res.size(); ++i)
			res[i] *= this->uncle[i];

		return res;
	}
};

template<typename T>
class GradNeg : public GradFunc<T>
{
public:
	using GradFunc<T>::GradFunc;
	NGector<T> operator()(const NGector<T>& grad) const override
	{
		auto res = grad;
		for (auto i = 0; i < grad.size(); ++i)
			res[i] = -res[i];
		return res;
	}
};
