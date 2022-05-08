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
	Gector<T>& parent_lhs;
	Gector<T>& parent_rhs;
	GradFunc() = default;

	GradFunc(Gector<T>& parent_lhs, Gector<T>& parent_rhs)
		: parent_lhs{ parent_lhs }
		, parent_rhs{ parent_rhs }
	{}

	virtual NGector<T> lhs_partial_deriv() const = 0;
	virtual NGector<T> rhs_partial_deriv() const = 0;
};

template<typename T>
class GradSum: public GradFunc<T>
{
public:
	using GradFunc<T>::GradFunc;
	NGector<T> lhs_partial_deriv() const override
	{
		return NGector<T>(this->parent_lhs.size(), 1);
	}

	NGector<T> rhs_partial_deriv() const override
	{
		return lhs_partial_deriv();
	}
};

template<typename T>
class GradAdd : public GradFunc<T>
{
public:
	using GradFunc<T>::GradFunc;
	NGector<T> lhs_partial_deriv() const override
	{
		return NGector<T>(this->parent_lhs.size(), 1);
	}

	NGector<T> rhs_partial_deriv() const override
	{
		return lhs_partial_deriv();
	}
};

template<typename T>
class GradMul : public GradFunc<T>
{
public:
	using GradFunc<T>::GradFunc;

	NGector<T> lhs_partial_deriv() const override
	{
		auto res = this->parent_rhs;
		return res;
	}

	NGector<T> rhs_partial_deriv() const override
	{
		auto res = this->parent_lhs;
		return res;
	}

	NGector<T> operator()(const NGector<T>& grad) const override
	{
		auto res = grad;
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
