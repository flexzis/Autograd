#pragma once
#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <initializer_list>
#include "Operation.h"
#include "NGector.h"
#include "GradFunc.h"

using std::pair;
using std::vector;
using std::shared_ptr;

template <typename T>
class NGector;

/*
	Class representing Variable.
*/
template <typename T>
class Gector
{
	shared_ptr<GradFunc<T>>	depends_on{};
	shared_ptr<Gector<T>> grad{};

public:
	NGector<T> data;

	bool requires_grad = true;

	Gector() = default;

	Gector(const NGector<T>& data, bool requires_grad = true )
		: data{ data }
		, requires_grad{ requires_grad }
	{}

	Gector(const NGector<T>&& data, bool requires_grad = true)
		: data{ std::move(data) }
		, requires_grad{ requires_grad }
	{}

	Gector(size_t size, const T& fill_val, bool requires_grad = true)
		: data{ NGector<T>(size, fill_val) }
		, requires_grad{ requires_grad }
	{}

	Gector(std::initializer_list<T> list)
		: data{ list }
	{}

	Gector(const Gector<T>& other)
		: data{ other.data }
		, requires_grad{ other.requires_grad }
		, depends_on{ other.depends_on }
	{}

	Gector(Gector<T>&& other) noexcept
		: data{ std::move(other.data) }
		, requires_grad{ other.requires_grad }
		, depends_on{ std::move(other.depends_on) }
	{}

	Gector& operator=(const Gector<T>& other)
	{
		data = other.data;
		requires_grad = other.requires_grad;
		depends_on = other.depends_on;
		return *this;
	}

	Gector& operator=(Gector<T>&& other) noexcept
	{
		data = std::move(other.data);
		requires_grad = other.requires_grad;
		depends_on = std::move(other.depends_on);
		return *this;
	}

	bool operator==(const Gector<T>& other) const
	{
		auto size = this->size();
		if (size != other.size())
			return false;
		for (auto i = 0; i < size; ++i)
		{
			if (data[i] != other.data[i])
				return false;
		}
		return true;
	}

	void add_dependency(GradFunc<T>* dep)
	{
		depends_on.reset(dep);
	}

	Gector<T> get_grad()
	{
		return *grad;
	}

	Gector<T> sum()
	{
		return Gsum(*this);
	}

	void resize(size_t new_size)
	{
		data.resize(new_size);
	}

	void backward(const Gector<T>& in_grad = { 1. })
	{
		assert(requires_grad);

		if (!grad)
			grad.reset(new Gector<T>(in_grad.size(), T{}));

		for (auto i = 0; i < in_grad.size(); ++i)
			(*grad)[i] += in_grad[i];

		if (depends_on)
		{
			if (depends_on->is_binary())
			{
				if (depends_on->get_parent().requires_grad)
				{
					Gector<T> partial_deriv = depends_on->get_partial_deriv();
					Gector<T> par_lhs_grad(grad->data * partial_deriv.data);
					depends_on->get_parent().backward(par_lhs_grad);
				}
				if (depends_on->get_other_parent().requires_grad)
				{
					Gector<T> other_partial_deriv = depends_on->get_other_partial_deriv();
					Gector<T> par_rhs_grad(grad->data * other_partial_deriv.data);
					depends_on->get_other_parent().backward(par_rhs_grad);
				}
			}
			else
			{
				if (depends_on->get_parent().requires_grad)
				{
					auto partial_deriv = depends_on->get_partial_deriv();
					Gector<T> par_grad(grad->data * partial_deriv.data);
					depends_on->get_parent().backward(par_grad);
				}
			}
		}
	}

	auto size() const
	{
		return data.size();
	}

	auto begin()
	{
		return data.begin();
	}

	auto begin() const
	{
		return data.begin();
	}

	auto end()
	{
		return data.end();
	}

	auto end() const
	{
		return data.end();
	}

	T& operator [] (size_t i)
	{
		return data[i];
	}

	const T& operator [] (size_t i) const
	{
		return data[i];
	}
};

template<typename T>
Gector<T> operator+(Gector<T>& lhs, Gector<T>& rhs)
{
	return Gadd(lhs, rhs);
}

template<typename T>
Gector<T> operator-(Gector<T>& lhs, Gector<T>& rhs)
{
	auto neg = Gneg(rhs);
	return Gadd(lhs, neg);
}

template<typename T>
Gector<T> operator*(Gector<T>& lhs, Gector<T>& rhs)
{
	return Gmul(lhs, rhs);
}

template<typename T>
Gector<T> operator/(Gector<T>& lhs, Gector<T>& rhs)
{
	return Gdiv(lhs, rhs);
}

template<typename T>
Gector<T> operator-(Gector<T>& operand)
{
	return Gneg(operand);
}

template<typename T>
std::ostream& operator <<(std::ostream& os, const Gector<T>& t)
{
	os << '[';
	for (auto i = 0; i < t.size(); ++i)
	{
		os << t[i] << " ";
	}
	os << "\b]\n";
	return os;
}
