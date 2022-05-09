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
using std::unique_ptr;


/*
	Class representing Variable.
*/
template <typename T>
class Gector
{
	unique_ptr<GradFunc<T>>	depends_on{};
	NGector<T> grad{};
public:
	NGector<T> data;

	bool requires_grad = true;

	Gector() = default;

	Gector(
		const vector<T>& data,
		bool requires_grad = true
	)
		: data{ data }
		, requires_grad{ requires_grad }
	{}

	Gector(
		const vector<T>&& data,
		bool requires_grad = true
	)
		: data{ std::move(data) }
		, requires_grad{ requires_grad }
	{}

	Gector(std::initializer_list<T> list)
		: data{ list }
	{}

	Gector(const Gector<T>& other)
		: data{ other.data }
		, requires_grad{ other.requires_grad }
	{}

	Gector(Gector<T>&& other) noexcept
		: data{ std::move(other.data) }
		, requires_grad{ other.requires_grad }
		, depends_on{ std::move(other.depends_on) }
	{}

	Gector& operator=(const Gector<T>& other)
	{
		this->data = other.data;
		requires_grad = other.requires_grad;
		return *this;
	}

	Gector& operator=(Gector<T>&& other) noexcept
	{
		this->data = std::move(other.data);
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

	NGector<T> get_grad()
	{
		return grad;
	}

	friend Gector<T> operator+ <>(Gector<T>&, Gector<T>&);
	friend Gector<T> operator- <>(Gector<T>&, Gector<T>&);
	friend Gector<T> operator* <>(Gector<T>&, Gector<T>&);
	friend Gector<T> operator/ <>(Gector<T>&, Gector<T>&);
	friend Gector<T> operator- <>(Gector<T>&);

	friend Gector<T> operator+ <>(Gector<T>&, NGector<T>&);
	friend Gector<T> operator+ <>(NGector<T>&, Gector<T>&);
	friend Gector<T> operator- <>(Gector<T>&, NGector<T>&);
	friend Gector<T> operator- <>(NGector<T>&, Gector<T>&);
	friend Gector<T> operator* <>(Gector<T>&, NGector<T>&);
	friend Gector<T> operator* <>(NGector<T>&, Gector<T>&);
	friend Gector<T> operator/ <>(Gector<T>&, NGector<T>&);
	friend Gector<T> operator/ <>(NGector<T>&, Gector<T>&);




	Gector<T> sum()
	{
		return Gsum(*this);
	}


	void backward(const NGector<T>& in_grad = { 1. })
	{
		assert(requires_grad);

		if (!grad.size())
			grad.resize(in_grad.size());

		for (auto i = 0; i < in_grad.size(); ++i)
			grad[i] += in_grad[i];

		if (depends_on)
		{
			if (depends_on->is_binary())
			{
				if (depends_on->get_parent().requires_grad)
				{
					std::cout << "In parent\n";
					NGector<T> partial_deriv = depends_on->get_partial_deriv();
					NGector<T> par_lhs_grad = grad * partial_deriv;
					depends_on->get_parent().backward(par_lhs_grad);
				}
				if (depends_on->get_other_parent().requires_grad)
				{
					std::cout << "In other parent\n";
					NGector<T> other_partial_deriv = depends_on->get_other_partial_deriv();
					NGector<T> par_rhs_grad = grad * other_partial_deriv;
					depends_on->get_other_parent().backward(par_rhs_grad);
				}
			}
			else
			{
				if (depends_on->get_parent().requires_grad)
				{
					std::cout << "In unary parent\n";
					auto par_grad = grad * depends_on->get_partial_deriv();
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
Gector<T> operator+(Gector<T>& lhs, NGector<T>& rhs)
{
	Gector<double> res(lhs);
	res.data = res.data + rhs;
	return res;
}

template<typename T>
Gector<T> operator+(NGector<T>& lhs, Gector<T>& rhs)
{
	Gector<double> res(rhs);
	res.data = res.data + lhs;
	return res;
}

template<typename T>
Gector<T> operator-(Gector<T>& lhs, NGector<T>& rhs)
{
	Gector<double> res(lhs);
	res.data = res.data - rhs;
	return res;
}

template<typename T>
Gector<T> operator-(NGector<T>& lhs, Gector<T>& rhs)
{
	Gector<double> res(rhs);
	res.data = lhs - res.data ;
	return res;
}

template<typename T>
Gector<T> operator*(Gector<T>& lhs, NGector<T>& rhs)
{
	Gector<double> res(lhs);
	res.data = res.data * rhs;
	return res;
}

template<typename T>
Gector<T> operator*(NGector<T>& lhs, Gector<T>& rhs)
{
	Gector<double> res(rhs);
	res.data = lhs * res.data;
	return res;
}

template<typename T>
Gector<T> operator/(Gector<T>& lhs, NGector<T>& rhs)
{
	Gector<double> res(lhs);
	res.data = res.data / rhs;
	return res;
}

template<typename T>
Gector<T> operator/(NGector<T>& lhs, Gector<T>& rhs)
{
	Gector<double> res(rhs);
	res.data = lhs / res.data;
	return res;
}
