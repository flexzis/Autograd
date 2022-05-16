#pragma once
#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <initializer_list>
#include <omp.h>
#include "Operation.h"
#include "NGector.h"
#include "GradFunc.h"


using std::pair;
using std::vector;
using std::shared_ptr;

template <typename T>
class NGector;

/*
	Class representing Node.
*/
template <typename T>
class Gector
{
	shared_ptr<GradFunc<T>>	depends_on{};

public:
	Gector<T>& store_node(const Gector<T>& node)
	{
		temp_nodes.push_back(std::make_shared<Gector<T>>(node));
		return *temp_nodes.back();
	}
	vector<shared_ptr<Gector<T>>> temp_nodes{};
	shared_ptr<Gector<T>> grad{};
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
		, grad{other.grad}
	{}

	Gector(Gector<T>&& other) noexcept
		: data{ std::move(other.data) }
		, requires_grad{ other.requires_grad }
		, depends_on{ std::move(other.depends_on) }
		, grad{std::move(other.grad)}
	{}

	friend Gector<T>& operator+ <>(Gector<T>&, Gector<T>&);
	friend Gector<T>& operator+ <>(const T&, Gector<T>&);
	friend Gector<T>& operator+ <>(Gector<T>&, const T&);

	friend Gector<T>& operator- <>(Gector<T>&, Gector<T>&);
	friend Gector<T>& operator- <>(const T&, Gector<T>&);
	friend Gector<T>& operator- <>(Gector<T>&, const T&);
	friend Gector<T>& operator- <>(Gector<T>&);

	friend Gector<T>& operator* <>(Gector<T>&, Gector<T>&);
	friend Gector<T>& operator* <>(const T&, Gector<T>&);
	friend Gector<T>& operator* <>(Gector<T>&, const T&);

	friend Gector<T>& operator/ <>(Gector<T>&, Gector<T>&);
	friend Gector<T>& operator/ <>(const T&, Gector<T>&);
	friend Gector<T>& operator/ <>(Gector<T>&, const T&);

	Gector<T>& operator=(const Gector<T>& other)
	{
		data = other.data;
		requires_grad = other.requires_grad;
		depends_on = other.depends_on;
		return *this;
	}

	Gector<T>& operator=(Gector<T>&& other) noexcept
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

	void zero_grad()
	{
		grad = std::make_shared<Gector<T>>(zeros(data.size()));
		depends_on.reset();
	}

	void add_dependency(GradFunc<T>* dep)
	{
		depends_on.reset(dep);
	}

	Gector<T>& get_grad()
	{
		return *grad;
	}

	Gector<T>& sum()
	{
		auto new_node = Gsum(*this);
		temp_nodes.push_back(std::make_shared<Gector<T>>(new_node));
		return *temp_nodes.back();
	}

	void resize(size_t new_size)
	{
		data.resize(new_size);
	}

	Gector<T> zeros(size_t size) const
	{
		return Gector<T>(size, 0);
	}

	Gector<T> ones(size_t size) const
	{
		return Gector<T>(size, 1);
	}

	void backward(const Gector<T>& in_grad = { 1. })
	{
		assert(requires_grad);
		if (!grad)
		{
			grad = std::make_shared<Gector<T>>(zeros(in_grad.size()));
		}

		//std::cout << grad->size() << "  " << in_grad.size() << "\n";
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
				// x = A @ w + w0, data = NGector<Ngector>
				if (depends_on->get_parent().requires_grad)
				{
					auto partial_deriv = depends_on->get_partial_deriv();
					Gector<T> par_grad(grad->data * partial_deriv.data);
					depends_on->get_parent().backward(par_grad);
				}
			}
		}
	}

	void pbackward(const Gector<T>& in_grad = {})
	{
		assert(requires_grad);

		if (!in_grad.size())
			in_grad = ones(data.size());

		if (!grad)
			grad.reset(new Gector<T>(in_grad.size(), T{}));

		for (auto i = 0; i < in_grad.size(); ++i)
			(*grad)[i] += in_grad[i];

		if (depends_on)
		{
			if (depends_on->is_binary())
			{
				if (depends_on->get_parent().requires_grad && depends_on->get_other_parent().requires_grad)
				{
					#pragma omp parallel
					{
						#pragma omp sections
						{
							#pragma omp section
							{
								Gector<T> partial_deriv = depends_on->get_partial_deriv();
								Gector<T> par_lhs_grad = grad->data * partial_deriv.data;
								depends_on->get_parent().backward(par_lhs_grad);
							}
							#pragma omp section
							{
								Gector<T> other_partial_deriv = depends_on->get_other_partial_deriv();
								Gector<T> par_rhs_grad = grad->data * other_partial_deriv.data;
								depends_on->get_other_parent().backward(par_rhs_grad);
							}
						}
					}
				}
				else if (depends_on->get_parent().requires_grad)
				{
					Gector<T> partial_deriv = depends_on->get_partial_deriv();
					Gector<T> par_lhs_grad(grad->data * partial_deriv.data);
					depends_on->get_parent().backward(par_lhs_grad);
				}
				else if (depends_on->get_other_parent().requires_grad)
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
Gector<T>& operator+(Gector<T>& lhs, Gector<T>& rhs)
{
	return lhs.store_node(Gadd(lhs, rhs));
}

template<typename T>
Gector<T>& operator+(Gector<T>& lhs, const T& rhs)
{
	Gector<T>& rhs_vectorized = lhs.store_node(Gector<T>(lhs.size(), rhs));
	return lhs + rhs_vectorized;
}

template<typename T>
Gector<T>& operator+(const T& lhs, Gector<T>& rhs)
{
	Gector<T>& lhs_vectorized = lhs.store_node(Gector<T>(rhs.size(), lhs));
	return lhs_vectorized + rhs;
}

template<typename T>
Gector<T>& operator-(Gector<T>& v)
{
	return v.store_node(Gneg(v));
}

template<typename T>
Gector<T>& operator-(Gector<T>& lhs, Gector<T>& rhs)
{
	Gector<T> neg = Gneg(rhs);
	lhs.store_node(neg);
	return lhs.store_node(Gadd(lhs, neg));
}

template<typename T>
Gector<T>& operator-(Gector<T>& lhs, const T& rhs)
{
	Gector<T>& rhs_vectorized_neg = lhs.store_node(Gector<T>(lhs.size(), -rhs));
	return lhs + rhs_vectorized_neg;
}

template<typename T>
Gector<T>& operator-(const T& lhs, Gector<T>& rhs)
{
	Gector<T>& lhs_vectorized = lhs.store_node(Gector<T>(lhs.size(), lhs));
	return lhs_vectorized - rhs;
}

template<typename T>
Gector<T>& operator*(Gector<T>& lhs, Gector<T>& rhs)
{
	return lhs.store_node(Gmul(lhs, rhs));
}

template<typename T>
Gector<T>& operator*(Gector<T>& lhs, const T& rhs)
{
	Gector<T>& rhs_vectorized = lhs.store_node(Gector<T>(lhs.size(), rhs));
	return lhs * rhs_vectorized;
}

template<typename T>
Gector<T>& operator*(const T& lhs, Gector<T>& rhs)
{
	Gector<T>& lhs_vectorized = rhs.store_node(Gector<T>(rhs.size(), lhs));
	return lhs_vectorized * rhs;
}

template<typename T>
Gector<T>& operator/(Gector<T>& lhs, Gector<T>& rhs)
{
	return lhs.store_node(Gdiv(lhs, rhs));
}

template<typename T>
Gector<T>& operator/(Gector<T>& lhs, const T& rhs)
{
	Gector<T>& rhs_vectorized = lhs.store_node(Gector<T>(lhs.size(), rhs));
	return lhs / rhs_vectorized;
}

template<typename T>
Gector<T>& operator/(const T& lhs, Gector<T>& rhs)
{
	Gector<T>& lhs_vectorized = lhs.store_node(Gector<T>(rhs.size(), lhs));
	return lhs_vectorized / rhs;
}

template<typename T>
Gector<T>& sin(Gector<T>& v)
{
	return v.store_node(Gsin(v));
}

template<typename T>
Gector<T>& cos(Gector<T>& v)
{
	return v.store_node(Gcos(v));
}

template<typename T>
Gector<T>& tan(Gector<T>& v)
{
	return v.store_node(Gtan(v));
}

template<typename T>
Gector<T>& exp(Gector<T>& v)
{
	return v.store_node(Gexp(v));
}

template<typename T>
Gector<T>& log(Gector<T>& v)
{
	return v.store_node(Glog(v));
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
