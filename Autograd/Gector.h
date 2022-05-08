#pragma once
#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include "Dependency.h"
#include "Operation.h"

using std::pair;
using std::vector;
using std::unique_ptr;

template <typename T>
class NGector
{
public:
	vector<T> data;

	NGector() = default;

	NGector(const vector<T>& data)
		: data{ data }
	{}

	NGector(vector<T>&& data)
		: data{ data }
	{}

	NGector(std::initializer_list<T> list)
		: data{ list }
	{}

	NGector(const NGector<T>& other)
		: data{other.data}
	{}

	NGector(NGector<T>&& other) noexcept
		: data{ std::move(other.data) }
	{}

	NGector(size_t size, const T& fill_val)
		: data{ vector<T>(size, fill_val) }
	{}

	NGector& operator=(NGector& other)
	{
		data = other.data;
	}

	NGector& operator=(NGector&& other)
	{
		data = other.data;
	}

	const vector<T>& get_data() const
	{
		return data;
	}

	T& operator [] (size_t i)
	{
		return data[i];
	}

	NGector<T> operator*(const NGector& rhs)
	{
		assert(data.size() == rhs.size() || data.size() == 1 || rhs.size() == 1);
		NGector<T> res(std::max(rhs.size(), data.size()), T{});
		if (rhs.size() == 1)
		{
			for (auto i = 0; i < data.size(); ++i)
				res[i] = data[i] * rhs[0];
		}
		else if (data.size() == 1)
		{
			for (auto i = 0; i < rhs.size(); ++i)
				res[i] = data[0] * rhs[i];
		}
		else
		{
			for (auto i = 0; i < rhs.size(); ++i)
				res[i] = data[i] * res[i];
		}
		return res;
		
	}
	const T& operator [] (size_t i) const
	{
		return data[i];
	}

	void resize(size_t new_size)
	{
		data.resize(new_size);
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
};

template <typename T>
class Gector : public NGector<T>
{
	unique_ptr<GradFunc<T>>	depends_on {};
	NGector<T> grad{};
public:
	
	bool requires_grad = true;

	Gector() = default;

	Gector(
		const vector<T>& data,
		bool requires_grad = true
	)
		: NGector<T>{ data }
		, requires_grad{ requires_grad }
	{}

	Gector(
		const vector<T>&& data,
		bool requires_grad = true
	)
			: NGector<T>{ std::move(data) }
			, requires_grad{ requires_grad }
	{}

	Gector(std::initializer_list<T> list)
		: NGector<T>{list}
	{}

	Gector(const Gector<T>& other)
		: NGector<T>{ other.data }
		, requires_grad{ other.requires_grad }
	{}

	Gector(Gector<T>&& other) noexcept
		: NGector<T>{ std::move(other.data) }
		, requires_grad{ other.requires_grad }
		, depends_on { std::move(other.depends_on) }
	{}	

	Gector(const NGector<T>& other)
		: NGector<T>{ other.data }
	{}

	Gector(NGector<T>&& other)
		: NGector<T>{ std::move(other.data) }
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
			if ((*this)[i] != other[i])
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
		return grad;
	}

	Gector<T> sum()
	{
		return Gsum(*this);
	}

	Gector<T> add(Gector<T>& other)
	{
		// maybe implement for different sizes
		return Gadd(*this, other);
	}

	Gector<T> mul(Gector<T>& other)
	{
		return Gmul(*this, other);
	}
	
	Gector<T> operator-()
	{
		return Gneg(*this);
	}

	Gector<T> operator-(Gector<T>& other)
	{
		auto neg_other = -other;
		return add(neg_other);
	}

	void backward(const NGector<T>& in_grad = {1.})
	{
		assert(requires_grad);

		if (!grad.size())
			grad.resize(in_grad.size());

		for (auto i = 0; i < in_grad.size(); ++i)
			grad[i] += in_grad[i];

		// 1 argument operator
		if (depends_on)
			if (depends_on->parent_lhs == depends_on->parent_rhs)
			{
				if (depends_on->parent_lhs.requires_grad)
				{
					auto par_grad = grad * depends_on->lhs_partial_deriv();
					depends_on->parent_lhs.backward(par_grad);
				}
			}
			else
			{
				if (depends_on->parent_lhs.requires_grad)
				{
					auto par_lhs_grad = grad * depends_on->lhs_partial_deriv();
					depends_on->parent_lhs.backward(par_lhs_grad);
				}
				if (depends_on->parent_rhs.requires_grad)
				{
					auto par_rhs_grad = grad * depends_on->rhs_partial_deriv();
					depends_on->parent_rhs.backward(par_rhs_grad);
				}
			}
	}
};

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