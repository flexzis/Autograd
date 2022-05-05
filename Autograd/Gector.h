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

	NGector(vector<T>& data)
		: data{ data }
	{}

	NGector(const vector<T>& data)
		: data{ data }
	{}

	NGector(vector<T>&& data)
		: data{ data }
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
	vector<unique_ptr<GradFunc<T>>> depends_on {};
	
public:
	NGector<T> grad{};
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
			: NGector<T>{ data }
			, requires_grad{ requires_grad }
	{}

	Gector(const Gector<T>& other)
		: NGector<T>{ other.data }
		, requires_grad{ other.requires_grad }
	{}

	Gector(Gector<T>&& other)
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

	Gector& operator=(Gector<T>&& other)
	{
		this->data = std::move(other.data);
		requires_grad = other.requires_grad;
		std::cout << "dep" << other.depends_on.size();
		depends_on = std::move(other.depends_on);
		return *this;
	}

	void add_dependency(GradFunc<T>* dep)
	{
		depends_on.emplace_back(dep);
	}

	Gector<T> sum()
	{
		return Gsum(*this);
	}

	void backward(const Gector<T>& in_grad = Gector<T>({T {}}))
	{
		assert(requires_grad);

		if (!grad.size())
			grad.resize(in_grad.size());

		for (auto i = 0; i < in_grad.size(); ++i)
			grad[i] += in_grad[i];

		for (auto i = 0; i < depends_on.size(); ++i)
		{
			auto backward_grad = (*depends_on[i])(grad);
			depends_on[i]->gector.backward(backward_grad);
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