#pragma once
#include <vector>
#include <cassert>
#include <memory>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

using std::vector;

/*
	std::vector wrapper with overloaded operators.
*/
template <typename T>
class NGector
{
public:
	vector<T> data;

	NGector() = default;

	NGector(const vector<T>& data)
		: data{ data }
	{}

	NGector(T data)
		: data{ data}
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

	//NGector<T>& operator=(NGector<T> other)
	//{
	//	data = other.data;
	//	return *this;
	//}

	NGector<T>& operator=(NGector<T>& other)
	{
		data = other.data;
		return *this;
	}

	NGector<T>& operator=(const NGector<T>& other)
	{
		data = other.data;
		return *this;
	}

	NGector<T>& operator=(NGector<T>&& other)
	{
		data = other.data;
		return *this;
	}

	NGector<T> sum()
	{
		T sum{};
		for (auto& el : data)
			sum += el;
		return { sum };
	}

	bool operator==(const NGector<T>& other) const
	{
		auto size = this->size();
		if (size != other.size())
			return false;
		for (auto i = 0; i < size; ++i)
		{
			if (data[i] != other[i])
				return false;
		}
		return true;
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


template<typename T>
NGector<T> operator+(const NGector<T>&lhs, const NGector<T>&rhs)
{
	assert(lhs.size() == rhs.size() || lhs.size() == 1 || rhs.size() == 1);
	if (lhs.size() == 1)
		return lhs[0] + rhs;
	if (rhs.size() == 1)
		return lhs + rhs[0];
	NGector<T> res(vector<T>(lhs.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < lhs.size(); ++i)
		#pragma omp critical
		res[i] = lhs[i] + rhs[i];
	return res;
}

template<typename T>
NGector<T> operator-(const NGector<T>&lhs, const NGector<T>&rhs)
{
	assert(lhs.size() == rhs.size() || lhs.size() == 1 || rhs.size() == 1);
	if (lhs.size() == 1)
		return lhs[0] - rhs;
	if (rhs.size() == 1)
		return lhs - rhs[0];
	NGector<T> res(vector<T>(lhs.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] - rhs[i];
	return res;
}

template<typename T>
NGector<T> operator*(const NGector<T>&lhs, const NGector<T>&rhs)
{
	assert(lhs.size() == rhs.size() || lhs.size() == 1 || rhs.size() == 1);
	if (lhs.size() == 1)
		return lhs[0] * rhs;
	if (rhs.size() == 1)
		return lhs * rhs[0];
	NGector<T> res(vector<T>(lhs.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < lhs.size(); ++i)
		#pragma omp critical
		res[i] = lhs[i] * rhs[i];
	return res;
}

template<typename T>
NGector<T> operator/(const NGector<T>&lhs, const NGector<T>&rhs)
{
	assert(lhs.size() == rhs.size() || lhs.size() == 1 || rhs.size() == 1);
	if (lhs.size() == 1)
		return lhs[0] / rhs;
	if (rhs.size() == 1)
		return lhs / rhs[0];
	NGector<T> res(vector<T>(lhs.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] / rhs[i];
	return res;
}

template<typename T>
NGector<T> operator-(const NGector<T>&operand)
{
	NGector<T> res(vector<T>(operand.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < operand.size(); ++i)
		res[i] = -operand[i];
	return res;
}


template<typename T>
NGector<T> operator+(const NGector<T>&lhs, T rhs)
{
	NGector<T> res(vector<T>(lhs.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] + rhs;
	return res;
}

template<typename T>
NGector<T> operator+(T lhs, const NGector<T>&rhs)
{
	NGector<T> res(vector<T>(rhs.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < rhs.size(); ++i)
		res[i] = lhs + rhs[i];
	return res;
}

template<typename T>
NGector<T> operator-(const NGector<T>&lhs, T rhs)
{
	NGector<T> res(vector<T>(lhs.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] - rhs;
	return res;
}

template<typename T>
NGector<T> operator-(T lhs, const NGector<T>&rhs)
{
	NGector<T> res(vector<T>(rhs.size(), 0));
	#pragma omp parallel for
	for (auto i = 0; i < rhs.size(); ++i)
		res[i] = lhs - rhs[i];
	return res;
}


template<typename T>
NGector<T> operator*(const NGector<T>&lhs, T rhs)
{
	NGector<T> res(vector<T>(lhs.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] * rhs;
	return res;
}

template<typename T>
NGector<T> operator*(T lhs, const NGector<T>&rhs)
{
	NGector<T> res(vector<T>(rhs.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < rhs.size(); ++i)
		res[i] = lhs * rhs[i];
	return res;
}

template<typename T>
NGector<T> operator/(const NGector<T>&lhs, T rhs)
{
	NGector<T> res(vector<T>(lhs.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] / rhs;
	return res;
}

template<typename T>
NGector<T> operator/(T lhs, const NGector<T>&rhs)
{
	NGector<T> res(vector<T>(rhs.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < rhs.size(); ++i)
		res[i] = lhs / rhs[i];
	return res;
}

template<typename T>
NGector<T> sin(const NGector<T>&v)
{
	NGector<T> res(vector<T>(v.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < v.size(); ++i)
		res[i] = std::sin(v[i]);
	return res;
}

template<typename T>
NGector<T> cos(const NGector<T>&v)
{
	NGector<T> res(vector<T>(v.size(), 0));

	#pragma omp parallel for
	for (auto i = 0; i < v.size(); ++i)
		res[i] = std::cos(v[i]);
	return res;
}

template<typename T>
NGector<T> tan(const NGector<T>&v)
{
	NGector<T> res(vector<T>(v.size(), 0));
	#pragma omp parallel for
	for (auto i = 0; i < v.size(); ++i)
		res[i] = std::tan(v[i]);
	return res;
}

template<typename T>
NGector<T> exp(const NGector<T>&v)
{
	NGector<T> res(vector<T>(v.size(), 0));
	#pragma omp parallel for
	for (auto i = 0; i < v.size(); ++i)
		res[i] = std::exp(v[i]);
	return res;
}

template<typename T>
NGector<T> log(const NGector<T>&v)
{
	NGector<T> res(vector<T>(v.size(), 0));
	#pragma omp parallel for
	for (auto i = 0; i < v.size(); ++i)
		res[i] = std::log(v[i]);
	return res;
}

template<typename T>
NGector<T> abs(const NGector<T>&v)
{
	NGector<T> res(vector<T>(v.size(), 0));
	#pragma omp parallel for
	for (auto i = 0; i < v.size(); ++i)
		res[i] = std::abs(v[i]);
	return res;
}

template<typename T>
std::ostream& operator <<(std::ostream& os, const NGector<T>& t)
{
	os << '[';
	for (auto i = 0; i < t.size(); ++i)
	{
		os << t[i] << " ";
	}
	os << "\b]\n";
	return os;
}