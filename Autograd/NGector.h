#pragma once
#include <vector>
#include <cassert>
#include <memory>

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

	friend NGector<T> operator+ <>(const NGector<T>&, const NGector<T>&);
	friend NGector<T> operator- <>(const NGector<T>&, const NGector<T>&);
	friend NGector<T> operator* <>(const NGector<T>&, const NGector<T>&);
	friend NGector<T> operator/ <>(const NGector<T>&, const NGector<T>&);

	//Operators for operations with plain objects

	friend NGector<T> operator+ <>(const NGector<T>&, T);
	friend NGector<T> operator+ <>(T, const NGector<T>&);
	friend NGector<T> operator- <>(const NGector<T>&, T);
	friend NGector<T> operator- <>(T, const NGector<T>&);
	friend NGector<T> operator* <>(const NGector<T>&, T);
	friend NGector<T> operator* <>(T, const NGector<T>&);
	friend NGector<T> operator/ <>(const NGector<T>&, T);
	friend NGector<T> operator/ <>(T, const NGector<T>&);

	friend std::ostream& operator<< <>(std::ostream&, const NGector<T>&);


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
NGector<T> operator+(const NGector<T>& lhs, const NGector<T>& rhs)
{
	assert(lhs.size() == rhs.size() || lhs.size() == 1 || rhs.size() == 1);
	if (lhs.size() == 1)
		return lhs[0] + rhs;
	if (rhs.size() == 1)
		return lhs + rhs[0];
	NGector<T> res(vector<T>(lhs.size(), 0));
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] + rhs[i];
	return res;
}

template<typename T>
NGector<T> operator-(const NGector<T>& lhs, const NGector<T>& rhs)
{
	assert(lhs.size() == rhs.size() || lhs.size() == 1 || rhs.size() == 1);
	if (lhs.size() == 1)
		return lhs[0] - rhs;
	if (rhs.size() == 1)
		return lhs - rhs[0];	NGector<T> res(vector<T>(lhs.size(), 0));
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] - rhs[i];
	return res;
}

template<typename T>
NGector<T> operator*(const NGector<T>& lhs, const NGector<T>& rhs)
{
	assert(lhs.size() == rhs.size() || lhs.size() == 1 || rhs.size() == 1);
	if (lhs.size() == 1)
		return lhs[0] * rhs;
	if (rhs.size() == 1)
		return lhs * rhs[0];	
	NGector<T> res(vector<T>(lhs.size(), 0));
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] * rhs[i];
	return res;
}

template<typename T>
NGector<T> operator/(const NGector<T>& lhs, const NGector<T>& rhs)
{
	assert(lhs.size() == rhs.size() || lhs.size() == 1 || rhs.size() == 1);
	if (lhs.size() == 1)
		return lhs[0] / rhs;
	if (rhs.size() == 1)
		return lhs / rhs[0];	
	NGector<T> res(vector<T>(lhs.size(), 0));
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] / rhs[i];
	return res;
}


template<typename T>
NGector<T> operator+(const NGector<T>& lhs, T rhs)
{
	NGector<T> res(vector<T>(lhs.size(), 0));
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] + rhs;
	return res;
}

template<typename T>
NGector<T> operator+(T lhs, const NGector<T>& rhs)
{
	NGector<T> res(vector<T>(rhs.size(), 0));
	for (auto i = 0; i < rhs.size(); ++i)
		res[i] = lhs + rhs[i];
	return res;
}

template<typename T>
NGector<T> operator-(const NGector<T>& lhs, T rhs)
{
	NGector<T> res(vector<T>(lhs.size(), 0));
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] - rhs;
	return res;
}

template<typename T>
NGector<T> operator-(T lhs, const NGector<T>& rhs)
{
	NGector<T> res(vector<T>(rhs.size(), 0));
	for (auto i = 0; i < rhs.size(); ++i)
		res[i] = lhs - rhs[i];
	return res;
}


template<typename T>
NGector<T> operator*(const NGector<T>& lhs, T rhs)
{
	NGector<T> res(vector<T>(lhs.size(), 0));
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] * rhs;
	return res;
}

template<typename T>
NGector<T> operator*(T lhs, const NGector<T>& rhs)
{
	NGector<T> res(vector<T>(rhs.size(), 0));
	for (auto i = 0; i < rhs.size(); ++i)
		res[i] = lhs * rhs[i];
	return res;
}

template<typename T>
NGector<T> operator/(const NGector<T>& lhs, T rhs)
{
	NGector<T> res(vector<T>(lhs.size(), 0));
	for (auto i = 0; i < lhs.size(); ++i)
		res[i] = lhs[i] / rhs;
	return res;
}

template<typename T>
NGector<T> operator/(T lhs, const NGector<T>& rhs)
{
	NGector<T> res(vector<T>(rhs.size(), 0));
	for (auto i = 0; i < rhs.size(); ++i)
		res[i] = lhs / rhs[i];
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
