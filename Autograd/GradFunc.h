#pragma once
#include <cassert>
#include <vector>
#include <stdexcept>

using std::vector;

template <typename T>
class Gector;

template<typename T>
class GradFunc
{
public:
	virtual ~GradFunc() = 0;
	
	virtual bool is_binary() const = 0;

	virtual Gector<T>& get_parent() const = 0;
	virtual Gector<T>& get_other_parent() const = 0;

	virtual Gector<T> get_partial_deriv() const = 0;
	virtual Gector<T> get_other_partial_deriv() const = 0;
};

template<typename T>
GradFunc<T>::~GradFunc()
{ }


template<typename T>
class UnaryGradFunc: public GradFunc<T>
{
protected:
	Gector<T>& parent;

public:
	UnaryGradFunc() = default;

	UnaryGradFunc(Gector<T>& parent)
		: parent{ parent }
	{}

	virtual bool is_binary() const override
	{
		return false;
	}

	virtual Gector<T>& get_parent() const override
	{
		return parent;
	}

	virtual Gector<T>& get_other_parent() const override
	{
		throw std::logic_error("UnaryGradFunc is not supposed to have the second parent.");
	}

	virtual Gector<T> get_other_partial_deriv() const override
	{
		throw std::logic_error("UnaryGradFunc is not supposed to have the second partial derivative.");
	}
};


template<typename T>
class BinaryGradFunc: public GradFunc<T>
{
protected:
	Gector<T>& parent_lhs;
	Gector<T>& parent_rhs;
public:
	BinaryGradFunc() = default;

	BinaryGradFunc(Gector<T>& parent_lhs, Gector<T>& parent_rhs)
		: parent_lhs{ parent_lhs }
		, parent_rhs{ parent_rhs }
	{}

	virtual bool is_binary() const override
	{
		return true;
	}

	virtual Gector<T>& get_parent() const override
	{
		return parent_lhs;
	}

	virtual Gector<T>& get_other_parent() const override
	{
		return parent_rhs;
	}
};

template<typename T>
class GradSum: public UnaryGradFunc<T>
{
public:
	using UnaryGradFunc<T>::UnaryGradFunc;
	Gector<T> get_partial_deriv() const override
	{
		return Gector<T>(this->parent.size(), 1);
	}
};

template<typename T>
class GradNeg : public UnaryGradFunc<T>
{
public:
	using UnaryGradFunc<T>::UnaryGradFunc;
	Gector<T> get_partial_deriv() const override
	{
		return Gector<T>(this->parent.size(), -1);
	}
};


template<typename T>
class GradAdd : public BinaryGradFunc<T>
{
public:
	using BinaryGradFunc<T>::BinaryGradFunc;
	Gector<T> get_partial_deriv() const override
	{
		return Gector<T>(this->parent_lhs.size(), 1);
	}

	Gector<T> get_other_partial_deriv() const override
	{
		return Gector<T>(this->parent_rhs.size(), 1);
	}
};

template<typename T>
class GradMul : public BinaryGradFunc<T>
{
public:
	using BinaryGradFunc<T>::BinaryGradFunc;

	Gector<T> get_partial_deriv() const override
	{
		return this->get_other_parent().data;
	}

	Gector<T> get_other_partial_deriv() const override
	{
		return this->get_parent().data;
	}
};

template<typename T>
class GradDiv : public BinaryGradFunc<T>
{
public:
	using BinaryGradFunc<T>::BinaryGradFunc;

	Gector<T> get_partial_deriv() const override
	{
		return 1/this->get_other_parent().data;
	}

	Gector<T> get_other_partial_deriv() const override
	{
		return -this->get_parent().data/(this->get_other_parent().data * this->get_other_parent().data);
	}
};
