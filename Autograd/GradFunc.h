#pragma once
#include <cassert>
#include <vector>
#include <stdexcept>

// Suspicious: NGector and Gector both have overloaded operators. 
// What happens if we pass Gector to some of BinaryGradFunc and use its overloaded operators? 


using std::vector;

template <typename T>
class NGector;

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

	virtual NGector<T> get_partial_deriv() const = 0;
	virtual NGector<T> get_other_partial_deriv() const = 0;
};

template<typename T>
GradFunc<T>::~GradFunc()
{ }


template<typename T>
class UnaryGradFunc: public GradFunc<T>
{
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

	virtual NGector<T> get_other_partial_deriv() const override
	{
		throw std::logic_error("UnaryGradFunc is not supposed to have the second partial derivative.");
	}
};


template<typename T>
class BinaryGradFunc: public GradFunc<T>
{
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
	NGector<T> get_partial_deriv() const override
	{
		return NGector<T>(vector<T>(UnaryGradFunc<T>::get_parent().size(), 1));
	}
};

template<typename T>
class GradNeg : public UnaryGradFunc<T>
{
public:
	using UnaryGradFunc<T>::UnaryGradFunc;
	NGector<T> get_partial_deriv() const override
	{
		return NGector<T>(vector<T>(UnaryGradFunc<T>::get_parent().size(), -1));
	}
};


template<typename T>
class GradAdd : public BinaryGradFunc<T>
{
public:
	using BinaryGradFunc<T>::BinaryGradFunc;
	NGector<T> get_partial_deriv() const override
	{
		return NGector<T>(this->get_parent().size(), 1);
	}

	NGector<T> get_other_partial_deriv() const override
	{
		return NGector<T>(this->get_other_parent().size(), 1);
	}
};

template<typename T>
class GradMul : public BinaryGradFunc<T>
{
public:
	using BinaryGradFunc<T>::BinaryGradFunc;

	NGector<T> get_partial_deriv() const override
	{
		return this->get_other_parent().data;
	}

	NGector<T> get_other_partial_deriv() const override
	{
		return this->get_parent().data;
	}
};

template<typename T>
class GradDiv : public BinaryGradFunc<T>
{
public:
	using BinaryGradFunc<T>::BinaryGradFunc;

	NGector<T> get_partial_deriv() const override
	{
		return 1/this->get_other_parent().data;
	}

	NGector<T> get_other_partial_deriv() const override
	{
		return -this->get_parent().data/(this->get_other_parent().data * this->get_other_parent().data);
	}
};
