#pragma once
#include <cassert>
#include <vector>
#include <memory>
#include "GradFunc.h"

using std::vector;

template<typename T>
struct Dependency
{
	vector<T> data;
	std::unique_ptr<GradFunc<T>> grad_fn;
};