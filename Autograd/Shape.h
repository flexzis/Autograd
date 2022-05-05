#pragma once
#include <vector>
#include <cassert>
// пока не используется

using std::pair;

class Shape
{
public:
	size_t h;
	size_t w;

	Shape(size_t h, size_t w) :
		h{ h }, w{ w }
	{}

	Shape(pair<size_t, size_t>& shape) :
		h{ shape.first }, w{ shape.second }
	{}

	size_t operator [](size_t ind)
	{
		assert(ind == 0 || ind == 1);
		if (ind == 0)
			return h;
		return w;
	}
};