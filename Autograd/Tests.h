#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include "Gector.h"
#include "Operation.h"

using std::vector;
using std::cin;
using std::cout;

void test_sum()
{
	Gector<double> g1{ 1. };
	auto res1 = g1.sum();
	assert(res1 == Gector<double>{ 1. });
	res1.backward({ 1. });
	assert(g1.get_grad() == Gector<double>({ 1. }));

	Gector<float> g2{ 5., 2., 3., 0. };
	auto res2 = g2.sum();
	assert(res2 == Gector<float>({ 10. }));
	res2.backward({2.});
	assert(g2.get_grad() == Gector<float>({ 2., 2., 2., 2.}));
}

void test_add()
{
	{
		Gector<double> g1{ 1. };
		Gector<double> g2{ 2. };

		auto r1 = g1.add(g2);
		assert(r1 == Gector<double>({ 3. }));
		auto r2 = r1.add(g2);
		assert(r2 == Gector<double>({ 5. }));
		auto r3 = r2.add(g1);
		assert(r3 == Gector<double>({ 6. }));
		r3.backward({1.});
		assert(r1.get_grad() == Gector<double>({ 1. }));
		assert(r2.get_grad() == Gector<double>({ 1. }));
		assert(g1.get_grad() == Gector<double>({ 2. }));
		assert(g2.get_grad() == Gector<double>({ 2. }));
	}
	{
		Gector<double> g1{ 1., 2. };
		Gector<double> g2{ 2., 1. };
		auto r1 = g1.add(g2);
		assert(r1 == Gector<double>({ 3., 3. }));
		r1.backward(NGector<double> {-1., 1.});
		assert(g1.get_grad() == Gector<double>({ -1., 1. }));
		assert(g2.get_grad() == Gector<double>({ -1., 1. }));
	}
}

void test_mul()
{
	Gector<double> g1{ 1, 2, 3 };
	Gector<double> g2{ 2, 2, 2 };
	auto r1 = g1.mul(g2);
	assert(r1 == Gector<double>({ 2., 4., 6. }));
	r1.backward({ -1, 1, 2 });
	assert(g1.get_grad() == Gector<double>({-2., 2., 4.}));
	assert(g2.get_grad() == Gector<double>({ -1., 2., 6. }));
}

void minimize()
{
	Gector<double> x{ 2., -2.};
	
	for (auto i = 0; i < 10; ++i)
	{
		std::cout << "x = " << x;
		auto square = x.mul(x);
		std::cout << "x ** 2 = " << square;
		auto sum_of_squares = square.sum();
		std::cout << "sum(x**2) = " << sum_of_squares;
		sum_of_squares.backward();
		std::cout << "grad(x**2) = " << square.get_grad();
		auto alpha = Gector<double>{ 0.1, 0.1 };
		auto delta_x = square.get_grad().mul(alpha);
		auto y = x - delta_x;
		x = y;
		std::cout << i << x << "\n\n";
	}
}

void test_all()
{
	test_sum();
	test_add();
	test_mul();
	minimize();
}

