#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>
#include <functional>
#include "Gector.h"
#include "Operation.h"

using std::vector;
using std::cin;
using std::cout;

template<typename Real>
bool are_close(const Gector<Real>& v1, const Gector<Real>& v2, Real eps = 1e-4)
{
	assert(v1.size() == v2.size());
	auto diff = abs(v1.data - v2.data);
	std::cout << diff;
	for (auto i = 0; i < diff.size(); ++i)
	{
		if (diff[i] >= eps)
			return false;
	}
	return true;
}

auto rng()
{
	std::random_device rnd_device;
	// Specify the engine and distribution.
	std::mt19937 mersenne_engine{ rnd_device() };  // Generates random integers
	std::uniform_int_distribution<int> dist(1, 2);

	auto gen = [&dist, &mersenne_engine]() {
		return double(dist(mersenne_engine));
	};

	return gen;
}


std::vector<double> get_filled_random_vector(size_t size)
{
	std::vector<double> result(size);
	auto generator = rng();

	std::generate(result.begin(), result.end(), generator);

	return result;
}

template<typename Real>
void test_sum()
{
	NGector<Real> ones = std::vector<Real>(10000, 1.);
	NGector<Real> twos = 2. * ones;

	Gector<Real> g1(twos);
	auto res1 = g1.sum();
	assert(res1 == Gector<Real>({ 20000. }));
	res1.backward();
	assert(g1.get_grad() == ones);

	Gector<Real> g2(twos);
	auto res2 = g2.sum();
	assert(res2 == Gector<Real>({ 20000. }));
	res2.backward({ 2. });
	assert(g2.get_grad() == twos);

	std::cout << "Test sum passed!" << std::endl;
}


template<typename Real>
void test_add()
{
	NGector<Real> v1 = get_filled_random_vector(10000);
	NGector<Real> v2 = get_filled_random_vector(10000);
	NGector<Real> ones = std::vector<double>(10000, 1.);
	NGector<Real> twos = ones + ones;
	Gector<Real> g1(v1);
	Gector<Real> g2{v2};

	auto r1 = g1 + g2;
	assert(r1 == v1 + v2);
	auto r2 = r1 + g2;
	assert(r2 == v1 + 2. * v2);
	auto r3 = r2 + g1;
	assert(r3 == 2. * (v1 + v2));
	r3.backward(ones);
	assert(r1.get_grad() == ones);
	assert(r2.get_grad() == ones);
	assert(g1.get_grad() == twos);
	assert(g2.get_grad() == twos);

	std::cout << "Test add passed!" << std::endl;
}


template<typename Real>
void test_mul()
{
	NGector<Real> ones = std::vector<double>(10000, 1.);
	NGector<Real> v1 = get_filled_random_vector(10000);
	NGector<Real> v2 = get_filled_random_vector(10000);
	NGector<Real> product = v1 * v2;

	Gector<Real> g1(v1);
	Gector<Real> g2(v2);
	auto r1 = g1 * g2;
	assert(r1 == product);
	r1.backward(ones);
	assert(g1.get_grad() == v2);
	assert(g2.get_grad() == v1);

	std::cout << "Test mul passed!" << std::endl;
}

template<typename Real>
void test_math_funcs()
{
	{
		Gector<Real> t1{ 10., 20. };
		auto t2 = sin(t1);
		assert(t2 == Gector<Real>({ sin(10.), sin(20.) }));
		t2.backward();
		assert(t1.get_grad() == Gector<Real>({ cos(10.), cos(20.) }));
	}
	{
		NGector<Real> d{ 10., 20. };
		Gector<Real> t1{ d };
		auto t2 = log(t1);
		assert(t2 == Gector<Real>(log(d)));
		t2.backward();
		Real one{ 1 };
		assert(t1.get_grad() == Gector<Real>(one / d));
	}
	{
		NGector<Real> d{ 10., 20. };
		Gector<Real> t1{ d };
		auto t2 = log(tan(exp(t1)));
		assert(t2 == Gector<Real>(log(tan(exp(d)))));
		t2.backward();
		auto e_x = exp(d);
		auto ans = Gector<Real>(e_x / (cos(e_x) * sin(e_x)));
		assert(are_close(t1.get_grad(), ans));
		//assert(t1.get_grad() == );
	}
}

template<typename Real>
void test_complex()
{
	size_t size = 1;
	NGector<Real> ones = std::vector<double>(size, 1.);
	NGector<Real> v1 = ones; // get_filled_random_vector(size);
	NGector<Real> v2 = ones; // get_filled_random_vector(size);
	Gector<Real> a(v1);
	Gector<Real> b(v2);

	auto c = a * b; // ab
	auto d = c + c; // 2ab
	// auto e = d + d * d; // 2ab + 4 a^2 b^2
	std::cout << (*c.temp_nodes[0])[0] << std::endl;
	d.backward(ones);
	std::cout << a.get_grad(); // 2

	//auto res = (1. + v2 + 2. * (1. + v2) * (v1 + v2 + v1 * v2));
	//std::cout << res;
	//assert(e.get_grad() == ones);

	//assert(a.get_grad() == res);
	//assert(b.get_grad() == v1 + 2. * ones);

	std::cout << "Test complex passed!" << std::endl;
}

void minimize()
{
	//Gector<double> x_{ 2., -2.};
	//
	//for (auto i = 0; i < 100; ++i)
	//{
	//	Gector<double> x = x_;
	//	std::cout << "x = " << x.data;
	//	auto square = x * x;
	//	std::cout << "x ** 2 = " << square.data;
	//	auto sum_of_squares = square.sum();
	//	std::cout << "sum(x**2) = " << sum_of_squares.data;
	//	sum_of_squares.backward();
	//	std::cout << "grad(x**2) = " << x.get_grad();
	//	auto alpha = 0.1;
	//	auto delta_x = Gector<double>(x.get_grad(), false) * alpha;
	//	Gector<double> y = x - delta_x;
	//	x = y;
	//	x_.data = x.data;
	//	std::cout << i  << ": "  << y.data << "\n\n";
	//}
}

template<typename Real>
void test_all()
{
	test_sum<Real>();
	test_add<Real>();
	test_mul<Real>();
	test_math_funcs<Real>();
	//test_complex<Real>();

	//minimize();
}

#include <chrono>
void run_timed_test()
{
	auto start = std::chrono::high_resolution_clock::now();
	test_all<double>();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}


int main()
{
	run_timed_test();
}