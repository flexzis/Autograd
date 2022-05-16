#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>
#include <functional>
#include <chrono>
#include "Gector.h"
#include "Operation.h"
#include "func_minimize.cpp"


using std::vector;
using std::cin;
using std::cout;

template<typename Real>
bool are_close(const Gector<Real>& v1, const Gector<Real>& v2, Real eps = 1e-4)
{
	assert(v1.size() == v2.size());
	auto diff = abs(v1.data - v2.data);
	for (auto i = 0; i < diff.size(); ++i)
		if (diff[i] >= eps)
			return false;
	return true;
}

auto rng()
{
	std::random_device rnd_device;
	// Specify the engine and distribution.
	std::mt19937 mersenne_engine{ rnd_device() };  // Generates random integers
	std::uniform_int_distribution<int> dist(1, 100);

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
void test_sum(NGector<double>& ones)
{
	NGector<Real> twos = 2. * ones;

	Gector<Real> g1(twos);
	auto res1 = g1.sum();
	assert(res1 == Gector<Real>({ twos.sum() }));
	res1.backward();
	assert(g1.get_grad() == ones);

	Gector<Real> g2(twos);
	auto res2 = g2.sum();
	assert(res2 == Gector<Real>({ twos.sum() }));
	res2.backward({ 2. });
	assert(g2.get_grad() == twos);

	std::cout << "Test sum passed!" << std::endl;
}


template<typename Real>
void test_add(NGector<Real>& v1, NGector<Real>& v2, NGector<Real>& ones)
{
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
void test_mul(NGector<Real>& v1, NGector<Real>& v2, NGector<Real>& ones)
{
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
void test_math_funcs(NGector<Real>& v1, NGector<Real>& ones)
{
	{
		Gector<Real> t1(v1);
		auto t2 = sin(t1);
		assert(t2 == Gector<Real>(sin(v1)));
		t2.backward();
		assert(t1.get_grad() == Gector<Real>(cos(v1)));

		std::cout << "Test sin passed!" << std::endl;
	}
	{
		Gector<Real> t1(v1);
		auto t2 = log(t1);
		assert(t2 == Gector<Real>(log(v1)));
		t2.backward();
		Real one{ 1 };
		assert(t1.get_grad() == Gector<Real>(one / v1));

		std::cout << "Test log passed!" << std::endl;
	}
	{
		NGector<Real> twos = 2. * ones;
		Gector<Real> t1(twos);
		auto t2 = log(tan(exp(t1)));
		assert(t2 == Gector<Real>(log(tan(exp(twos)))));
		t2.backward();
		auto e_x = exp(twos);
		auto ans = Gector<Real>(e_x / (cos(e_x) * sin(e_x)));
		assert(are_close(t1.get_grad(), ans));

		std::cout << "Test function of function passed!" << std::endl;
	}
}

template<typename Real>
void test_complex(NGector<Real>& v1, NGector<Real>& v2, NGector<Real>& ones)
{
	Gector<Real> a(v1);
	Gector<Real> b(v2);
	Gector<Real> c(ones);
	Gector<Real> d(v1 + ones);

	auto e = (a + b + c + d) * (a + b + c + d) * (a + b + c + d) + (a * b) + (a + c) * (c * d) + a * a;

	e.backward();

	// b + 2 a + c d + 3 (a + b + c + d)^2
	assert(a.get_grad() == b.data + 2. * a.data + c.data * d.data 
		+ 3. * (a.data + b.data + c.data + d.data) * (a.data + b.data + c.data + d.data));
	// a + 3 (a + b + c + d)^2
	assert(b.get_grad() == a.data 
		+ 3. * (a.data + b.data + c.data + d.data) * (a.data + b.data + c.data + d.data));
	// c d + (a + c) d + 3 (a + b + c + d)^2
	assert(c.get_grad() == c.data * d.data + (a.data + c.data) * d.data 
		+ 3. * (a.data + b.data + c.data + d.data) * (a.data + b.data + c.data + d.data));
	// c (a + c) + 3 (a + b + c + d)^2
	assert(d.get_grad() == c.data * (a.data + c.data) 
		+ 3. * (a.data + b.data + c.data + d.data) * (a.data + b.data + c.data + d.data));

	std::cout << "Test complex passed!" << std::endl;
}

template<class Real>
void minimize(bool verbose = false)
{
	Gector<Real> x{ 2., -2.};
	
	for (auto i = 0; i < 100; ++i)
	{
		auto sum_of_squares = (x * x).sum();
		sum_of_squares.backward();
		Real alpha{ 0.1 };
		x = x - alpha * x.get_grad();
		if (verbose)
			std::cout << i  << ": "  << x << "\n\n";
		x.zero_grad();
	}
	Gector<Real> ans(x.size(), 0.);	
	assert(are_close(x, ans, 1e-3));
	std::cout << "minimize test passed!" << "\n";
}

template<typename Real>
void test_all(NGector<Real>& v1, NGector<Real>& v2, NGector<Real>& ones)
{
	test_sum<Real>(ones);
	test_add<Real>(v1, v2, ones);
	test_mul<Real>(v1, v2, ones);
	test_math_funcs<Real>(v1, ones);
	test_complex<Real>(v1, v2, ones);

	minimize<Real>();
}


template<typename Real>
void run_timed_test()
{
	size_t size = 1000000;
	NGector<Real> v1 = get_filled_random_vector(size);
	NGector<Real> v2 = get_filled_random_vector(size);
	NGector<Real> ones = std::vector<Real>(size, 1.);
	auto start = std::chrono::high_resolution_clock::now();
	test_all<double>(v1, v2, ones);
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

int main()
{
	//run_timed_test();
	minimize_f(30);


}