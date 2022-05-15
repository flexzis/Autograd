#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>
#include <functional>
#include "gector.h"
#include "operation.h"

using std::vector;
using std::cin;
using std::cout;


auto rng()
{
	std::random_device rnd_device;
	// Specify the engine and distribution.
	std::mt19937 mersenne_engine{ rnd_device() };  // Generates random integers
	std::uniform_int_distribution<int> dist(1, 10000);

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
void test_complex()
{
	NGector<Real> ones = std::vector<double>(5000000, 1.);
	NGector<Real> v1 = get_filled_random_vector(5000000);
	NGector<Real> v2 = get_filled_random_vector(5000000);
	Gector<Real> a(v1);
	Gector<Real> b(v2);

	auto c = a * b + (a + b) + a * a;

	auto d = c + a + b; // 2(a+b)+a^2+a

	auto e = d * c + (d + c) - ((d * d) + (c * c)); // 

	d.pbackward(ones);

	assert(c.get_grad() == ones);
	assert(a.get_grad() == v2 + 2. * (ones + v1));
	assert(b.get_grad() == v1 + 2. * ones);

	std::cout << "Test complex passed!" << std::endl;
}

//template<typename Real>
//void test_long()
//{
//	NGector<Real> ones = std::vector<double>(10000, 1.);
//	NGector<Real> v1 = get_filled_random_vector(10000);
//	NGector<Real> v2 = get_filled_random_vector(10000);
//	Gector<Real> a(v1);
//	Gector<Real> b(v2);
//
//	auto c = (a + b) * a * b; // ba^2 + ab^2 
//
//	for (int i = 0; i < 1000; ++i)
//	{
//		c += a * b;
//	} // ba^2 + ab^2 + 1000ab
//
//	//c.backward(ones);
//
//	assert(c == v2 * v1 * v1 + v1 * v2 * v2 + 1000. * v1 * v2);
//
//	//assert(a.get_grad() == 2. * v1 * v2 + v2 * v2 + 1000. * v2);
//	//assert(b.get_grad() == v1 * v1 + 2. * v1 * v2 + 1000. * v1);
//
//	std::cout << "Test long passed!" << std::endl;
//
//	// TODO: Make += work!
//}

//void test_sum()
//{
//	Gector<double> g1{ 1. };
//	auto res1 = g1.sum();
//	assert(res1 == Gector<double>({ 1. }));
//	res1.backward({ 1. });
//	assert(g1.get_grad() == Gector<double>({ 1. }));
//
//	Gector<float> g2{ 5., 2., 3., 0. };
//	auto res2 = g2.sum();
//	assert(res2 == Gector<float>({ 10. }));
//	res2.backward({2.});
//	assert(g2.get_grad() == Gector<float>({ 2., 2., 2., 2.}));
//
//	std::cout << "Test sum passed!" << std::endl;
//}

//void test_add()
//{
//	{
//		Gector<double> g1{ 1. };
//		Gector<double> g2{ 2. };
//
//		auto r1 = g1 + g2;
//		assert(r1 == Gector<double>({ 3. }));
//		auto r2 = r1 + g2;
//		assert(r2 == Gector<double>({ 5. }));
//		auto r3 = r2 + g1;
//		assert(r3 == Gector<double>({ 6. }));
//		r3.backward({1.});
//		assert(r1.get_grad() == Gector<double>({ 1. }));
//		assert(r2.get_grad() == Gector<double>({ 1. }));
//		assert(g1.get_grad() == Gector<double>({ 2. }));
//		assert(g2.get_grad() == Gector<double>({ 2. }));
//	}
//	{
//		Gector<double> g1{ 1., 2. };
//		Gector<double> g2{ 2., 1. };
//		auto r1 = g1 + g2;
//		assert(r1 == Gector<double>({ 3., 3. }));
//		r1.backward(Gector<double> {-1., 1.});
//		assert(g1.get_grad() == Gector<double>({ -1., 1. }));
//		assert(g2.get_grad() == Gector<double>({ -1., 1. }));
//	}
//
//	std::cout << "Test add passed!" << std::endl;
//}

//void test_mul()
//{
//	Gector<double> g1{ 1, 2, 3 };
//	Gector<double> g2{ 2, 2, 2 };
//	auto r1 = g1 * g2;
//	assert(r1 == Gector<double>({ 2., 4., 6. }));
//	r1.backward({ -1, 1, 2 });
//	assert(g1.get_grad() == Gector<double>({-2., 2., 4.}));
//	assert(g2.get_grad() == Gector<double>({ -1., 2., 6. }));
//
//	std::cout << "Test mul passed!" << std::endl;
//}

//void test_complex()
//{
//	Gector a{ 2., -2. };
//	Gector b{ 5., 10. };
//
//	auto c = a * b + (a + b) + a * a;
//
//	auto d = c + a + b;
//
//	d.backward({ 1., 1. });
//
//	assert(c.get_grad() == Gector<double>({1., 1.}));
//	auto a_copy = Gector<double>(2. * a.data);
//	auto b_copy = Gector<double>(b.data);
//	auto unit = Gector<double>(std::vector<double>(a.size(), 1));
//	unit = unit + unit;
//	assert(a.get_grad() == b_copy + unit + a_copy);
//
//	std::cout << "Test complex passed!" << std::endl;
//}

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
	test_complex<Real>();

	//minimize();
}

