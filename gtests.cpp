#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <functional>
#include "Gector.h"
#include "Operation.h"
#include "gtest/gtest.h"


using std::vector;
using std::cin;
using std::cout;

bool are_close(const Gector<double>& v1, const Gector<double>& v2, double eps = 1e-4)
{
	auto diff = abs(v1.data - v2.data);
	for (auto i = 0; i < diff.size(); ++i)
		if (diff[i] >= eps)
			return false;
	return true;
}

auto rng()
{
	std::random_device rnd_device;
	std::mt19937 mersenne_engine{ rnd_device() };
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

TEST(GectorOperationsTest, GectorSum)
{
	size_t size = 10;
	NGector<double> ones = std::vector<double>(size, 1.);
	NGector<double> twos = 2. * ones;

	Gector<double> g1(twos);
	auto res1 = g1.sum();
	EXPECT_EQ(res1, Gector<double>({ twos.sum() }));
	res1.backward(ones);
	EXPECT_EQ(g1.get_grad(), ones);

	Gector<double> g2(twos);
	auto res2 = g2.sum();
	EXPECT_EQ(res2, Gector<double>({ twos.sum() }));
	res2.backward({ 2. });
	EXPECT_EQ(g2.get_grad(), twos);
}

TEST(GectorOperationsTest, GectorMul)
{
	size_t size = 10;
	NGector<double> v1 = get_filled_random_vector(size);
	NGector<double> v2 = get_filled_random_vector(size);
	NGector<double> ones = std::vector<double>(size, 1.);

	NGector<double> product = v1 * v2;

	Gector<double> g1(v1);
	Gector<double> g2(v2);
	auto r1 = g1 * g2;
	EXPECT_EQ(r1, product);
	r1.backward(ones);
	EXPECT_EQ(g1.get_grad(), v2);
	EXPECT_EQ(g2.get_grad(), v1);
}

TEST(GectorOperationsTest, GectorMathFuncs)
{
	size_t size = 10;
	NGector<double> v1 = get_filled_random_vector(size);
	std::cout << v1;
	NGector<double> v2 = get_filled_random_vector(size);
	NGector<double> ones = std::vector<double>(size, 1.);
	{
		Gector<double> t1(v1);
		auto t2 = sin(t1);
		EXPECT_EQ(t2, Gector<double>(sin(v1)));
		t2.backward(ones);
		EXPECT_EQ(t1.get_grad(), Gector<double>(cos(v1)));
	}
	{
		Gector<double> t1(v1);
		auto t2 = log(t1);
		EXPECT_EQ(t2, Gector<double>(log(v1)));
		t2.backward(ones);
		double one{ 1 };
		EXPECT_EQ(t1.get_grad(), Gector<double>(one / v1));
	}
	{
		NGector<double> twos = 2. * ones;
		Gector<double> t1(twos);
		auto t2 = log(tan(exp(t1)));
		EXPECT_EQ(t2, Gector<double>(log(tan(exp(twos)))));
		t2.backward(ones);
		auto e_x = exp(twos);
		auto ans = Gector<double>(e_x / (cos(e_x) * sin(e_x)));
		EXPECT_TRUE(are_close(t1.get_grad(), ans));
	}
}

TEST(GectorOperationsTest, GectorComplex)
{
	size_t size = 10;
	NGector<double> v1 = get_filled_random_vector(size);
	std::cout << v1;
	NGector<double> v2 = get_filled_random_vector(size);
	NGector<double> ones = std::vector<double>(size, 1.);
	Gector<double> a(v1);
	Gector<double> b(v2);
	Gector<double> c(ones);
	Gector<double> d(v1 + ones);

	auto e = (a + b + c + d) * (a + b + c + d) * (a + b + c + d) + (a * b) + (a + c) * (c * d) + a * a;

	e.backward(ones);

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

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}