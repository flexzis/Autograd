#include "tests.cpp"
#include <chrono>
#include <algorithm>
#include <random>
#include <functional>


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