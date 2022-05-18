//#include "tests.cpp"
//#include <chrono>
//#include <algorithm>
//#include <random>
//#include <functional>
//
//
//template<typename Real>
//void run_timed_test()
//{
//	size_t size = 1000000;
//	NGector<Real> v1 = get_filled_random_vector(size);
//	NGector<Real> v2 = get_filled_random_vector(size);
//	NGector<Real> ones = std::vector<Real>(size, 1.);
//	auto start = std::chrono::high_resolution_clock::now();
//	test_all<double>(v1, v2, ones);
//	auto end = std::chrono::high_resolution_clock::now();
//	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//}
//
//
//int main()
//{
//	run_timed_test<double>();
//}