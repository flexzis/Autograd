#include "gector.h"
#include <fstream>
#include <string>

void save_data_to_file(const vector<double> data, const std::string& fname = "data.txt")
{
	std::ofstream outFile(fname);
	for (const auto& e : data)
		outFile << e << "\n";
}

void minimize_f(size_t iters = 10000)
{
	Gector<double> x{ -0.5 };
	Gector<double> y{ 1.2 };
	double alpha = 0.002;

	vector<double> x1_history{x[0]};
	vector<double> x2_history{y[0]};
	for (auto i = 0; i < iters; ++i)
	{
	 	auto ones = Gector<double>(x.size(), 1.);
		//auto f_x = x * x + log(ones + x * x) + sin(0.1 * x * x) * exp(-0.2 * x);
		Gector<double> f_x = (x + (-1.))*(x + (-1.)) + 100. * (y + (-x * x)) * (y + (-x * x));
		f_x.backward({ 1. });
		x = x - alpha * x.get_grad();
		y = y - alpha * y.get_grad();
			// std::cout << x;
		x1_history.push_back(x[0]);
		x2_history.push_back(y[0]);
		x.zero_grad();
		y.zero_grad();
		//if (i % 100 == 0)
		//	std::cout << "i = " << i << "  " << x[0] << "  " << y[0] << "\n";
	}
	save_data_to_file(x1_history, "xs.txt");
	save_data_to_file(x2_history, "ys.txt");
}

