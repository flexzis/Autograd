#include "gector.h"
#include <fstream>
#include <string>

void save_data_to_file(const vector<double> data, const std::string& fname = "data.txt")
{
	std::ofstream outFile(fname);
	for (const auto& e : data)
		outFile << e << "\n";
}

void minimize_f(size_t iters = 100)
{
	Gector<double> x{ 7., 6. };
	double alpha = 0.1;

	vector<double> x1_history{x[0]};
	vector<double> x2_history{x[1]};
	for (auto i = 0; i < iters; ++i)
	{
	 	auto ones = Gector<double>(x.size(), 1.);
		auto f_x = x * x + log(ones + x * x) + sin(0.1 * x) * exp(-0.2 * x);
		f_x.backward();
		x = x - alpha * x.get_grad();
			// std::cout << x;
		x1_history.push_back(x[0]);
		x2_history.push_back(x[1]);
		x.zero_grad();
	}
	save_data_to_file(x1_history, "xs.txt");
	save_data_to_file(x2_history, "ys.txt");
}

