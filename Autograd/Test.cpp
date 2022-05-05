#include <iostream>
#include <vector>
#include "Gector.h"

using std::vector;
using std::cin;
using std::cout;

int main()
{
	Gector<double> g({ 5., 10., 3. });
	std::cout << g.requires_grad << std::endl;
	auto f = g.sum();
	f.backward();
	std::cout << Gector<double>(g.grad);
}