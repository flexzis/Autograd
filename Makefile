.PHONY: build
build:
	cmake -S . -B build
	cmake --build build

.PHONY: test
test:
	cd build && ctest

.PHONY: run
run:
	build/Debug/Autograd.exe