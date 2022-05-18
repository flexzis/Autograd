cmake_minimum_required (VERSION 3.9)

project(Autograd
		VERSION 0.2
		DESCRIPTION "Automatic gradient count for R^n->R^m analytic fuctions"
		LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
option(CMAKE_USE_WIN32_THREADS_INIT "using WIN32 threads" ON)

include(CTest)
enable_testing()

find_package(OpenMP REQUIRED)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  gtests gtests.cpp
)

target_link_libraries(
  gtests gtest_main
)

include(GoogleTest)
gtest_discover_tests(gtests)
gtest_discover_tests(gtests PROPERTIES TIMEOUT 600)
