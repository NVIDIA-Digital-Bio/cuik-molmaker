# Catch2 Testing Framework for CUIK MolMaker

This directory contains C++ tests using the Catch2 testing framework. Catch2 is a modern, C++-native, test framework for unit-tests, TDD and BDD.

## Usage

### Building and Running Tests

To build and run the tests:

1. Ensure CMake option `CUIKMOLMAKER_BUILD_TESTS` is enabled:

```bash
cmake -DCUIKMOLMAKER_BUILD_TESTS=ON -B build
cmake --build build
```

2. Run the tests:

```bash
cd build
ctest
```

Or run the test executable directly:

```bash
./build/catch2_tests
```

### Running Specific Tests

Catch2 allows running specific tests using tags:

```bash
./build/catch2_tests [features]  # Only run tests with the 'features' tag
./build/catch2_tests "Molecule parsing*"  # Run tests matching the pattern
```

### Adding New Tests

To add new tests:

1. Create a new test file in this directory
2. Include Catch2 headers:

```cpp
#include <catch2/catch_test_macros.hpp>
```

3. Write your tests using the `TEST_CASE` macro:

```cpp
TEST_CASE("Description of the test", "[tag1][tag2]") {
    // Setup
    
    SECTION("Description of this section") {
        // Test code
        REQUIRE(condition);  // Must be true to continue
        CHECK(condition);    // Reported if false, but test continues
    }
    
    SECTION("Another section") {
        // More test code
    }
}
```

4. Add your new test file to the `catch2_tests` executable in the root CMakeLists.txt:

```cmake
add_executable(catch2_tests 
    tests/cpp/catch2/main.cpp
    tests/cpp/catch2/test_features.cpp
    tests/cpp/catch2/your_new_test.cpp  # Add your new file here
)
```

## Catch2 Documentation

For more details, see the [Catch2 documentation](https://github.com/catchorg/Catch2/blob/devel/docs/Readme.md). 