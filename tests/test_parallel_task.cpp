
#include <SEvoBench/utility/parallel_task.hpp>

#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() try {
  for (auto repetition = 0; repetition < 250; ++repetition) {
    auto pool = sevobench::parallel_task{4};
    auto futures = std::vector<std::future<int>>{};
    futures.reserve(24);
    for (auto value = 0; value < 24; ++value)
      futures.push_back(pool.submit([value] { return value * value; }));
    for (auto value = 0; value < 24; ++value) {
      if (futures[static_cast<std::size_t>(value)].get() != value * value)
        throw std::runtime_error{"parallel task returned a wrong value"};
    }
  }
  std::cout << "parallel task lifecycle: pass\n";
  return EXIT_SUCCESS;
} catch (std::exception const &error) {
  std::cerr << "parallel task lifecycle: " << error.what() << '\n';
  return EXIT_FAILURE;
}
