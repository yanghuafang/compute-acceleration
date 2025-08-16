// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_TESTS_SUPPORT_TEST_HARNESS_H_
#define ACCEL_TESTS_SUPPORT_TEST_HARNESS_H_

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Self-contained test Registry — no external framework, no network.
//
// The project must stay buildable on a machine with nothing but a compiler and
// CMake, so vendoring or fetching GoogleTest is not an option. Roughly a
// hundred lines buys registration, isolation of failures, and a useful
// summary, which is all these suites need.
//
// Tests exist mainly so the sanitizers have something small and fast to run:
// the benchmark shapes take minutes natively and far longer under ASan,
// whereas these cases finish in milliseconds. See docs/Sanitizers.md.

namespace accel::test {

using TestBody = void (*)();

struct TestCase {
  const char* name;
  TestBody body;
};

// Function-local static, so registration order across translation units is
// irrelevant and there is no static-initialisation-order fiasco to trip over.
inline std::vector<TestCase>& Registry() {
  static std::vector<TestCase> cases;
  return cases;
}

// Registers one case at static-initialisation time via its constructor.
struct Registrar {
  Registrar(const char* name, TestBody body) {
    Registry().push_back(TestCase{name, body});
  }
};

// Thrown by the CHECK_* macros; carries an already-formatted location.
class AssertionFailure : public std::exception {
 public:
  explicit AssertionFailure(std::string message)
      : message_(std::move(message)) {}
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
};

namespace detail {

[[noreturn]] inline void Fail(const char* file, int line,
                              const std::string& detail) {
  std::ostringstream out;
  out << file << ':' << line << ": " << detail;
  throw AssertionFailure(out.str());
}

// Absolute-or-relative closeness, so the same macro works for values near Zero
// and for accumulated GEMM outputs several orders of magnitude larger.
inline bool CloseEnough(double lhs, double rhs, double tolerance) {
  if (std::isnan(lhs) || std::isnan(rhs)) {
    return false;
  }
  const double difference = std::fabs(lhs - rhs);
  if (difference <= tolerance) {
    return true;
  }
  const double scale = std::fmax(std::fabs(lhs), std::fabs(rhs));
  return scale > 0.0 && difference / scale <= tolerance;
}

}  // namespace detail

// Runs every registered case, isolating failures.
//
// Each body runs inside its own try block so one failure does not hide the
// remaining cases.
//
// Returns 0 if all cases passed, 1 otherwise — suitable as a `main` result and
// as a CTest verdict.
inline int RunAll() {
  std::size_t failures = 0;
  for (const TestCase& test_case : Registry()) {
    try {
      test_case.body();
      std::cout << "[ ok ] " << test_case.name << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cout << "[FAIL] " << test_case.name << "\n       " << error.what()
                << '\n';
    } catch (...) {
      // A throw of something not derived from std::exception would otherwise
      // escape RunAll() and terminate the process, discarding the summary and
      // every result after this case.
      ++failures;
      std::cout << "[FAIL] " << test_case.name
                << "\n       threw a non-std::exception object\n";
    }
  }
  std::cout << "\n"
            << (Registry().size() - failures) << '/' << Registry().size()
            << " tests passed\n";
  return failures == 0 ? 0 : 1;
}

}  // namespace accel::test

// Defines and registers a test case. The body follows the macro like a
// function body: `TEST_CASE(MyTest) { ... }`.
#define TEST_CASE(TestName)                                              \
  static void TestName();                                                \
  static const ::accel::test::Registrar registrar_##TestName(#TestName,  \
                                                             &TestName); \
  static void TestName()

#define CHECK_TRUE(condition)                                    \
  do {                                                           \
    if (!(condition)) {                                          \
      ::accel::test::detail::Fail(__FILE__, __LINE__,            \
                                  "expected true: " #condition); \
    }                                                            \
  } while (0)

#define CHECK_FALSE(condition)                                    \
  do {                                                            \
    if ((condition)) {                                            \
      ::accel::test::detail::Fail(__FILE__, __LINE__,             \
                                  "expected false: " #condition); \
    }                                                             \
  } while (0)

#define CHECK_EQ(actual, expected)                                      \
  do {                                                                  \
    const auto& check_actual = (actual);                                \
    const auto& check_expected = (expected);                            \
    if (!(check_actual == check_expected)) {                            \
      std::ostringstream check_out;                                     \
      check_out << #actual " == " #expected " failed: " << check_actual \
                << " vs " << check_expected;                            \
      ::accel::test::detail::Fail(__FILE__, __LINE__, check_out.str()); \
    }                                                                   \
  } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                           \
  do {                                                                    \
    const double check_actual = static_cast<double>(actual);              \
    const double check_expected = static_cast<double>(expected);          \
    const double check_tolerance = static_cast<double>(tolerance);        \
    if (!::accel::test::detail::CloseEnough(check_actual, check_expected, \
                                            check_tolerance)) {           \
      std::ostringstream check_out;                                       \
      check_out << #actual " ~= " #expected " failed: " << check_actual   \
                << " vs " << check_expected << " (tolerance "             \
                << check_tolerance << ')';                                \
      ::accel::test::detail::Fail(__FILE__, __LINE__, check_out.str());   \
    }                                                                     \
  } while (0)

// Asserts that `statement` throws `exception_type`. Used to pin the
// input-validation contract, which is otherwise easy to regress silently.
#define CHECK_THROWS_AS(statement, exception_type)                           \
  do {                                                                       \
    bool check_threw = false;                                                \
    try {                                                                    \
      statement;                                                             \
    } catch (const exception_type&) {                                        \
      check_threw = true;                                                    \
    }                                                                        \
    if (!check_threw) {                                                      \
      ::accel::test::detail::Fail(                                           \
          __FILE__, __LINE__, #statement " did not throw " #exception_type); \
    }                                                                        \
  } while (0)

#endif  // ACCEL_TESTS_SUPPORT_TEST_HARNESS_H_
