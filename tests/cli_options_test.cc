// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

// Tests for the benchmark option parser.
//
// The parser exists to stop a mistyped flag from silently invalidating a
// measurement, so the rejection paths matter more here than the happy path: a
// driver that accepts `--iter=5` and then runs 100 iterations produces a number
// nobody can reproduce.
#include "core/cli_options.h"

#include <cstddef>
#include <stdexcept>

#include "tests/support/test_harness.h"

namespace {

using accel::CliOptions;

// Builds a parser from a literal argument list, prepending a dummy argv[0].
template <std::size_t N>
CliOptions Parse(const char* const (&args)[N]) {
  const char* argv[N + 1];
  argv[0] = "driver";
  for (std::size_t i = 0; i < N; ++i) {
    argv[i + 1] = args[i];
  }
  return CliOptions(static_cast<int>(N + 1), argv, {"m", "k", "n", "iters"});
}

}  // namespace

TEST_CASE(CliOptionsReturnsFallbackForAbsentKeys) {
  const char* const args[] = {"--m=64"};
  const CliOptions options = Parse(args);
  CHECK_EQ(options.PositiveInt("m", 4096), 64);
  CHECK_EQ(options.PositiveInt("n", 4096), 4096);
  CHECK_FALSE(options.has("n"));
  CHECK_TRUE(options.has("m"));
}

TEST_CASE(CliOptionsAcceptsValuelessFlags) {
  const char* const args[] = {"--help"};
  const CliOptions options = Parse(args);
  CHECK_TRUE(options.has("help"));
}

// The regression this parser was tightened for: an unknown key must be fatal,
// not silently dropped in favour of the default.
TEST_CASE(CliOptionsRejectsUnknownKeys) {
  const char* const args[] = {"--iter=5"};  // note: --iters is the real name
  CHECK_THROWS_AS(Parse(args), std::invalid_argument);

  const char* const positional[] = {"64"};
  CHECK_THROWS_AS(Parse(positional), std::invalid_argument);

  const char* const short_form[] = {"-m=64"};
  CHECK_THROWS_AS(Parse(short_form), std::invalid_argument);
}

TEST_CASE(CliOptionsRejectsMalformedNumbers) {
  const char* const trailing[] = {"--m=512x"};
  CHECK_THROWS_AS(Parse(trailing).PositiveInt("m", 1), std::invalid_argument);

  const char* const empty[] = {"--m="};
  CHECK_THROWS_AS(Parse(empty).PositiveInt("m", 1), std::invalid_argument);

  const char* const words[] = {"--m=large"};
  CHECK_THROWS_AS(Parse(words).PositiveInt("m", 1), std::invalid_argument);
}

// Extents and repetition counts are the only integer options, and zero or a
// negative value is meaningless for both.
TEST_CASE(CliOptionsRejectsNonPositiveValues) {
  const char* const zero[] = {"--m=0"};
  CHECK_THROWS_AS(Parse(zero).PositiveInt("m", 1), std::invalid_argument);

  const char* const negative[] = {"--k=-8"};
  CHECK_THROWS_AS(Parse(negative).PositiveInt("k", 1), std::invalid_argument);

  const char* const negative_size[] = {"--n=-1"};
  CHECK_THROWS_AS(Parse(negative_size).PositiveSize("n", 1),
                  std::invalid_argument);
}

// A value beyond INT_MAX must be refused rather than truncated, since the
// extent would otherwise silently become a small or negative number.
TEST_CASE(CliOptionsRejectsIntOverflowButAllowsWideCounts) {
  const char* const too_big[] = {"--m=4294967296"};
  CHECK_THROWS_AS(Parse(too_big).PositiveInt("m", 1), std::invalid_argument);

  const char* const wide[] = {"--n=4294967296"};
  CHECK_EQ(Parse(wide).PositiveSize("n", 1), std::size_t{4294967296ULL});
}

// Last occurrence wins, matching every other command-line tool: scripts append
// overrides to an argument list rather than rewriting it.
TEST_CASE(CliOptionsRepeatedKeyTakesLastValue) {
  const char* const args[] = {"--m=64", "--m=128"};
  CHECK_EQ(Parse(args).PositiveInt("m", 1), 128);
}

int main() { return accel::test::RunAll(); }
