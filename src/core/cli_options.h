// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_CORE_CLI_OPTIONS_H_
#define ACCEL_CORE_CLI_OPTIONS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace accel {

// Minimal --key=value / --flag parser. Only the attached form: the detached
// --key value form needs a per-option arity table, and getting that wrong
// silently swallows the next argument.
//
// Each driver declares its own option names and anything outside that set is
// rejected. That check is why this exists rather than a two-line argv loop --
// a mistyped --iter=5 that left the default of 100 in place would quietly
// invalidate a published measurement.
//
// Immutable after construction. Throws std::invalid_argument from the
// constructor on a malformed or unknown argument, and from the accessors on a
// value that will not parse.
class CliOptions {
 public:
  //   argc,argv  As received by `main`; `argv[0]` is skipped.
  //   known_keys  Option names accepted by this driver, without the leading
  //              dashes. `"help"` is always accepted and need not be listed.
  CliOptions(int argc, const char* const* argv,
             std::initializer_list<const char*> known_keys) {
    std::vector<std::string> allowed(known_keys.begin(), known_keys.end());
    allowed.emplace_back("help");

    for (int i = 1; i < argc; ++i) {
      std::string arg(argv[i]);
      if (arg.rfind("--", 0) != 0 || arg.size() <= 2) {
        throw std::invalid_argument("unrecognised argument '" + arg +
                                    "' (expected --key=value or --flag)");
      }
      arg.erase(0, 2);
      const std::size_t separator = arg.find('=');
      std::string key = arg.substr(0, separator);
      if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
        throw std::invalid_argument("unknown option '--" + key +
                                    "'; try --help");
      }
      std::string value = separator == std::string::npos
                              ? std::string()
                              : arg.substr(separator + 1);
      // Assignment, not emplace: a repeated option takes its last value, which
      // is what every other command-line tool does and what a scripted
      // "append an override" invocation expects.
      values_[std::move(key)] = std::move(value);
    }
  }

  // True when `key` was supplied, with or without a value.
  bool has(const std::string& key) const {
    return values_.find(key) != values_.end();
  }

  // Reads `key` as a positive integer.
  //
  // Returns `fallback` when `key` is absent.
  //
  // Throws std::invalid_argument if the value is missing, unparsable, or not
  // strictly positive — every integer option here is an extent or a repetition
  // count, and zero is never meaningful.
  int PositiveInt(const std::string& key, int fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) {
      return fallback;
    }
    const std::int64_t parsed = ParseInt64(key, it->second);
    if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("--" + key + " must be a positive int, got " +
                                  it->second);
    }
    return static_cast<int>(parsed);
  }

  // As PositiveInt(), widened for element counts that exceed 2^31.
  std::size_t PositiveSize(const std::string& key, std::size_t fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) {
      return fallback;
    }
    const std::int64_t parsed = ParseInt64(key, it->second);
    if (parsed <= 0) {
      throw std::invalid_argument(
          "--" + key + " must be a positive count, got " + it->second);
    }
    return static_cast<std::size_t>(parsed);
  }

 private:
  // int64_t rather than long: long is 32-bit on Windows and 64-bit
  // elsewhere, which would silently make the range check in PositiveInt()
  // unreachable on one of them.
  static std::int64_t ParseInt64(const std::string& key,
                                 const std::string& text) {
    if (text.empty()) {
      throw std::invalid_argument("--" + key + " requires a value");
    }
    std::size_t consumed = 0;
    std::int64_t parsed = 0;
    try {
      parsed = std::stoll(text, &consumed);
    } catch (const std::exception&) {
      throw std::invalid_argument("--" + key + " is not a number: " + text);
    }
    // Reject trailing junk such as "512x" that stol() would otherwise accept.
    if (consumed != text.size()) {
      throw std::invalid_argument("--" + key + " is not a number: " + text);
    }
    return parsed;
  }

  std::unordered_map<std::string, std::string> values_;
};

}  // namespace accel

#endif  // ACCEL_CORE_CLI_OPTIONS_H_
