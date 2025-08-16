# Sanitizer wiring for host C++ targets.
#
# Sanitizer flags must reach both the compiler and the linker: the driver
# injects the runtime library at link time, and a target compiled with
# -fsanitize=address but linked without it fails with unresolved __asan_*
# symbols. Applying them as INTERFACE properties of a single target keeps the
# two in step.
#
# Only CXX sources are instrumented. Device code compiled by NVCC is checked
# with `compute-sanitizer` instead, which is why the sanitizer presets in
# scripts/build-accel.sh disable the CUDA targets outright. See docs/Sanitizers.md.

include_guard(GLOBAL)

# Creates the interface target `accel_sanitizers`; link it into anything
# that should be instrumented.
function(accel_configure_sanitizers)
  add_library(accel_sanitizers INTERFACE)

  set(_flags "")
  set(_enabled "")

  if(ACCEL_ASAN)
    list(APPEND _flags -fsanitize=address -fno-omit-frame-pointer)
    list(APPEND _enabled address)
  endif()

  if(ACCEL_UBSAN)
    # -fno-sanitize-recover turns a diagnosed UB report into a non-zero exit
    # status. Without it UBSan prints and continues, and a scripted run reports
    # success while having found real defects.
    list(APPEND _flags -fsanitize=undefined -fno-sanitize-recover=all
                       -fno-omit-frame-pointer)
    list(APPEND _enabled undefined)
  endif()

  if(ACCEL_TSAN)
    if(ACCEL_ASAN)
      # The two runtimes both claim the shadow-memory address ranges and cannot
      # coexist in one process.
      message(FATAL_ERROR
              "ThreadSanitizer and AddressSanitizer cannot be combined; "
              "configure two build directories instead")
    endif()
    list(APPEND _flags -fsanitize=thread)
    list(APPEND _enabled thread)
  endif()

  if(_flags)
    # Sanitizer reports are only actionable with symbols and un-elided frames,
    # so force debug info regardless of the build type.
    list(APPEND _flags -g)
    target_compile_options(accel_sanitizers INTERFACE
                           $<$<COMPILE_LANGUAGE:CXX>:${_flags}>)
    target_link_options(accel_sanitizers INTERFACE ${_flags})
    message(STATUS "Sanitizers enabled: ${_enabled}")
  endif()
endfunction()

# Applies the sanitizer interface to `target`. A no-op when no sanitizer is on.
function(accel_apply_sanitizers target)
  if(TARGET accel_sanitizers)
    target_link_libraries(${target} PRIVATE accel_sanitizers)
  endif()
endfunction()
