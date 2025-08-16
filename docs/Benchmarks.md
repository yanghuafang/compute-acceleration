# Benchmarks

All measurements use the same product throughout:

```
A(4096 x 2048) * B(2048 x 4096) = C(4096 x 4096)
```

That is 34,359,738,368 multiply-accumulates, or **68.719 GFLOP** counting two
flops per MAC. Every GFLOP/s figure below is that constant divided by the
measured wall-clock time, which is exactly what `GemmGflops()` reports.

## Test systems

| Tag | CPU | GPU | OS | Compiler | Source |
|-----|-----|-----|----|----------|--------|
| `M5/clang`    | Apple M5 | — | macOS 26.6.2 | Apple clang 21.0.0 | measured with the current binaries |
| `7950X/gcc`   | AMD Ryzen 9 7950X | — | Ubuntu 26.04 | g++ 15.2.0 | inherited, see [below](#provenance) |
| `7950X/clang` | AMD Ryzen 9 7950X | — | Ubuntu 26.04 | clang++ 21.1.8 | inherited |
| `M2Pro/clang` | Apple M2 Pro | — | macOS 26.5 | Apple clang 21.0.0 | inherited |
| `3090Ti`      | AMD Ryzen 9 7950X | RTX 3090 Ti FE | Ubuntu 26.04 | CUDA 12.4 | inherited |

All builds are `-O2` or better. The five serial CPU kernels are single-threaded
by design: memory access order and parallelism are separate axes, and measuring
them separately is the only way to say which one a speedup came from. The
OpenMP tables below add the second axis on top of the winner of the first.

OpenMP figures are pinned with `OpenmpSetThreads()` and report the thread
count the runtime actually supplied, not the count requested. An unpinned run
on a hybrid CPU spans performance and efficiency cores and is not reproducible.

## Reproducing

```bash
scripts/bench.sh
```

The default shape takes a few minutes, most of it inside `GemmIjk`, which is
the slow baseline on purpose. For a shape that finishes in seconds:

```bash
scripts/bench.sh --quick
```

Every driver verifies its own output and exits non-zero on a mismatch, so a
green run is a correctness result as well as a timing one.

## CPU: cache blocking alone

`bench_cpu_gemm_row_major` — both operands row-major, so both kernels stride B
by `n` floats in the innermost loop. The only variable is blocking.

| Kernel | `M5/clang` | `7950X/gcc` | `7950X/clang` | `M2Pro/clang` |
|--------|-----------:|------------:|--------------:|--------------:|
| `GemmIjk`   | 76.371 s (0.90 GFLOP/s) | 154 s (0.45 GFLOP/s) | 168 s (0.41 GFLOP/s) | 116 s (0.59 GFLOP/s) |
| `GemmTiled` | 28.895 s (2.38 GFLOP/s) | 112 s (0.61 GFLOP/s) | 117 s (0.59 GFLOP/s) |  31 s (2.22 GFLOP/s) |

Blocking buys 2.6x on the M5, 3.7x on the M2 Pro, and only about 1.4x on the
x86 parts. That spread is the interesting part: the same source change pays off
very differently depending on cache geometry. A 64x64 `float` tile is 16 KiB, so
the three live tiles of a blocked update fit comfortably in the Apple parts'
larger private caches, while on the 7950X they are still being evicted.

Neither result is close to the machine's capability. Both kernels still fetch a
full cache line of B per useful float — blocking reduces how often, but does not
change the access pattern.

## CPU: access order

`bench_cpu_gemm_layouts` — the same product, three different ways of walking
memory. `GemmIjkBColMajor` is the baseline here.

| Kernel | `M5/clang` | `7950X/gcc` | `7950X/clang` | `M2Pro/clang` |
|--------|-----------:|------------:|--------------:|--------------:|
| `GemmIjkBColMajor`   | 15.556 s ( 4.42 GFLOP/s) | 18 s (3.82 GFLOP/s) | 18 s ( 3.82 GFLOP/s) | 28 s ( 2.45 GFLOP/s) |
| `GemmIkj`               |  2.013 s (34.14 GFLOP/s) | 11 s (6.25 GFLOP/s) |  4 s (17.18 GFLOP/s) |  2 s (34.36 GFLOP/s) |
| `GemmTiledBColMajor` |  5.267 s (13.05 GFLOP/s) | 10 s (6.87 GFLOP/s) | 12 s ( 5.73 GFLOP/s) | 10 s ( 6.87 GFLOP/s) |

Three things stand out.

**Transposing B is worth more than blocking.** Going from `GemmIjk` (76.4 s on
`M5/clang`) to `GemmIjkBColMajor` (15.6 s) is 4.9x from a change that touches
no arithmetic — only the order the same bytes are read in. Blocking alone, on the
same machine, bought 2.6x. The transpose itself costs 34 ms, about 1.7% of even
the fastest kernel's runtime, and the driver times and reports it separately so
the comparison is not hiding it.

**Loop reordering beats everything, and its payoff is compiler-dependent.**
`GemmIkj` turns the innermost statement into a scalar-times-row AXPY over
unit-stride B and C, which is the shape an auto-vectoriser accepts. Clang
vectorises it and gcc largely does not: 4 s versus 11 s on the same silicon.
On `M5/clang` it is 37.9x faster than the `GemmIjk` baseline. This is the single
most cost-effective change in the whole CPU set, and it is three lines.

**Blocking and transposing do not compound, and how they interact is
machine-dependent.** `GemmTiledBColMajor` is 3.0x faster than the un-blocked
`GemmIjkBColMajor` on the M5, but it is *slower* than un-blocked on
`7950X/clang` (12 s versus 18 s), and it loses to `GemmIkj` everywhere — by
2.6x on the M5. Both optimisations target the same bottleneck, so stacking them
mostly buys the extra index arithmetic and loop overhead. This is the result
worth remembering: the two changes are not additive, and on some machines the
combination is a regression.

## OpenMP: GEMM scaling

`bench_omp_gemm_scaling` — `GemmIkjOmp` at a smaller shape (1024x512x1024)
than the serial tables use, because it runs the shape once per ladder step per
repetition. Measured on `M5/clang`, ten cores, best of five.

| Threads | Time | GFLOP/s | vs serial | vs 1 thread |
|--------:|-----:|--------:|----------:|------------:|
| serial `GemmIkj` | 0.030 s |  36.3 | — | — |
| 1  | 0.030 s |  36.4 | 1.00x | 1.00x |
| 2  | 0.015 s |  71.1 | 1.96x | 1.95x |
| 4  | 0.009 s | 123.1 | 3.39x | 3.38x |
| 8  | 0.007 s | 158.2 | 4.35x | 4.35x |
| 10 | 0.006 s | 172.3 | 4.74x | 4.74x |

Two denominators because they answer different questions, and at small shapes
they disagree: below roughly 512x256x512 the one-thread OpenMP run measures
1.2-1.4x faster than the serial kernel, an artefact of the outlined region
optimising differently, not of threading. At the shape above the two columns
converge, which is the check that the ladder is measuring what it claims.

**4.74x on ten cores, not 10x.** The kernel streams all of B for every block of
rows, so it hits memory bandwidth well before it runs out of cores; the M5's
efficiency cores flatten the tail further. Anyone quoting the GPU's 93x over
the best *serial* CPU kernel should note that 4.7x of it is available from
threads alone.

`GemmTiledBColMajorOmp` reaches 59.7 GFLOP/s at ten threads — still far
behind `GemmIkjOmp`, so the serial finding that blocking loses to loop
reordering survives parallelisation rather than being reversed by it.

## OpenMP: sum reduction

`bench_omp_sum_reduction` — 67,108,864 floats (256 MiB, chosen to exceed
last-level cache), best of five, `M5/clang`. Effective read bandwidth, so the
row is directly comparable with the CUDA reduction below.

| Kernel | Threads | Time | Bandwidth |
|--------|--------:|-----:|----------:|
| `SumSequential` | 1 | 0.0324 s |  7.7 GiB/s |
| `SumBlocked`    | 1 | 0.0055 s | **45.5 GiB/s** |
| `SumOmp`        | 1 | 0.0324 s |  7.7 GiB/s |
| `SumOmp`        | 2 | 0.0164 s | 15.2 GiB/s |
| `SumOmp`        | 4 | 0.0086 s | 29.1 GiB/s |
| `SumOmp`        | 8 | 0.0060 s | 41.5 GiB/s |
| `SumOmp`        | 10 | 0.0053 s | 47.4 GiB/s |

This is the table that justifies having a serial *optimised* row at all.
Reported as OpenMP-versus-baseline, threads look worth 6.1x. But `SumBlocked`
gets 5.9x of that on a single core, by splitting one accumulator into eight so
the adds stop waiting on each other — the sequential loop was latency-bound,
not bandwidth-bound. It takes all ten cores for `SumOmp` to edge past one core
running `SumBlocked`.

`SumOmp` at one thread matching `SumSequential` exactly is the expected
result: the reduction clause gives one thread one accumulator, which is the
serial loop.

## CUDA: tiled GEMM

`bench_cuda_gemm_tiled` — average over 100 timed launches after a discarded
warm-up, bracketed by CUDA events so the figure is device time, not host
latency.

| Kernel | `3090Ti` |
|--------|---------:|
| `GemmTiledKernel` | 21.758 ms (3158 GFLOP/s) |

Against the best CPU result in this project (2.013 s on `M5/clang`) that is
about **93x**; against the naive `GemmIjk` baseline on the same CPU, roughly
**3500x**.

For scale: a 3090 Ti's FP32 peak is around 40 TFLOP/s, so this kernel reaches
roughly 8% of it. Shared-memory tiling raises arithmetic intensity enough to
Stop being purely bandwidth-bound, but the kernel still issues one shared-memory
load per multiply-add. Closing the rest of the gap needs register blocking
(several outputs per thread), vectorised loads, and double-buffering the tile
stage — none of which is here, because the point of this kernel is to be the
readable version of the idea.

## CUDA: sum reduction

`bench_cuda_sum_reduction` — 4096 floats, 8 blocks of 512 threads, one partial
sum per block, final add on the host.

At this size the measurement is dominated by launch overhead: 4096 floats is
16 KiB, which the device reads in well under a microsecond, while a kernel
launch costs a few microseconds. The driver reports effective read bandwidth
rather than GFLOP/s, since a reduction is memory-bound by nature — but at 16 KiB
that number describes the launch, not the memory system. Pass a larger input to
get a bandwidth figure that means anything:

```bash
bench_cuda_sum_reduction --count=268435456 --blocks=1024 --threads=256
```

The grid-stride loop makes that legal: grid size and input size are independent,
so the same kernel folds 4096 elements or 256 M with no change to the launch
geometry. A kernel that instead required `blocks * threads == count` exactly
would read out of bounds for any other pairing.

## Provenance

The `M5/clang` column was measured with the current binaries from this
repository, `Release`, in one session; every driver reported `[ok]` on its
result check. The OpenMP tables are `M5/clang` only — the machines behind the
other columns were not available to re-run them, and a scaling curve is
meaningless without knowing the core topology it came from.

The `7950X/*`, `M2Pro/clang` and `3090Ti` columns were recorded on hardware that
was not available when this table was assembled. The loop nests are the same, so
the timings remain representative, but they were **not** produced by the binaries
in this tree. The CUDA figures in particular are unverified here, because no
NVIDIA device was on hand. Treat those columns as a documented baseline to beat,
and re-run `scripts/bench.sh` on your own machine before drawing
conclusions.

Two conventions in this table are worth stating outright, because both differ
from how a GEMM loop is commonly micro-benchmarked:

- Operands are filled with a **scaled** ramp (`i * 1e-6`). An unscaled `A[i] = i`
  drives partial sums past 1e17, where `float` spacing exceeds 1e10; compared
  against an absolute tolerance of 1e-3, that is a check which cannot fail.
  Scaling costs nothing in time and makes the verification mean something.
- Timings are reported as fractional seconds. Truncating with
  `duration_cast<seconds>` prints "1 seconds" for a 1.9 s kernel — and is why
  the columns measured elsewhere are whole numbers.
