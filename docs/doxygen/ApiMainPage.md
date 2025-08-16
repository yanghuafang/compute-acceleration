# compute-acceleration API reference

Generated from the `//` comments on the declarations under `src/`. The prose
is the same either way — reading it in the header is often quicker — but the
generated site adds two things a header cannot show: which layer includes which,
and a link from every declaration to the places that use it.

The narrative documentation — architecture, benchmarks, sanitizers, testing —
lives in the repository's `docs/` directory. This site is the reference half.

## Layers

| Layer | Contents |
|-------|----------|
| `core/` | `Span`, `GemmShape`, `MatrixUtils`, `Stopwatch`, `CliOptions` — everything the three backends share |
| `cpu/` | Serial kernels: five GEMM variants, two sum reductions |
| `omp/` | OpenMP kernels and the thread-count control used to pin and report them |
| `cuda/` | Device kernels plus the RAII owners for device memory and events |

The dependency edges run one way: `cpu/`, `omp/` and `cuda/` all include
`core/`, and none of them includes another. `omp/gemm_omp.h` includes
`cpu/cpu_gemm.h` for the shared tile-size constant and nothing else.

## Where to start

- `cpu/cpu_gemm.h` — the five serial GEMM variants and what distinguishes them
- `omp/gemm_omp.h` — why only the fastest serial kernels have parallel counterparts
- `cpu/reduction.h` — why a serial *optimised* reduction exists beside the baseline
- `cuda/device_buffer.cuh` — the ownership discipline the error paths depend on
