# Handoff: mixed_hilbert.cpp / decodeHilbert investigation

## Original task
Fix `domain/test/unit/sfc/mixed_hilbert.cpp` (a WIP, uncommitted test file), build via
`cmake --build . --parallel 4` in `build_cpu`, run via `run_tests.sh`.

## Status: main fix DONE and verified passing (both locally and via sbatch)

### Bug #1 (fixed): `KeyType` used out of scope
The user had manually rewritten the `level` args in
`TEST(MixedHilbertLeafCenters, EqualDistancesAtLevel)` to
`maxTreeLevel<KeyType>{}-N`, but `KeyType` isn't a template parameter of the `TEST(...)`
macro body — it only exists inside `equalLeafCenterDistances<KeyType>`. Fixed by
substituting the explicit type used in each call (`maxTreeLevel<unsigned>{}-1` /
`maxTreeLevel<uint64_t>{}-1` etc., matching each line's own template argument).

### Bug #2 (fixed): infeasible leaf counts from brute-force enumeration
`equalLeafCenterDistances<KeyType>(lx,ly,lz,level)` enumerates **every** MixD leaf at
`level` (counted from the right, 0 = leaves) and stores all centers in a
`std::vector`. Original `level` values (1,2,3,5) were too close to full resolution:
- `uint64_t(1,1,1, level=1)` → `2^60` leaves → `vector::reserve` throws
  `std::length_error` (exceeds `max_size()` ≈ 7.7e17).
- `unsigned(1,1,1, level=1)` → `2^27` (~134M) leaves → technically completes but eats
  ~51s of the ~51.58s total test runtime.

Fixed (per user, confirmed working) by flipping to `maxTreeLevel<KeyType>{}-N` so
`level` is close to `maxTreeLevel` (coarse/near-root) instead of close to 0
(fine/near-leaf). Resulting leaf counts are small (8 to ~32768), whole test now runs
in ~6ms.

**Both the local `component_units --gtest_filter=*MixedHilbert*` run (6/6 pass) and
the official `sbatch run_tests.sh` submission (through husk's SLURM broker) now
pass**, per the user's confirmation ("The test now pass also with the sbatch script").

### `run_tests.sh` edits (untracked file, not in original scope, but needed for husk)
User iteratively fixed several husk-broker-incompatible `#SBATCH` directives:
- `-N1` → `--nodes 1` (husk's directive parser wants spaced/long form, not attached
  short options)
- `-c1` → removed entirely (same parser issue; omitting it just uses defaults)
- `--uenv=...` / `--view=...` → commented out (`##SBATCH`) since husk always forces
  uenv from the launching session and refuses a script that tries to set it
- `-A csstaff` → **kept** (per explicit user instruction: "The job should be
  submitted under the -A csstaff"); husk's own message about this being
  "controlled by the broker" turned out to still work once submitted with
  `sbatch --partition=preemptible run_tests.sh` (husk only allows the
  `preemptible` partition for brokered jobs — must pass `--partition=preemptible`
  on the CLI, script's `#SBATCH --partition debug` gets force-overridden anyway).

Submission command that ultimately worked (per user confirmation):
```
sbatch --partition=preemptible run_tests.sh
```
(poll with `squeue -j <id>` / `sacct -j <id> -o State`, since husk refuses `--wait`.)

## Current sub-task: the previously-disabled TODO test case

The last line of the test was:
```cpp
// TODO(iomaganaris): Fix this case when the rest of the tests pass
// equalLeafCenterDistances<uint64_t>(10000, 1400, 40, 12);
```

User asked to leave the rest of the test as-is and **uncomment this line**, see if it
still passes. It does NOT crash/timeout anymore (runs in 279ms, leaf count = 2^16 =
65536, `(bx,by,bz)=(21,18,13)`), but fails with a **genuine numeric mismatch**, not a
scale issue:

```
consecutive center distance mismatch at pair (65521, 65522): d=21.875, firstDist=20
consecutive center distance mismatch at pair (65523, 65524): d=19.53125, firstDist=20
consecutive center distance mismatch at pair (65525, 65526): d=21.875, firstDist=20
consecutive center distance mismatch at pair (65527, 65528): d=21.875, firstDist=20
consecutive center distance mismatch at pair (65529, 65530): d=19.53125, firstDist=20
```
All failures cluster at the **tail** of the sequence (indices 65521–65534 out of
65536 total), not scattered through the middle.

**Current instruction from user**: leave the test file as-is (with this case now
uncommented, currently failing), and fix the actual bug in
`domain/include/cstone/sfc/hilbert.hpp`'s `decodeHilbert` (the MixD inverse Hilbert
function) so this test passes. This is the file to edit — **not yet modified**.

## Investigation so far (in progress, not concluded)

Key facts confirmed:
- `maxTreeLevel<unsigned> = 10`, `maxTreeLevel<uint64_t> = 21`
  (`domain/include/cstone/tree/definitions.h`).
- For box `(10000, 1400, 40)`: `getBoxDimBits` gives `bx=21, by=18, bz=13` (x is the
  longest axis, gets full `maxTreeLevel`; y and z get reduced bits based on
  `floor(log2(maxDim/length) + bias)`, `bias = 0.5849625007211563`).
- `equalLeafCenterDistances(..., level=12)` → `octreeLevel = maxLevel - level = 9`.
- Sorted bit budgets: `b0=bz=13` (min), `b1=by=18` (mid), `b2=bx=21` (max, full).
- In `iHilbert`/`decodeHilbert`, the encoding is tiered by depth-from-root:
  - Shallowest digits (near root) are **1D-only** (x alone), count =
    `bits[0]-bits[1] = bx-by = 3` digits.
  - Next tier is **2D** (x,y only), count = `bits[1]-bits[2] = by-bz = 5` digits.
  - Deepest/finest tier (near leaves) is full **3D** Hilbert, count =
    `bits[2] = bz = 13` digits.
  - This is the reverse of what I initially assumed — 1D happens near the root
    (coarse), not near the leaves.
- `octreeLevel=9` falls inside... **still need to nail down exactly which tier
  boundary 9 falls into relative to the 1D(3)/2D(5)/3D(13) split measured from the
  root** — I was mid-derivation of this when interrupted. Rough math suggests depth 9
  from the root would be within the "1D + 2D" tiers (3+5=8) plus 1 digit into the 3D
  tier, i.e. right at a tier transition — which is suspicious given the failures
  cluster at specific indices, possibly exactly where increaseKey's carry crosses a
  tier boundary.
- `centerAndSize<KeyType>(ibox, box)` (in `domain/include/cstone/sfc/box.hpp:400`)
  converts integer `IBox` coords to physical space using **per-axis** scaling:
  `halfUnitLength[i] = 0.5 * uL * box.length(i) * 2^(maxTreeLevel - axesBits[i])`,
  where `uL = 1/2^maxTreeLevel`. I had NOT yet finished verifying whether the
  "equal consecutive distance" assumption is even mathematically valid at
  `octreeLevel=9` for this specific aspect ratio, or whether the assumption only
  holds once the depth has passed the tier-transition point (i.e., whether the test
  itself has a flawed premise vs. `decodeHilbert` has an actual bug). **This is the
  key open question** — need to finish tracing whether isotropic step size is
  actually guaranteed at this depth by construction, then find where in
  `decodeHilbert` the discrepancy originates (likely in the 1D/2D/3D tier
  stitching logic around lines 364-465 of `hilbert.hpp`, particularly the
  swap/complement rotation logic for the 2D tier, or the `permutation[]`
  application at the end).

## Useful debug tooling already built

A standalone reproducer at:
```
/tmp/claude-29613/-capstor-scratch-cscs-ioannmag-CORNERSTONE-sphexa/849c48ac-5cbe-431c-9c9a-cb65eb58ab4d/scratchpad/debug_hilbert.cpp
```
(NOTE: this is in the session-specific scratchpad — may not survive into a new
session/tmpdir. Recreate if needed.) It directly calls `hilbertIBox` for the
failing `(10000,1400,40, level=12)` case and prints `IBox` ranges for a range of
indices. **Critical**: must compile with `-std=gnu++20` (matching the project's
actual build flags) — compiling with `-std=c++17` breaks `get<N>(tuple)` ADL/
two-phase-lookup resolution in `decodeHilbert` and produces spurious
`operator>>` errors that have nothing to do with the real bug. Working compile
command (from `build_cpu`'s ninja rules):
```
/user-environment/env/default/bin/g++ \
  -I<repo>/domain/test/unit/../../include \
  -I<repo>/domain/test/unit/.. \
  -O2 -std=gnu++20 -Wall -Wextra -Wno-unknown-pragmas \
  -o <out> <debug_hilbert.cpp>
```

Sample output around the failure (already captured, reusable without rerunning):
```
bx=21 by=18 bz=13
b0=13 b1=18 b2=21 octreeLevel=9
idx=65510 ibox=[2080768,2084864)x[4096,8192)x[4096,8192)
idx=65511 ibox=[2080768,2084864)x[4096,8192)x[0,4096)
idx=65512 ibox=[2080768,2084864)x[8192,12288)x[0,4096)
idx=65513 ibox=[2080768,2084864)x[8192,12288)x[4096,8192)
idx=65514 ibox=[2080768,2084864)x[12288,16384)x[4096,8192)
idx=65515 ibox=[2080768,2084864)x[12288,16384)x[0,4096)
idx=65516 ibox=[2084864,2088960)x[12288,16384)x[0,4096)
idx=65517 ibox=[2084864,2088960)x[12288,16384)x[4096,8192)
idx=65518 ibox=[2084864,2088960)x[8192,12288)x[4096,8192)
idx=65519 ibox=[2084864,2088960)x[8192,12288)x[0,4096)
idx=65520 ibox=[2088960,2093056)x[8192,12288)x[0,4096)
idx=65521 ibox=[2088960,2093056)x[8192,12288)x[4096,8192)
idx=65522 ibox=[2088960,2093056)x[12288,16384)x[4096,8192)
idx=65523 ibox=[2088960,2093056)x[12288,16384)x[0,4096)
idx=65524 ibox=[2093056,2097152)x[12288,16384)x[0,4096)
idx=65525 ibox=[2093056,2097152)x[12288,16384)x[4096,8192)
idx=65526 ibox=[2093056,2097152)x[8192,12288)x[4096,8192)
idx=65527 ibox=[2093056,2097152)x[8192,12288)x[0,4096)
idx=65528 ibox=[2093056,2097152)x[4096,8192)x[0,4096)
idx=65529 ibox=[2093056,2097152)x[4096,8192)x[4096,8192)
idx=65530 ibox=[2088960,2093056)x[4096,8192)x[4096,8192)
idx=65531 ibox=[2088960,2093056)x[4096,8192)x[0,4096)
idx=65532 ibox=[2088960,2093056)x[0,4096)x[0,4096)
idx=65533 ibox=[2088960,2093056)x[0,4096)x[4096,8192)
idx=65534 ibox=[2093056,2097152)x[0,4096)x[4096,8192)
idx=65535 ibox=[2093056,2097152)x[0,4096)x[0,4096)
```
All cell edges here are uniformly 4096 (=`2^(21-9)`) in x, y, AND z in this shared
`[0, 2^21)` integer frame — i.e. depth 9 behaves like a **regular isotropic**
octree subdivision at this point (all three axes stepping together), which is
notable since bz=13 axis "should" have already frozen its resolution if the naive
"3D-tier-near-leaves-only" reading were the full story. This needs reconciling with
the tier-boundary math above.

## Immediate next steps for whoever picks this up
1. Finish precisely mapping `octreeLevel=9` (`posFromLeft` range) onto the
   1D(3)/2D(5)/3D(13) tier boundaries to determine whether uniform-step isotropy is
   even supposed to hold at this specific depth for this aspect ratio (10000:1400:40).
2. Rerun the debug reproducer (recreate the `.cpp` above if the scratchpad is gone)
   to print the **decoded (ix,iy,iz)** coordinates (not just the derived IBox) for
   the failing key range, and manually re-encode them with `iHilbert<KeyType>` to
   check round-trip consistency — this will show whether `decodeHilbert` is
   mis-decoding specific keys (round-trip breaks) vs. the keys are decoded correctly
   but the equal-spacing assumption itself doesn't hold there.
3. Pay particular attention to `decodeHilbert`'s handling of `maxCoord` and the
   swap/complement loop (`hilbert.hpp` lines ~434-459) that undoes the 2D-tier
   rotation applied to the 3D suffix — this is the most likely place for a boundary
   bug given failures cluster right at the tail of the key range (near `maxCoord`).
4. Once the exact wrong output is identified, fix `decodeHilbert` in
   `domain/include/cstone/sfc/hilbert.hpp` (this is the only file the user wants
   touched for this sub-task), rebuild `component_units`, and rerun
   `--gtest_filter=*MixedHilbert*` locally before doing another `sbatch
   run_tests.sh` round.
5. Do NOT touch `run_tests.sh` further unless another husk directive issue comes up
   — it's already in a working state per the user.
6. Scope reminder from earlier in the conversation: avoid touching
   `domain/include/cstone/sfc/box.hpp` (has unrelated WIP env-var/debug changes,
   already dirty, not part of this task), or any other currently-modified files
   (`ifile_io_hdf5.cpp`, `std_disk.hpp`, `sphexa.cpp`, `traversal_gpu.cu`,
   `visualize_hilbert_mixd.py`) — none of these are implicated in this bug.

## Repo/environment notes
- Working dir: `/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa`, branch `develop`
  (per earlier git status; user later mentioned branch `mixd_fixes` in a worktree
  listing — worth double-checking `git branch --show-current` at the start of a new
  session).
- Build dir: `build_cpu` (already configured, incremental builds work). Full project
  build previously succeeded (exit 0) after the mixed_hilbert.cpp fix.
- Test binary: `build_cpu/domain/test/unit/component_units` — fast local iteration
  via `./domain/test/unit/component_units --gtest_filter="*MixedHilbert*"` (or
  `*MixedHilbertLeafCenters*` to scope to just this one).
- This cluster runs under **husk** (a sandboxing/broker layer) — see the `husk`
  skill for submitting jobs. Key gotchas already discovered: only `preemptible`
  partition is submittable, must pass `--partition=preemptible` explicitly on
  `sbatch`, `--uenv`/`--view` can't be set by the job (inherited from session),
  `#SBATCH` directives must use spaced/long-option form (not attached short
  options like `-N1`/`-c1`), and `-A`/account gets forced by the broker but
  apparently still needs to be present/consistent per this session's experience —
  reread the husk skill if new broker errors show up.
- There is also a **separate git worktree** at
  `.claude/worktrees/hardcode-mixd-bits/` (branch `worktree-hardcode-mixd-bits`) —
  user initially mis-typed this as the target path for the `hilbert.hpp` fix, then
  corrected to the main tree's `domain/include/cstone/sfc/hilbert.hpp`. Do not
  confuse the two paths.
