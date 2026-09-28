/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Tests the global bounding box
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <numeric>
#include <random>
#include <vector>

#include "cstone/sfc/box_mpi.hpp"

using namespace cstone;

TEST(GlobalBox, localMinMax)
{
    using T = double;

    int numElements = 1000;
    std::vector<T> x(numElements);
    std::iota(begin(x), end(x), 1);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(begin(x), end(x), g);

    auto [gmin, gmax] = minMax(execution::cpu, x.data(), x.data() + x.size());
    EXPECT_EQ(gmin, T(1));
    EXPECT_EQ(gmax, T(numElements));
}

template<class T>
void makeGlobalBox(int rank, int numRanks)
{
    T val = rank + 1;
    std::vector<T> x{-val, val};
    std::vector<T> y{val, 2 * val};
    std::vector<T> z{-val, -2 * val};

    Box<T> box = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), MPI_COMM_WORLD, Box<T>{0, 1});

    T rVal = numRanks;
    EXPECT_EQ(box.xmin(), -rVal);
    EXPECT_EQ(box.xmax(), rVal);
    EXPECT_EQ(box.ymin(), T(1));
    EXPECT_EQ(box.ymax(), 2 * rVal);
    EXPECT_EQ(box.zmin(), -2 * rVal);
    EXPECT_EQ(box.zmax(), T(-1));

    auto open     = BoundaryType::open;
    auto periodic = BoundaryType::periodic;

    // PBC case
    {
        Box<T> pbcBox{0, 1, 0, 1, 0, 1, periodic, periodic, periodic};
        Box<T> newPbcBox = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), MPI_COMM_WORLD, pbcBox);
        EXPECT_EQ(pbcBox, newPbcBox);
    }
    // partial PBC
    {
        Box<T> pbcBox{0, 1, 0, 1, 0, 1, open, periodic, periodic};
        Box<T> newPbcBox = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), MPI_COMM_WORLD, pbcBox);
        Box<T> refBox{-rVal, rVal, 0, 1, 0, 1, open, periodic, periodic};
        EXPECT_EQ(refBox, newPbcBox);
    }
    {
        Box<T> pbcBox{0, 1, 0, 1, 0, 1, periodic, open, periodic};
        Box<T> newPbcBox = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), MPI_COMM_WORLD, pbcBox);
        Box<T> refBox{0, 1, T(1), 2 * rVal, 0, 1, periodic, open, periodic};
        EXPECT_EQ(refBox, newPbcBox);
    }
    {
        Box<T> pbcBox{0, 1, 0, 1, 0, 1, periodic, periodic, open};
        Box<T> newPbcBox = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), MPI_COMM_WORLD, pbcBox);
        Box<T> refBox{0, 1, 0, 1, -2 * rVal, T(-1), periodic, periodic, open};
        EXPECT_EQ(refBox, newPbcBox);
    }
}

TEST(GlobalBox, diskBoundMultiplier)
{
    using T = double;
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // particles on rank 0 only, other ranks are empty
    std::vector<T> x{1, 3}, y{2, 6}, z{0, 1};
    size_t         numElements = rank == 0 ? x.size() : 0;

    auto open      = BoundaryType::open;
    auto cubicOpen = BoundaryType::cubic_open;
    auto periodic  = BoundaryType::periodic;

    setenv("SPHEXA_DISK_BOUND_MULTIPLIER", "4", 1);

    // open x/y are scaled about their center, z is not
    {
        Box<T> prev{0, 1, 0, 1, 0, 1, open, open, open};
        Box<T> box = makeGlobalBox(execution::cpu, x.data(), y.data(), z.data(), numElements, MPI_COMM_WORLD, prev);
        EXPECT_EQ(box.xmin(), T(-2));
        EXPECT_EQ(box.xmax(), T(6));
        EXPECT_EQ(box.ymin(), T(-4));
        EXPECT_EQ(box.ymax(), T(12));
        EXPECT_EQ(box.zmin(), T(0));
        EXPECT_EQ(box.zmax(), T(1));

        // recomputing from the scaled box must not compound the multiplier, even with empty ranks
        Box<T> box2 = makeGlobalBox(execution::cpu, x.data(), y.data(), z.data(), numElements, MPI_COMM_WORLD, box);
        EXPECT_EQ(box, box2);
    }
    // cubic_open: scaled first, then expanded to the largest side
    {
        Box<T> prev{0, 1, 0, 1, 0, 1, cubicOpen, cubicOpen, cubicOpen};
        Box<T> box = makeGlobalBox(execution::cpu, x.data(), y.data(), z.data(), numElements, MPI_COMM_WORLD, prev);
        EXPECT_EQ(box.xmin(), T(-2));
        EXPECT_EQ(box.xmax(), T(14));
        EXPECT_EQ(box.ymin(), T(-4));
        EXPECT_EQ(box.ymax(), T(12));
        EXPECT_EQ(box.zmin(), T(0));
        EXPECT_EQ(box.zmax(), T(16));
    }
    // periodic axes keep their limits
    {
        Box<T> prev{0, 1, 0, 1, 0, 1, periodic, open, open};
        Box<T> box = makeGlobalBox(execution::cpu, x.data(), y.data(), z.data(), numElements, MPI_COMM_WORLD, prev);
        EXPECT_EQ(box.xmin(), T(0));
        EXPECT_EQ(box.xmax(), T(1));
        EXPECT_EQ(box.ymin(), T(-4));
        EXPECT_EQ(box.ymax(), T(12));
    }

    unsetenv("SPHEXA_DISK_BOUND_MULTIPLIER");
}
