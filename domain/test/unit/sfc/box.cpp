/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test box functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include "gtest/gtest.h"
#include "cstone/sfc/box.hpp"

using namespace cstone;

TEST(SfcBox, pbcAdjust)
{
    EXPECT_EQ(pbcAdjust<1024>(-1024), 0);
    EXPECT_EQ(pbcAdjust<1024>(-1), 1023);
    EXPECT_EQ(pbcAdjust<1024>(0), 0);
    EXPECT_EQ(pbcAdjust<1024>(1), 1);
    EXPECT_EQ(pbcAdjust<1024>(1023), 1023);
    EXPECT_EQ(pbcAdjust<1024>(1024), 0);
    EXPECT_EQ(pbcAdjust<1024>(1025), 1);
    EXPECT_EQ(pbcAdjust<1024>(2047), 1023);
}

TEST(SfcBox, pbcDistance)
{
    int R = 1024;
    EXPECT_EQ(pbcDistance(-1024, R), 0);
    EXPECT_EQ(pbcDistance(-513, R), 511);
    EXPECT_EQ(pbcDistance(-512, R), 512);
    EXPECT_EQ(pbcDistance(-1, R), -1);
    EXPECT_EQ(pbcDistance(0, R), 0);
    EXPECT_EQ(pbcDistance(1, R), 1);
    EXPECT_EQ(pbcDistance(512, R), 512);
    EXPECT_EQ(pbcDistance(513, R), -511);
    EXPECT_EQ(pbcDistance(1024, R), 0);
}

TEST(SfcBox, applyPbc)
{
    using T = double;

    Box<T> box(0, 1, BoundaryType::periodic);
    Vec3<T> X{0.9, 0.9, 0.9};
    auto Xpbc = cstone::applyPbc(X, box);

    EXPECT_NEAR(Xpbc[0], -0.1, 1e-10);
    EXPECT_NEAR(Xpbc[1], -0.1, 1e-10);
    EXPECT_NEAR(Xpbc[2], -0.1, 1e-10);
}

TEST(SfcBox, putInBox)
{
    using T = double;
    {
        Box<T> box(0, 1, BoundaryType::periodic);
        Vec3<T> X{0.9, 0.9, 0.9};
        auto Xpbc = cstone::putInBox(X, box);

        EXPECT_NEAR(Xpbc[0], 0.9, 1e-10);
        EXPECT_NEAR(Xpbc[1], 0.9, 1e-10);
        EXPECT_NEAR(Xpbc[2], 0.9, 1e-10);
    }
    {
        Box<T> box(0, 1, BoundaryType::periodic);
        Vec3<T> X{1.1, 1.1, 1.1};
        auto Xpbc = cstone::putInBox(X, box);

        EXPECT_NEAR(Xpbc[0], 0.1, 1e-10);
        EXPECT_NEAR(Xpbc[1], 0.1, 1e-10);
        EXPECT_NEAR(Xpbc[2], 0.1, 1e-10);
    }
    {
        Box<T> box(-1, 1, BoundaryType::periodic);
        Vec3<T> X{-0.9, -0.9, -0.9};
        auto Xpbc = cstone::putInBox(X, box);

        EXPECT_NEAR(Xpbc[0], -0.9, 1e-10);
        EXPECT_NEAR(Xpbc[1], -0.9, 1e-10);
        EXPECT_NEAR(Xpbc[2], -0.9, 1e-10);
    }
}

TEST(SfcBox, createIBox)
{
    {
        using T                = double;
        using KeyType          = uint32_t;
        constexpr int maxCoord = 1u << maxTreeLevel<KeyType>{};

        Box<T> box(0, 1);

        T r = T(1.0) / maxCoord;
        T c = 1.0 - 0.5 * r;
        T s = 0.5 * r;
        Vec3<T> aCenter{c, c, c};
        Vec3<T> aSize{s, s, s};

        IBox probe = createIBox<KeyType>(aCenter, aSize, box);
        IBox ref{maxCoord - 1, maxCoord};
        EXPECT_EQ(ref, probe);
    }
    {
        using T       = double;
        using KeyType = uint64_t;

        Box<T> box(-1, 1, -2, 2, -3, 3);
        Vec3<T> aCenter{0.1, 0.2, 0.3};
        Vec3<T> aSize{0.01, 0.02, 0.03};

        IBox probe = createIBox<KeyType>(aCenter, aSize, box);
        IBox ref{1142947, 1163920};
        EXPECT_EQ(ref, probe);
    }
}

template<typename T>
static bool contains(const Box<T>& large_box, const Box<T>& small_box)
{
    return (large_box.xmin() <= small_box.xmin() && large_box.ymin() <= small_box.ymin() &&
            large_box.zmin() <= small_box.zmin() && large_box.xmax() >= small_box.xmax() &&
            large_box.ymax() >= small_box.ymax() && large_box.zmax() >= small_box.zmax());
};

//! @brief newer box bigger than old box, limitBoxShrink has no effect
TEST(limitBox, expand)
{
    using T                  = double;
    constexpr T shrinkFactor = 0.1;
    auto pbc                 = BoundaryType::periodic;
    auto open                = BoundaryType::open;
    Box<T> previousBox(0, 1, 2, 3, 4, 5, open, pbc, open);
    Box<T> currentBox(-1, 2, -2, 3, -4, 6, open, pbc, open);
    Box<T> limitedBox = limitBoxShrinking(currentBox, previousBox, shrinkFactor);
    EXPECT_EQ(limitedBox, currentBox);
}

//! @brief newer box bigger than shrink limit, limitBoxShrink has no effect
TEST(limitBox, aboveShrinkLimit)
{
    using T                  = double;
    constexpr T shrinkFactor = 0.1001;
    auto pbc                 = BoundaryType::periodic;
    auto open                = BoundaryType::open;
    Box<T> previousBox(0, 1, 2, 3, 4, 5, open, pbc, open);
    Box<T> currentBox(0.1, 0.9, 2.1, 2.9, 4.1, 4.9, open, pbc, open);
    Box<T> limitedBox = limitBoxShrinking(currentBox, previousBox, shrinkFactor);
    EXPECT_EQ(limitedBox, currentBox);
}

//! @brief newer box smaller than shrink limit, limitBoxShrink kicks in
TEST(limitBox, belowShrinkLimit)
{
    using T                  = double;
    constexpr T shrinkFactor = 0.05;
    Box<T> previousBox(1, 2, 10, 20, 100, 200);
    Box<T> currentBox(1.1, 1.9, 11, 19, 110, 190);
    Box<T> limitedBox = limitBoxShrinking(currentBox, previousBox, shrinkFactor);
    EXPECT_NEAR(limitedBox.xmin(), 1.05, 1e-6);
    EXPECT_NEAR(limitedBox.xmax(), 1.95, 1e-6);
    EXPECT_NEAR(limitedBox.ymin(), 10.5, 1e-6);
    EXPECT_NEAR(limitedBox.ymax(), 19.5, 1e-6);
    EXPECT_NEAR(limitedBox.zmin(), 105, 1e-6);
    EXPECT_NEAR(limitedBox.zmax(), 195, 1e-6);

    EXPECT_TRUE(contains(previousBox, limitedBox));
}

TEST(SfcBox, limitBoxShrinking)
{
    using T                   = double;
    constexpr T shrink_factor = 0.1;
    Box<T> previousBox(0., 2., -1., 1., -1., 5.);
    Box<T> currentBox(0., 0.05, 0., 1., 1., 2.);
    Box<T> limitedBox = limitBoxShrinking(currentBox, previousBox, shrink_factor);

    EXPECT_NEAR(limitedBox.lx(), (1. - shrink_factor) * previousBox.lx(), 1e-6);
    EXPECT_NEAR(limitedBox.ly(), (1. - shrink_factor) * previousBox.ly(), 1e-6);
    EXPECT_NEAR(limitedBox.lz(), (1. - 2. * shrink_factor) * previousBox.lz(), 1e-6);

    EXPECT_TRUE(contains(limitedBox, currentBox));
}

TEST(SfcBox, getBoxDimBits)
{
    using T = double;

    using KeyType               = uint64_t;
    constexpr unsigned maxLevel = maxTreeLevel<KeyType>{};
    {
        Box<T> cube(0, 1);
        auto axesBits = cube.getBoxDimBits(maxTreeLevel<KeyType>{});
        EXPECT_EQ(axesBits[0], maxLevel);
        EXPECT_EQ(axesBits[1], maxLevel);
        EXPECT_EQ(axesBits[2], maxLevel);
    }
    {
        // unlimited bits {21, 15, 13}, limited to at most 3 levels difference: {21, 18, 15}
        Box<T> mixDBox(0, 1, 0, 0.015625, 0, 0.00390625);
        auto axesBits = mixDBox.getBoxDimBits(maxTreeLevel<KeyType>{});
        EXPECT_EQ(axesBits[0], 21);
        EXPECT_EQ(axesBits[1], 18);
        EXPECT_EQ(axesBits[2], 15);
        // limited axes keep their min coordinate and are widened to maxExtent / 2^reduction
        EXPECT_NEAR(mixDBox.ly(), 0.125, 1e-12);
        EXPECT_NEAR(mixDBox.lz(), 0.015625, 1e-12);
        EXPECT_EQ(mixDBox.ymin(), 0);
        EXPECT_EQ(mixDBox.zmin(), 0);
        EXPECT_NEAR(mixDBox.ymax(), 0.125, 1e-12);
        EXPECT_NEAR(mixDBox.zmax(), 0.015625, 1e-12);
    }
    {
        Box<T> mixDBox = Box<T>(0, 1, 0, 0.76, 0, 1);
        auto axesBits  = mixDBox.getBoxDimBits(maxTreeLevel<KeyType>{});
        EXPECT_EQ(axesBits[0], 21);
        EXPECT_EQ(axesBits[1], 21);
        EXPECT_EQ(axesBits[2], 21);
    }
    {
        Box<T> mixDBox = Box<T>(0, 1, 0, 0.74, 0, 1);
        auto axesBits  = mixDBox.getBoxDimBits(maxTreeLevel<KeyType>{});
        EXPECT_EQ(axesBits[0], 21);
        EXPECT_EQ(axesBits[1], 20);
        EXPECT_EQ(axesBits[2], 21);
    }
    {
        // zslab disk box: aspect ratio ~11.4 in z gives a reduction of 4 with the default bias (SPHEXA_MIXD_BIAS unset)
        Box<T> mixDBox = Box<T>(-44.3247, 48.449, -47.316, 40.1737, -3.72407, 4.43861);
        auto axesBits  = mixDBox.getBoxDimBits(maxTreeLevel<KeyType>{});
        EXPECT_EQ(axesBits[0], 21);
        EXPECT_EQ(axesBits[1], 21);
        EXPECT_EQ(axesBits[2], 17);
    }
}

/*! @brief MixD level limit: open axes have at most mixDMaxLevelDiff() fewer bits than the next finer axis
 *
 * Assumes SPHEXA_MIXD_BIAS, SPHEXA_MIXD_MAX_LEVEL_DIFF and SPHEXA_BOUNDARY_TYPE are unset (bias 0.585, limit 3).
 */
TEST(SfcBox, mixDLevelLimit)
{
    using T                     = double;
    constexpr unsigned maxLevel = maxTreeLevel<uint64_t>{};
    ASSERT_EQ(mixDMaxLevelDiff(), 3u);

    auto rebuild = [](const Box<T>& b)
    {
        return Box<T>(b.xmin(), b.xmax(), b.ymin(), b.ymax(), b.zmin(), b.zmax(), b.boundaryX(), b.boundaryY(),
                      b.boundaryZ());
    };
    auto expectBits = [](const Box<T>& b, unsigned bx, unsigned by, unsigned bz)
    {
        auto axesBits = b.getBoxDimBits(maxLevel);
        EXPECT_EQ(axesBits[0], bx);
        EXPECT_EQ(axesBits[1], by);
        EXPECT_EQ(axesBits[2], bz);
    };

    {
        // within the limit: {21, 19, 17}, extents unchanged
        Box<T> box(0, 1, 0, 0.2, 0, 0.05);
        expectBits(box, 21, 19, 17);
        EXPECT_EQ(box.ymin(), 0);
        EXPECT_EQ(box.ymax(), 0.2);
        EXPECT_EQ(box.zmin(), 0);
        EXPECT_EQ(box.zmax(), 0.05);
    }
    {
        // unlimited {21, 14, 13} -> {21, 18, 15}
        Box<T> box(0, 1, 0, 1. / 128, 0, 1. / 256);
        expectBits(box, 21, 18, 15);
        EXPECT_NEAR(box.ly(), 1. / 8, 1e-12);
        EXPECT_NEAR(box.lz(), 1. / 64, 1e-12);
        EXPECT_EQ(box.ymin(), 0);
        EXPECT_EQ(box.zmin(), 0);
        EXPECT_NEAR(box.ymax(), 1. / 8, 1e-12);
        EXPECT_NEAR(box.zmax(), 1. / 64, 1e-12);
        EXPECT_NEAR(box.ily(), 8., 1e-9);
        EXPECT_NEAR(box.ilz(), 64., 1e-9);
        EXPECT_EQ(box.xmin(), 0);
        EXPECT_EQ(box.xmax(), 1);
    }
    {
        // equal axes stay equal: unlimited {21, 15, 15} -> {21, 18, 18}
        Box<T> box(0, 1, 0, 1. / 64, 0, 1. / 64);
        expectBits(box, 21, 18, 18);
        EXPECT_NEAR(box.ly(), 1. / 8, 1e-12);
        EXPECT_NEAR(box.lz(), 1. / 8, 1e-12);
    }
    {
        // longest axis along y: unlimited {13, 21, 18} -> {15, 21, 18}
        Box<T> box(0, 1. / 64, -2, 2, 0, 0.5);
        expectBits(box, 15, 21, 18);
        EXPECT_NEAR(box.lx(), 4. / 64, 1e-12);
        EXPECT_EQ(box.xmin(), 0);
        EXPECT_NEAR(box.xmax(), 4. / 64, 1e-12);
        EXPECT_EQ(box.lz(), 0.5);
    }
    {
        // periodic y keeps its bits and extent, open z is limited relative to y: {21, 15, 13} unchanged
        Box<T> box(0, 1, 0, 1. / 64, 0, 1. / 256, BoundaryType::open, BoundaryType::periodic, BoundaryType::open);
        expectBits(box, 21, 15, 13);
        EXPECT_EQ(box.ymax(), 1. / 64);
        EXPECT_EQ(box.zmax(), 1. / 256);
    }
    {
        // periodic z keeps its bits and extent, open y is limited: {21, 15, 13} -> {21, 18, 13}
        Box<T> box(0, 1, 0, 1. / 64, 0, 1. / 256, BoundaryType::open, BoundaryType::open, BoundaryType::periodic);
        expectBits(box, 21, 18, 13);
        EXPECT_NEAR(box.ly(), 1. / 8, 1e-12);
        EXPECT_EQ(box.zmin(), 0);
        EXPECT_EQ(box.zmax(), 1. / 256);
    }
    {
        // fixed and cubic_open axes are not limited either
        Box<T> fixedBox(0, 1, 0, 1. / 64, 0, 1. / 256, BoundaryType::fixed, BoundaryType::fixed, BoundaryType::fixed);
        expectBits(fixedBox, 21, 15, 13);
        Box<T> cubicBox(0, 1, 0, 1. / 64, 0, 1. / 256, BoundaryType::cubic_open, BoundaryType::cubic_open,
                        BoundaryType::cubic_open);
        expectBits(cubicBox, 21, 15, 13);
        EXPECT_EQ(cubicBox.ymax(), 1. / 64);
    }
    {
        // thin axis away from the origin: unlimited z {.., .., 11} -> {21, 21, 18}, widened from its min coordinate
        Box<T> box(1000, 1001, -3, -2, 5000, 5000.001);
        expectBits(box, 21, 21, 18);
        EXPECT_NEAR(box.lz(), 1. / 8, 1e-9);
        EXPECT_EQ(box.zmin(), 5000);
        EXPECT_NEAR(box.zmax(), 5000.125, 1e-9);
    }
    {
        // rebuilding a widened box from its limits gives the same box and bits
        std::vector<Box<T>> boxes{Box<T>(0, 1, 0, 1. / 128, 0, 1. / 256), Box<T>(0, 1, 0, 1. / 64, 0, 1. / 64),
                                  Box<T>(0, 1. / 64, -2, 2, 0, 0.5), Box<T>(1000, 1001, -3, -2, 5000, 5000.001),
                                  Box<T>(-4438, 4438, -755, 755, -25, 25)};
        for (const auto& box : boxes)
        {
            Box<T> rebuilt = rebuild(box);
            EXPECT_EQ(rebuilt, box);
            EXPECT_EQ(rebuilt.getBoxDimBits(maxLevel), box.getBoxDimBits(maxLevel));
        }
    }
    {
        // single precision: widened box is also stable under rebuild
        Box<float> box(1000.f, 1001.f, -3.f, -2.f, 5000.f, 5000.01f);
        Box<float> rebuilt(box.xmin(), box.xmax(), box.ymin(), box.ymax(), box.zmin(), box.zmax());
        EXPECT_EQ(rebuilt, box);
        EXPECT_EQ(rebuilt.getBoxDimBits(maxLevel), box.getBoxDimBits(maxLevel));
        EXPECT_EQ(box.getBoxDimBits(maxLevel)[2], 18);
    }
    {
        // TDE-like disk: unlimited {21, 18, 13} -> {21, 18, 15}, z widened from zmin to 8876 / 64
        Box<T> box(-4438, 4438, -755, 755, -25, 25);
        expectBits(box, 21, 18, 15);
        EXPECT_NEAR(box.lz(), 8876. / 64, 1e-9);
        EXPECT_EQ(box.zmin(), -25);
        EXPECT_NEAR(box.zmax(), -25 + 8876. / 64, 1e-9);
    }
}

TEST(SfcBox, nodeSizeExponents)
{
    using KeyType               = uint64_t;
    constexpr unsigned maxLevel = maxTreeLevel<KeyType>{};
    {
        // cubic box: all axes are subdivided at every level
        AxesBits cube{maxLevel, maxLevel, maxLevel};
        EXPECT_EQ(nodeSizeExponents<KeyType>(cube, 0), (AxesBits{0, 0, 0}));
        EXPECT_EQ(nodeSizeExponents<KeyType>(cube, 1), (AxesBits{1, 1, 1}));
        EXPECT_EQ(nodeSizeExponents<KeyType>(cube, maxLevel), (AxesBits{maxLevel, maxLevel, maxLevel}));
    }
    {
        AxesBits bits{21, 19, 17};
        // level 0: the root node covers the entire box on every axis
        EXPECT_EQ(nodeSizeExponents<KeyType>(bits, 0), (AxesBits{0, 0, 0}));
        // levels 1-2, height 20-19: only x is subdivided (1 valid bit)
        EXPECT_EQ(nodeSizeExponents<KeyType>(bits, 1), (AxesBits{1, 0, 0}));
        EXPECT_EQ(nodeSizeExponents<KeyType>(bits, 2), (AxesBits{2, 0, 0}));
        // levels 3-4, height 18-17: x and y are subdivided (2 valid bits)
        EXPECT_EQ(nodeSizeExponents<KeyType>(bits, 3), (AxesBits{3, 1, 0}));
        EXPECT_EQ(nodeSizeExponents<KeyType>(bits, 4), (AxesBits{4, 2, 0}));
        // levels 5 and deeper, height <= 16: all axes are subdivided (3 valid bits)
        EXPECT_EQ(nodeSizeExponents<KeyType>(bits, 5), (AxesBits{5, 3, 1}));
        EXPECT_EQ(nodeSizeExponents<KeyType>(bits, 18), (AxesBits{18, 16, 14}));
        EXPECT_EQ(nodeSizeExponents<KeyType>(bits, maxLevel), (AxesBits{21, 19, 17}));
    }
    {
        // axesBits are not necessarily sorted, the shortest axis can be any of the three
        AxesBits bits{21, 20, 21};
        EXPECT_EQ(nodeSizeExponents<KeyType>(bits, 1), (AxesBits{1, 0, 1}));
        EXPECT_EQ(nodeSizeExponents<KeyType>(bits, 2), (AxesBits{2, 1, 2}));
    }
}

TEST(SfcBox, nodeSizeFractions)
{
    using T       = double;
    using KeyType = uint64_t;
    {
        auto frac = nodeSizeFractions<KeyType, T>(AxesBits{21, 19, 17}, 3);
        EXPECT_DOUBLE_EQ(frac[0], 1.0 / 8);
        EXPECT_DOUBLE_EQ(frac[1], 1.0 / 2);
        EXPECT_DOUBLE_EQ(frac[2], 1.0);
        // node volume as a fraction of the box volume
        EXPECT_DOUBLE_EQ(frac[0] * frac[1] * frac[2], 1.0 / 16);
    }
    {
        // cubic boxes reduce to the pre-MixD behavior: an edge length of 2^-level on every axis
        constexpr unsigned maxLevel = maxTreeLevel<KeyType>{};
        auto frac                   = nodeSizeFractions<KeyType, T>(AxesBits{maxLevel, maxLevel, maxLevel}, 2);
        EXPECT_DOUBLE_EQ(frac[0], 1.0 / 4);
        EXPECT_DOUBLE_EQ(frac[1], 1.0 / 4);
        EXPECT_DOUBLE_EQ(frac[2], 1.0 / 4);
    }
}
