/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Space filling curve octree assignment to ranks tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/domain/buffer_description.hpp"
#include "cstone/domain/domaindecomp.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

TEST(DomainDecomposition, uniformBins)
{
    {
        unsigned umax = std::numeric_limits<unsigned>::max();
        int numSplits = 2;
        std::vector<unsigned> counts{umax - 10, 5, 5, umax - 11, 5, 6};

        std::vector<TreeNodeIndex> bins(numSplits + 1);
        std::vector<unsigned> binCounts(numSplits);
        uniformBins(counts, bins, binCounts);

        std::vector<TreeNodeIndex> ref{0, 3, 6};
        std::vector<unsigned> refCnt{umax, umax};
        EXPECT_EQ(bins, ref);
        EXPECT_EQ(binCounts, refCnt);
    }
    {
        int numSplits = 2;
        std::vector<unsigned> counts{5, 5, 5, 15, 1, 0};

        std::vector<TreeNodeIndex> bins(numSplits + 1);
        std::vector<unsigned> binCounts(numSplits);
        uniformBins(counts, bins, binCounts);

        std::vector<TreeNodeIndex> ref{0, 3, 6};
        EXPECT_EQ(bins, ref);
    }
    {
        int numSplits = 2;
        std::vector<unsigned> counts{15, 0, 1, 5, 5, 5};

        std::vector<TreeNodeIndex> bins(numSplits + 1);
        std::vector<unsigned> binCounts(numSplits);
        uniformBins(counts, bins, binCounts);

        EXPECT_EQ(*std::min_element(binCounts.begin(), binCounts.end()), 15);
        EXPECT_EQ(*std::max_element(binCounts.begin(), binCounts.end()), 16);
    }
    {
        int numSplits = 7;
        std::vector<unsigned> counts{4, 3, 4, 3, 4, 3, 4, 3, 4, 3};

        std::vector<TreeNodeIndex> bins(numSplits + 1);
        std::vector<unsigned> binCounts(numSplits);
        uniformBins(counts, bins, binCounts);

        EXPECT_EQ(*std::min_element(binCounts.begin(), binCounts.end()), 3);
        EXPECT_EQ(*std::max_element(binCounts.begin(), binCounts.end()), 7);
    }
}

//! @brief boxes that are not thin in Z produce the same bins as uniformBins
TEST(DomainDecomposition, spacialBinsFallback)
{
    using KeyType = uint64_t;
    unsigned l    = maxTreeLevel<KeyType>{};

    int numRanks = 3;
    std::vector<unsigned> counts{4, 3, 4, 3, 4, 3, 4, 3, 4, 3};
    std::vector<KeyType> tree(counts.size() + 1);
    std::iota(tree.begin(), tree.end(), 0);

    std::vector<TreeNodeIndex> refBins(numRanks + 1);
    std::vector<unsigned> refCounts(numRanks);
    uniformBins(counts, refBins, refCounts);

    for (AxesBits axesBits : {AxesBits{l, l, l}, AxesBits{l - 2, l, l}, AxesBits{l, l - 1, l}})
    {
        std::vector<TreeNodeIndex> bins(numRanks + 1);
        std::vector<unsigned> binCounts(numRanks);
        spacialBins(counts, bins, binCounts, tree.data(), axesBits);

        EXPECT_EQ(bins, refBins);
        EXPECT_EQ(binCounts, refCounts);
    }
}

/*! @brief rank boundaries are snapped to the nearest X/Y column boundary
 *
 * One 2D level (Z has one bit less than X/Y) gives 4 columns, starting at keys c << 60 for c = 0..3.
 * Each column is split into two leaves. The last column extends to the end of the key range.
 */
TEST(DomainDecomposition, spacialBinsSnapToColumn)
{
    using KeyType = uint64_t;
    unsigned l    = maxTreeLevel<KeyType>{};

    const KeyType c = KeyType(1) << 60;
    const KeyType s = KeyType(1) << 57;
    std::vector<KeyType> tree{0, s, c, c + s, 2 * c, 2 * c + s, 3 * c, 3 * c + s, nodeRange<KeyType>(0)};
    AxesBits axesBits{l, l, l - 1};

    int numRanks = 2;
    {
        // column counts {20, 10, 30, 20}, target 40 is closer to the start of column 2 (30) than column 3 (60)
        std::vector<unsigned> counts{20, 0, 5, 5, 30, 0, 20, 0};

        std::vector<TreeNodeIndex> bins(numRanks + 1);
        std::vector<unsigned> binCounts(numRanks);
        spacialBins(counts, bins, binCounts, tree.data(), axesBits);

        EXPECT_EQ(bins, (std::vector<TreeNodeIndex>{0, 4, 8}));
        EXPECT_EQ(binCounts, (std::vector<unsigned>{30, 50}));

        // uniformBins splits column 2
        uniformBins(counts, bins, binCounts);
        EXPECT_EQ(bins, (std::vector<TreeNodeIndex>{0, 5, 8}));
    }
    {
        // column counts {20, 30, 20, 10}, target 40 is closer to the start of column 2 (50) than column 1 (20)
        std::vector<unsigned> counts{20, 0, 20, 10, 10, 10, 5, 5};

        std::vector<TreeNodeIndex> bins(numRanks + 1);
        std::vector<unsigned> binCounts(numRanks);
        spacialBins(counts, bins, binCounts, tree.data(), axesBits);

        EXPECT_EQ(bins, (std::vector<TreeNodeIndex>{0, 4, 8}));
        EXPECT_EQ(binCounts, (std::vector<unsigned>{50, 30}));

        // uniformBins splits column 1
        uniformBins(counts, bins, binCounts);
        EXPECT_EQ(bins, (std::vector<TreeNodeIndex>{0, 3, 8}));
    }
}

//! @brief each rank gets at least one column if there are enough columns, even if the columns are empty
TEST(DomainDecomposition, spacialBinsOneColumnPerRank)
{
    using KeyType = uint64_t;
    unsigned l    = maxTreeLevel<KeyType>{};

    const KeyType c = KeyType(1) << 60;
    const KeyType s = KeyType(1) << 57;
    std::vector<KeyType> tree{0, s, c, c + s, 2 * c, 2 * c + s, 3 * c, 3 * c + s, nodeRange<KeyType>(0)};
    AxesBits axesBits{l, l, l - 1};

    {
        int numRanks = 4;
        std::vector<unsigned> counts{0, 0, 0, 0, 0, 0, 0, 100};

        std::vector<TreeNodeIndex> bins(numRanks + 1);
        std::vector<unsigned> binCounts(numRanks);
        spacialBins(counts, bins, binCounts, tree.data(), axesBits);

        EXPECT_EQ(bins, (std::vector<TreeNodeIndex>{0, 2, 4, 6, 8}));
        EXPECT_EQ(binCounts, (std::vector<unsigned>{0, 0, 0, 100}));
    }
    {
        int numRanks = 3;
        std::vector<unsigned> counts{100, 0, 0, 0, 0, 0, 0, 0};

        std::vector<TreeNodeIndex> bins(numRanks + 1);
        std::vector<unsigned> binCounts(numRanks);
        spacialBins(counts, bins, binCounts, tree.data(), axesBits);

        EXPECT_EQ(bins, (std::vector<TreeNodeIndex>{0, 2, 4, 8}));
        EXPECT_EQ(binCounts, (std::vector<unsigned>{100, 0, 0}));
    }
    {
        // more ranks than columns: some ranks are empty, but boundaries stay on column starts and don't decrease
        int numRanks = 6;
        std::vector<unsigned> counts{10, 10, 10, 10, 10, 10, 10, 10};

        std::vector<TreeNodeIndex> bins(numRanks + 1);
        std::vector<unsigned> binCounts(numRanks);
        spacialBins(counts, bins, binCounts, tree.data(), axesBits);

        EXPECT_EQ(bins.front(), 0);
        EXPECT_EQ(bins.back(), 8);
        EXPECT_TRUE(std::is_sorted(bins.begin(), bins.end()));
        for (int r = 0; r <= numRanks; ++r)
        {
            EXPECT_EQ(bins[r] % 2, 0);
        }
        EXPECT_EQ(std::accumulate(binCounts.begin(), binCounts.end(), 0u), 80u);
    }
}

/*! @brief two 2D levels with 32-bit keys give 16 columns
 *
 * The column index c has the coarser 2D level in its upper 2 bits, i.e. column c starts at key
 * ((c >> 2) << 27) | ((c & 3) << 24).
 */
TEST(DomainDecomposition, spacialBinsTwoLevels)
{
    using KeyType = unsigned;
    unsigned l    = maxTreeLevel<KeyType>{};

    std::vector<KeyType> tree;
    for (unsigned column = 0; column < 16; ++column)
    {
        tree.push_back(((column >> 2) << 27) | ((column & 3) << 24));
    }
    tree.push_back(nodeRange<KeyType>(0));
    ASSERT_TRUE(std::is_sorted(tree.begin(), tree.end()));

    int numRanks = 4;
    std::vector<unsigned> counts(16, 5);

    std::vector<TreeNodeIndex> bins(numRanks + 1);
    std::vector<unsigned> binCounts(numRanks);
    spacialBins(counts, bins, binCounts, tree.data(), AxesBits{l, l, l - 2});

    EXPECT_EQ(bins, (std::vector<TreeNodeIndex>{0, 4, 8, 12, 16}));
    EXPECT_EQ(binCounts, (std::vector<unsigned>{20, 20, 20, 20}));
}

//! @brief bins on a cornerstone octree of random particles in a box that is thin in Z
TEST(DomainDecomposition, spacialBinsRandomThinBox)
{
    using KeyType = uint64_t;
    unsigned l    = maxTreeLevel<KeyType>{};

    Box<double> box{0, 1, 0, 1, 0, 0.25};
    const auto axesBits = box.getBoxDimBits(l);
    ASSERT_EQ(axesBits, (AxesBits{l, l, l - 2}));

    const unsigned numColumns  = 16;
    const unsigned columnShift = 3 * (l - 2);
    // index of the column that contains key, leaves in invalid key ranges (digits > 3) belong to the preceding column
    auto columnOf = [columnShift](KeyType key)
    {
        unsigned coarse = (key >> (columnShift + 3)) & 7;
        unsigned fine   = (key >> columnShift) & 7;
        return coarse > 3 ? numColumns - 1 : 4 * coarse + std::min(fine, 3u);
    };

    LocalIndex numParticles = 20000;
    RandomCoordinates<double, SfcKind<KeyType>> coords(numParticles, box);
    auto [tree, counts]     = computeOctree<KeyType>(coords.particleKeys(), 64);
    TreeNodeIndex numLeaves = nNodes(tree);

    std::vector<uint64_t> countScan(counts.size() + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), countScan.begin() + 1);

    std::vector<uint64_t> columnCounts(numColumns, 0);
    for (TreeNodeIndex i = 0; i < numLeaves; ++i)
    {
        columnCounts[columnOf(tree[i])] += counts[i];
    }
    uint64_t maxColumnCount = *std::max_element(columnCounts.begin(), columnCounts.end());

    for (int numRanks : {2, 3, 5, 16})
    {
        std::vector<TreeNodeIndex> bins(numRanks + 1);
        std::vector<unsigned> binCounts(numRanks);
        spacialBins(counts, bins, binCounts, tree.data(), axesBits);

        EXPECT_EQ(bins.front(), 0);
        EXPECT_EQ(bins.back(), numLeaves);
        for (int r = 1; r < numRanks; ++r)
        {
            EXPECT_LT(bins[r - 1], bins[r]);

            // boundary is the first leaf of a column
            KeyType boundary = tree[bins[r]];
            EXPECT_EQ(boundary % (KeyType(1) << columnShift), 0);
            EXPECT_NE(columnOf(boundary), columnOf(tree[bins[r] - 1]));

            // deviation from a perfectly balanced boundary is limited by the heaviest column
            double target = double(r) * numParticles / numRanks;
            EXPECT_LE(std::abs(double(countScan[bins[r]]) - target), maxColumnCount);
        }
        for (int r = 0; r < numRanks; ++r)
        {
            EXPECT_EQ(binCounts[r], countScan[bins[r + 1]] - countScan[bins[r]]);
        }
        EXPECT_EQ(std::accumulate(binCounts.begin(), binCounts.end(), 0u), numParticles);
    }
}

TEST(DomainDecomposition, makeSfcAssignment)
{
    using KeyType = uint64_t;
    std::vector<KeyType> csarray{0, 10, 20, 30, 40};
    std::vector<unsigned> counts{5, 5, 5, 5};

    Box<double> box{0, 1};
    auto        a = makeSfcAssignment(2, counts, csarray.data(), box);
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 20);
    EXPECT_EQ(a[2], 40);

    std::vector<TreeNodeIndex> refOffsets{0, 2, 4};
    EXPECT_TRUE(std::equal(a.treeOffsets().begin(), a.treeOffsets().end(), refOffsets.begin()));

    std::vector<TreeNodeIndex> refNumNodesPerRank{2, 2};
    EXPECT_TRUE(std::equal(a.numNodesPerRank().begin(), a.numNodesPerRank().end(), refNumNodesPerRank.begin()));
}

//! @brief test that the SfcLookupKey can lookup the rank for a given code
TEST(DomainDecomposition, assignmentFindRank)
{
    using KeyType = uint64_t;
    int nRanks    = 3;
    SfcAssignment<KeyType> assignment(nRanks);
    assignment.set(0, 0, 0);
    assignment.set(1, 1, 0);
    assignment.set(2, 3, 0);
    assignment.set(3, 4, 0);

    EXPECT_EQ(0, assignment.findRank(0));
    EXPECT_EQ(1, assignment.findRank(1));
    EXPECT_EQ(1, assignment.findRank(2));
    EXPECT_EQ(2, assignment.findRank(3));
}

TEST(DomainDecomposition, createSendList)
{
    using KeyType = uint64_t;

    std::vector<KeyType> keys{0, 0, 1, 3, 4, 5, 6, 6, 9, 10};

    int numRanks = 2;
    SfcAssignment<KeyType> assignment(numRanks);
    LocalIndex ignored = -1;
    assignment.set(0, 0, ignored);
    assignment.set(1, 6, ignored);
    assignment.set(2, 10, ignored); // note: this excludes the last key

    // note: input keys need to be sorted
    auto sendList = createSendRanges<KeyType>(assignment, keys);

    EXPECT_EQ(sendList.count(0), 6);
    EXPECT_EQ(sendList.count(1), 3);

    EXPECT_EQ(sendList[0], 0);
    EXPECT_EQ(sendList[1], 6);
    EXPECT_EQ(sendList[2], 9);
}

TEST(DomainDecomposition, initialDomainSplit)
{
    using KeyType = uint64_t;

    auto ret = initialDomainSplits<KeyType>(3, 5);
    EXPECT_EQ(ret.front(), 0);
    EXPECT_EQ(ret.back(), nodeRange<KeyType>(0));
}
