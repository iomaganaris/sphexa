/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Functions to assign a global cornerstone octree to different ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Any code in this file relies on a global cornerstone octree on each calling rank.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

#include "cstone/tree/csarray.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/sfc/box.hpp"
#include "index_ranges.hpp"

namespace cstone
{

//! @brief determine bins that produce a histogram with uniform number of elements
template<class IndexType>
void uniformBins(const std::vector<IndexType>& counts, std::span<TreeNodeIndex> bins, std::span<LocalIndex> binCounts)
{
    std::vector<uint64_t> countScan(counts.size() + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), countScan.begin() + 1, std::plus<>{}, uint64_t(0));

    int numBins   = bins.size() - 1;
    auto binCount = double(countScan.back()) / numBins;

    bins.front() = 0;
    bins.back()  = counts.size();
#pragma omp parallel for
    for (int i = 1; i < numBins; ++i)
    {
        uint64_t targetCount = i * binCount;
        bins[i]              = std::lower_bound(countScan.begin(), countScan.end(), targetCount) - countScan.begin();
    }
    for (int i = 1; i < numBins; ++i)
    {
        binCounts[i - 1] = countScan[bins[i]] - countScan[bins[i - 1]];
    }
    binCounts.back() = countScan.back() - countScan[bins[numBins - 1]];
}

/*! @brief group leaves into bins by X/Y-plane column, for boxes with fewer octree levels in Z than X/Y
 *
 * @tparam     KeyType    32- or 64-bit unsigned integer SFC key type
 * @tparam     IndexType  integer type of per-leaf particle counts
 * @param[in]  counts     particle counts per leaf of @p tree, size N
 * @param[out] bins       tree-node index of the start of each rank's range, size numRanks + 1
 * @param[out] binCounts  particle count assigned to each rank, size numRanks
 * @param[in]  tree       leaf keys of the global cornerstone octree, sorted ascending, size N + 1
 * @param[in]  axesBits   per-axis SFC bit depth {bx, by, bz}, e.g. from Box::getBoxDimBits
 *
 * numColumns = 4^xyDiffWithZ columns tile the X/Y plane, where xyDiffWithZ is the number of octree levels
 * where X and Y still refine but Z has run out of bits (box thin in Z). Like uniformBins, rank boundaries
 * target equal particle counts, but each boundary is snapped to the nearest column boundary, so a column
 * is never split across ranks and each rank gets at least one column when numColumns >= numRanks. Balance is
 * therefore limited by the heaviest column. Currently assumes
 * axesBits[0] == axesBits[1] (square X/Y footprint), where the 2D levels sit at the top of the key.
 * Falls back to uniformBins when the box isn't thin in Z (xyDiffWithZ == 0).
 */
template<class KeyType, class IndexType>
void spacialBins(const std::vector<IndexType>& counts, std::span<TreeNodeIndex> bins, std::span<LocalIndex> binCounts,
                  const KeyType* tree, AxesBits axesBits)
{
    unsigned minXY       = std::min(axesBits[0], axesBits[1]);
    unsigned xyDiffWithZ = minXY > axesBits[2] ? minXY - axesBits[2] : 0;
    if (xyDiffWithZ == 0)
    {
        // std::cout << "Box is not thin in Z, falling back to uniformBins" << std::endl;
        uniformBins(counts, bins, binCounts);
        return;
    }

    assert(axesBits[0] == axesBits[1]);
    // 64-bit to allow up to maxTreeLevel<uint64_t> 2D levels
    uint64_t numColumns = uint64_t(1) << (2u * xyDiffWithZ);

    int numRanks = int(bins.size()) - 1;
    // std::cout << "Assigning particles to " << numRanks << " ranks with balanced counts, snapped to " << numColumns
    //           << " X/Y-plane columns of the SFC" << std::endl;

    // each 2D level stores its 2-bit quadrant in a full 3-bit octal digit of the MixD Hilbert key
    unsigned columnShift    = 3u * (maxTreeLevel<KeyType>{} - xyDiffWithZ);
    TreeNodeIndex numLeaves = TreeNodeIndex(counts.size());

    std::vector<uint64_t> countScan(counts.size() + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), countScan.begin() + 1, std::plus<>{}, uint64_t(0));

    // columns are only evaluated at rank boundaries, such that the cost does not depend on numColumns

    // first key of column
    auto columnKey = [xyDiffWithZ, columnShift](uint64_t column)
    {
        KeyType key = 0;
        for (unsigned level = 0; level < xyDiffWithZ; ++level)
        {
            key |= KeyType((column >> (2u * level)) & 3u) << (3u * level);
        }
        return KeyType(key << columnShift);
    };
    // last column with columnKey(column) <= key, keys in invalid ranges (digit > 3) map to the preceding column
    auto columnOf = [xyDiffWithZ, columnShift](KeyType key)
    {
        uint64_t column = 0;
        for (int level = int(xyDiffWithZ) - 1; level >= 0; --level)
        {
            unsigned digit = (key >> (columnShift + 3u * level)) & 7u;
            if (digit > 3u) { return ((column + 1) << (2u * (level + 1))) - 1; }
            column = (column << 2u) | digit;
        }
        return column;
    };
    // index of the first leaf in column, numLeaves for column == numColumns
    auto columnStart = [tree, numLeaves, numColumns, &columnKey](uint64_t column)
    {
        if (column == numColumns) { return numLeaves; }
        return TreeNodeIndex(std::lower_bound(tree, tree + numLeaves + 1, columnKey(column)) - tree);
    };

    double rankCount    = double(countScan.back()) / numRanks;
    bins.front()        = 0;
    bins.back()         = numLeaves;
    uint64_t prevColumn = 0;
    for (int r = 1; r < numRanks; ++r)
    {
        uint64_t target = uint64_t(r * rankCount);
        // countScan reaches target at this leaf, therefore the first column that reaches target is the first column
        // starting after the preceding leaf
        TreeNodeIndex leaf = std::lower_bound(countScan.begin(), countScan.end(), target) - countScan.begin();
        uint64_t column    = (leaf == 0) ? 0 : columnOf(tree[leaf - 1]) + 1;
        if (column > 0 && target - countScan[columnStart(column - 1)] < countScan[columnStart(column)] - target)
        {
            --column;
        }
        // every rank keeps at least one column, as long as there are enough columns
        if (numColumns >= uint64_t(numRanks))
        {
            column = std::clamp(column, prevColumn + 1, numColumns - uint64_t(numRanks - r));
        }
        else { column = std::max(column, prevColumn); }
        prevColumn = column;
        bins[r]    = columnStart(column);
    }
    for (int r = 1; r < numRanks; ++r)
    {
        binCounts[r - 1] = countScan[bins[r]] - countScan[bins[r - 1]];
    }
    binCounts.back() = countScan.back() - countScan[bins[numRanks - 1]];
}

//! @brief Stores which parts of the SFC belong to which rank. Each rank has an identical copy
template<class KeyType>
class SfcAssignment
{
public:
    SfcAssignment()
        : rankBoundaries_(1)
    {
    }

    explicit SfcAssignment(int numRanks)
        : rankBoundaries_(numRanks + 1)
        , counts_(numRanks)
        , numNodesPerRank_(numRanks)
        , treeOffsets_(numRanks + 1)
    {
    }

    KeyType* data() { return rankBoundaries_.data(); }
    const KeyType* data() const { return rankBoundaries_.data(); }

    std::span<LocalIndex> counts() { return counts_; }

    std::span<TreeNodeIndex> numNodesPerRank() { return numNodesPerRank_; }
    std::span<const TreeNodeIndex> numNodesPerRankConst() const { return numNodesPerRank_; }

    std::span<TreeNodeIndex> treeOffsets() { return treeOffsets_; }
    std::span<const TreeNodeIndex> treeOffsetsConst() const { return treeOffsets_; }

    void set(int rank, KeyType a, LocalIndex count)
    {
        rankBoundaries_[rank] = a;
        if (rank < int(counts_.size())) { counts_[rank] = count; }
    }

    [[nodiscard]] int numRanks() const { return int(rankBoundaries_.size()) - 1; }
    [[nodiscard]] KeyType operator[](int rank) const { return rankBoundaries_[rank]; }
    [[nodiscard]] LocalIndex totalCount(int rank) const { return counts_[rank]; }

    [[nodiscard]] int findRank(KeyType key) const
    {
        auto it = std::upper_bound(begin(rankBoundaries_), end(rankBoundaries_), key);
        return int(it - begin(rankBoundaries_)) - 1;
    }

private:
    std::vector<KeyType> rankBoundaries_;
    std::vector<LocalIndex> counts_;

    //! number of assigned global tree nodes for each rank
    std::vector<TreeNodeIndex> numNodesPerRank_;
    //! scan of numTreeNodes
    std::vector<TreeNodeIndex> treeOffsets_;
};

template<class KeyType, class T>
SfcAssignment<KeyType> makeSfcAssignment(int numRanks, const std::vector<unsigned>& counts, const KeyType* tree, const Box<T>& box)
{
    SfcAssignment<KeyType> ret(numRanks);
    const auto axesBits = box.getBoxDimBits(maxTreeLevel<KeyType>{});

    const char* spacialBinsEnv = std::getenv("SPHEXA_SPACIAL_BINS");
    if (spacialBinsEnv && std::atoi(spacialBinsEnv) == 1)
    {
        spacialBins(counts, ret.treeOffsets(), ret.counts(), tree, axesBits);
    }
    else { uniformBins(counts, ret.treeOffsets(), ret.counts()); }
    gather(ret.treeOffsetsConst(), tree, ret.data());

    std::span numNodesPerRank = ret.numNodesPerRank();
    std::span offsets         = ret.treeOffsets();
    for (TreeNodeIndex i = 0; i < numRanks; ++i)
    {
        numNodesPerRank[i] = offsets[i + 1] - offsets[i];
    }

    return ret;
}

/*! @brief translates an assignment of a given tree to a new tree
 *
 * @tparam     KeyType         32- or 64-bit unsigned integer
 * @param[in]  assignment      domain assignment
 * @param[in]  focusTree       focus tree leaves
 * @param[out] focusAssignment assignment with the same SFC key ranges per
 *                             peer rank as the domain @p assignment,
 *                             but with indices valid w.r.t @p focusTree
 */
template<class KeyType>
void translateAssignment(const SfcAssignment<KeyType>& assignment,
                         std::span<const KeyType> focusTree,
                         std::vector<TreeIndexPair>& focusAssignment)
{
    int numRanks = assignment.numRanks();
    focusAssignment.resize(numRanks);
    std::fill(focusAssignment.begin(), focusAssignment.end(), TreeIndexPair(0, 0));
#pragma omp parallel for schedule(static)
    for (int rank = 0; rank < numRanks; ++rank)
    {
        // Note: start-end range is narrowed down if no exact match is found.
        TreeNodeIndex startIndex = findNodeAbove(focusTree.data(), focusTree.size(), assignment[rank]);
        TreeNodeIndex endIndex   = findNodeBelow(focusTree.data(), focusTree.size(), assignment[rank + 1]);

        if (endIndex < startIndex) { endIndex = startIndex; }
        focusAssignment[rank] = TreeIndexPair(startIndex, endIndex);
    }
}

inline void extractPeerRanges(std::span<const int> peers,
                              int myRank,
                              std::span<const TreeIndexPair> focusAssignment,
                              std::vector<TreeIndexPair>& peerRanges)
{
    peerRanges.resize(peers.size() + 1);
    gather<int>(peers, focusAssignment.data(), peerRanges.data());
    peerRanges.back() = focusAssignment[myRank];
    std::sort(peerRanges.begin(), peerRanges.end());
}

/*! @brief Based on global assignment, create the list of local particle index ranges to send to each rank
 *
 * @tparam KeyType      32- or 64-bit integer
 * @param assignment    global space curve assignment to ranks
 * @param particleKeys  sorted list of SFC keys of local particles present on this rank
 * @return              for each rank, a list of index ranges into @p particleKeys to send
 *
 * Converts the global assignment particle keys ranges into particle indices with binary search
 */
template<class KeyType>
SendRanges createSendRanges(const SfcAssignment<KeyType>& assignment, std::span<const KeyType> particleKeys)
{
    int numRanks = assignment.numRanks();

    SendRanges ret(numRanks + 1);
    for (int rank = 0; rank <= numRanks; ++rank)
    {
        KeyType rangeStart = assignment[rank];
        ret[rank] = std::lower_bound(particleKeys.begin(), particleKeys.end(), rangeStart) - particleKeys.begin();
    }

    return ret;
}

//! @brief translate send ranges indexed in o1 ordering to o2 ordering
inline SendRanges shiftSendRanges(SendRanges ranges, int thisRank, LocalIndex numIncoming)
{
    for (int rank = thisRank + 1; rank <= ranges.numRanks(); ++rank)
    {
        ranges[rank] += numIncoming;
    }
    return ranges;
}

/*! @brief return @p numRanks equal length SFC segments for initial domain decomposition
 *
 * @tparam KeyType
 * @param numRanks    number of segments
 * @param level       maximum tree depths or (=number of non-zero leading octal digits)
 * @return            the segments
 *
 * Example: returns [0 2525200000 5252500000 10000000000] for numRanks = 3 and level = 5
 */
template<class KeyType>
std::vector<KeyType> initialDomainSplits(int numRanks, int level)
{
    std::vector<KeyType> ret(numRanks + 1);
    KeyType delta = nodeRange<KeyType>(0) / numRanks;

    ret.front() = 0;
    for (int i = 1; i < numRanks; ++i)
    {
        ret[i] = enclosingBoxCode(KeyType(i) * delta, level);
    }
    ret.back() = nodeRange<KeyType>(0);

    return ret;
}

} // namespace cstone
