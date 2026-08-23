#ifndef DFTRACER_UTILS_TRACE_VIEWS_DICT_FOLD_H
#define DFTRACER_UTILS_TRACE_VIEWS_DICT_FOLD_H

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/hash_table_dictionary.h>

#include <memory>

namespace dftracer::utils::trace::views::detail {

/// The dictionary fold: harvests hash -> name (FH/HH/SH/PR) from the metadata
/// events the scan already delivers under emit_all_metadata and writes it to
/// the index's HASH_TABLES, so the query builds this index artifact as a
/// byproduct of the one pass it runs anyway.
class DictFold : public Fold {
   public:
    explicit DictFold(dftracer::utils::StringIntern& intern,
                      std::uint64_t budget_bytes = 64ull << 20)
        : intern_(&intern), dict_(budget_bytes) {}

    bool accepts(const ScanShape& shape) const override {
        return shape.include_metadata && shape.emit_all_metadata &&
               !shape.filtered;
    }
    bool needs_args() const override { return true; }

    std::unique_ptr<Fold> slice() const override {
        return std::make_unique<DictFold>(*intern_, dict_.budget());
    }

    void step(const FoldBatch& batch) override;
    void seal_unit(const ScanUnit& unit) override {
        dict_.seal_unit(unit.file_path);
    }
    void drop_unit(const ScanUnit& /*unit*/) override { dict_.drop_unit(); }
    void merge(Fold& slice) override {
        dict_.merge(static_cast<DictFold&>(slice).dict_);
    }
    coro::CoroTask<bool> finalize(const CoverageSet& covered) override {
        return dict_.commit(covered);
    }

    /// Write the sealed hash entries to a caller-owned sink, for the streaming
    /// index build. Seal the units first; no coverage gate.
    void write_to_sink(utilities::indexer::IndexBatchSink& sink) const {
        dict_.write_to_sink(sink);
    }

    std::size_t entry_count() const noexcept { return dict_.entry_count(); }
    bool abandoned() const noexcept { return dict_.abandoned(); }

   private:
    dftracer::utils::StringIntern* intern_;
    HashTableDictionary dict_;
};

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_DICT_FOLD_H
