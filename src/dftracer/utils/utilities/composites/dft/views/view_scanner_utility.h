#ifndef DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_SCANNER_UTILITY_H
#define DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_SCANNER_UTILITY_H

#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/utilities/streaming_utility.h>
#include <dftracer/utils/utilities/common/query/query.h>
#include <dftracer/utils/utilities/composites/dft/views/fold_event.h>
#include <dftracer/utils/utilities/composites/dft/views/view_definition.h>
#include <dftracer/utils/utilities/indexer/internal/indexer.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW
#include <dftracer/utils/utilities/common/arrow/arrow_export.h>
#include <dftracer/utils/utilities/common/arrow/column_builder.h>
#endif

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace dftracer::utils::utilities::composites::dft::views {

struct ViewScannerInput {
    std::string file_path;
    std::string index_path;
    std::size_t checkpoint_size =
        utilities::indexer::internal::Indexer::DEFAULT_CHECKPOINT_SIZE;
    std::size_t start_byte = 0;
    std::size_t end_byte = 0;
    std::uint64_t checkpoint_idx = 0;
    std::size_t batch_size = 4 * 1024 * 1024;  // IO buffer size
    std::size_t event_batch_size = 10000;      // events per batch
    ViewDefinition view;
    std::optional<common::query::Query> query;

    /// Fold mode: when set, the scanner parses each matching event once and
    /// emits an owned interned FoldEvent (in ViewScannerBatch::fold_events)
    /// instead of a string_view, so the fold consumer does not re-parse. The
    /// intern table is shared across workers for consistent ids.
    dftracer::utils::StringIntern* fold_intern = nullptr;
    bool fold_needs_args = false;

    ViewScannerInput& with_file_path(const std::string& path);
    ViewScannerInput& with_index_path(const std::string& path);
    ViewScannerInput& with_checkpoint_size(std::size_t sz);
    ViewScannerInput& with_byte_range(std::size_t start, std::size_t end);
    ViewScannerInput& with_checkpoint_idx(std::uint64_t idx);
    ViewScannerInput& with_batch_size(std::size_t sz);
    ViewScannerInput& with_event_batch_size(std::size_t sz);
    ViewScannerInput& with_view(const ViewDefinition& v);
};

struct ViewScannerBatch {
    /// Event lines. In stream mode these are string_view into the
    /// decompressed chunk (zero copy, valid until next generator resume).
    /// Metadata events use owned strings stored in owned_events.
    std::vector<std::string_view> events;
    /// Owned storage for metadata events that outlive their source chunk.
    /// Uses deque so push_back doesn't invalidate string_view refs.
    std::deque<std::string> owned_events;
    /// Fold mode (ViewScannerInput::fold_intern set): each matching event
    /// parsed once into an owned FoldEvent, so `events` stays empty.
    std::vector<detail::FoldEvent> fold_events;
    std::uint64_t events_matched = 0;
    std::uint64_t events_scanned = 0;

#ifdef DFTRACER_UTILS_ENABLE_ARROW
    common::arrow::ArrowExportResult to_arrow() const;
    common::arrow::ArrowExportResult to_arrow(
        common::arrow::RecordBatchBuilder& builder) const;
#endif
};

class ViewScannerUtility
    : public StreamingUtility<ViewScannerInput, ViewScannerBatch> {
   public:
    coro::AsyncGenerator<ViewScannerBatch> process(
        const ViewScannerInput& input) override;
};

}  // namespace dftracer::utils::utilities::composites::dft::views

#endif  // DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_SCANNER_UTILITY_H
