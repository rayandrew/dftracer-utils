#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/core/common/error.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/utilities/common/arrow/array_view.h>
#include <dftracer/utils/utilities/common/arrow/column_builder.h>
#include <dftracer/utils/utilities/common/arrow/gapfill.h>
#include <dftracer/utils/utilities/common/arrow/join_internal.h>
#include <nanoarrow/nanoarrow.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <vector>

namespace dftracer::utils::utilities::common::arrow {

namespace {

constexpr std::int64_t GRID_MAX_POINTS = 10'000'000;

// floor(t / width) as an integer grid index; the grid point value is index *
// width, so comparisons on indices stay exact for a floating-point time column.
std::int64_t floor_index(std::int64_t t, std::int64_t width) {
    std::int64_t q = t / width;
    if (t % width != 0 && t < 0) --q;
    return q;
}
std::int64_t floor_index(double t, double width) {
    return static_cast<std::int64_t>(std::floor(t / width));
}

std::int64_t read_time(const ArrowArrayView* v, std::int64_t row, KeyKind k) {
    if (k == KeyKind::UNSIGNED)
        return static_cast<std::int64_t>(ArrowArrayViewGetUIntUnsafe(v, row));
    return ArrowArrayViewGetIntUnsafe(v, row);
}

template <typename Time>
struct Real {
    std::int64_t row;
    Time time;
    std::int64_t bucket;  // grid index (floor(time / width))
};

template <typename Time>
ArrowExportResult gap_fill_impl(const ArrowSchema* s, const ArrowArray* a,
                                const std::uint32_t* partition_cols,
                                std::uint32_t n_part, std::uint32_t time_col,
                                Time bucket_width,
                                const std::uint32_t* value_cols,
                                std::uint32_t n_value, GapFillMode mode,
                                bool has_range, Time range_start,
                                Time range_end) {
    if (bucket_width <= 0) {
        throw DFTUtilsException(ErrorCode::INVALID_ARGUMENT,
                                "gap_fill: bucket_width must be positive");
    }

    ArrowArrayView av;
    if (init_array_view(av, const_cast<ArrowSchema*>(s),
                        const_cast<ArrowArray*>(a)) != NANOARROW_OK) {
        throw DFTUtilsException(ErrorCode::INTERNAL,
                                "gap_fill: failed to view batch");
    }
    struct Guard {
        ArrowArrayView* v;
        ~Guard() { ArrowArrayViewReset(v); }
    } guard{&av};

    const std::int64_t ncols = av.n_children;
    if (time_col >= static_cast<std::uint32_t>(ncols)) {
        throw DFTUtilsException(ErrorCode::INVALID_ARGUMENT,
                                "gap_fill: time column index out of range");
    }
    const ArrowArrayView* tview = av.children[time_col];
    auto tk = key_kind_from_storage(tview->storage_type);
    KeyKind tkind;
    if constexpr (std::is_same_v<Time, double>) {
        if (!tk || *tk != KeyKind::FLOAT) {
            throw DFTUtilsException(
                ErrorCode::INVALID_ARGUMENT,
                "gap_fill: time column must be floating point");
        }
        tkind = *tk;
    } else {
        if (!tk || (*tk != KeyKind::SIGNED && *tk != KeyKind::UNSIGNED)) {
            throw DFTUtilsException(ErrorCode::INVALID_ARGUMENT,
                                    "gap_fill: time column must be integer");
        }
        tkind = *tk;
    }

    std::vector<char> is_part(static_cast<std::size_t>(ncols), 0);
    std::vector<char> is_value(static_cast<std::size_t>(ncols), 0);
    for (std::uint32_t k = 0; k < n_part; ++k) {
        if (partition_cols[k] >= static_cast<std::uint32_t>(ncols)) {
            throw DFTUtilsException(
                ErrorCode::INVALID_ARGUMENT,
                "gap_fill: partition column index out of range");
        }
        is_part[partition_cols[k]] = 1;
    }
    for (std::uint32_t k = 0; k < n_value; ++k) {
        if (value_cols[k] >= static_cast<std::uint32_t>(ncols)) {
            throw DFTUtilsException(
                ErrorCode::INVALID_ARGUMENT,
                "gap_fill: value column index out of range");
        }
        is_value[value_cols[k]] = 1;
        if (mode == GapFillMode::LINEAR) {
            auto vk =
                key_kind_from_storage(av.children[value_cols[k]]->storage_type);
            if (!vk || *vk == KeyKind::BYTES) {
                throw DFTUtilsException(
                    ErrorCode::INVALID_ARGUMENT,
                    "gap_fill: LINEAR fill over a non-numeric column");
            }
        }
    }

    KeyViews pkeys(n_part);
    std::vector<KeyKind> pkinds(n_part);
    for (std::uint32_t k = 0; k < n_part; ++k) {
        const ArrowArrayView* v = av.children[partition_cols[k]];
        auto kk = key_kind_from_storage(v->storage_type);
        if (!kk) {
            throw DFTUtilsException(
                ErrorCode::INVALID_ARGUMENT,
                "gap_fill: unsupported partition column type");
        }
        pkeys[k] = v;
        pkinds[k] = *kk;
    }

    std::vector<ColumnSpec> out_specs;
    std::vector<ValueCol> lvals;
    out_specs.reserve(static_cast<std::size_t>(ncols));
    lvals.reserve(static_cast<std::size_t>(ncols));
    for (std::int64_t c = 0; c < ncols; ++c) {
        ColumnSpec spec;
        lvals.push_back(plan_value_col(av.children[c], s->children[c],
                                       s->children[c]->name, spec));
        if (mode == GapFillMode::LINEAR &&
            is_value[static_cast<std::size_t>(c)])
            spec.type = ColumnType::DOUBLE;
        out_specs.push_back(std::move(spec));
    }

    RecordBatchBuilder b;
    b.declare_schema(out_specs);

    const std::int64_t N = av.length;
    std::vector<std::int64_t> idx(static_cast<std::size_t>(N));
    std::iota(idx.begin(), idx.end(), 0);

    auto null_last_cmp = [&](const ArrowArrayView* v, KeyKind k, std::int64_t x,
                             std::int64_t y) {
        bool nx = ArrowArrayViewIsNull(v, x);
        bool ny = ArrowArrayViewIsNull(v, y);
        if (nx || ny) return nx && ny ? 0 : (nx ? 1 : -1);
        return cell_cmp(v, x, v, y, k);
    };
    std::sort(idx.begin(), idx.end(), [&](std::int64_t x, std::int64_t y) {
        for (std::uint32_t k = 0; k < n_part; ++k) {
            int c = null_last_cmp(pkeys[k], pkinds[k], x, y);
            if (c != 0) return c < 0;
        }
        int c = null_last_cmp(tview, tkind, x, y);
        if (c != 0) return c < 0;
        return x < y;
    });

    auto same_partition = [&](std::int64_t x, std::int64_t y) {
        for (std::uint32_t k = 0; k < n_part; ++k) {
            bool nx = ArrowArrayViewIsNull(pkeys[k], x);
            bool ny = ArrowArrayViewIsNull(pkeys[k], y);
            if (nx != ny) return false;
            if (nx) continue;
            if (cell_cmp(pkeys[k], x, pkeys[k], y, pkinds[k]) != 0)
                return false;
        }
        return true;
    };

    // Grid point value for index k as a double (for LINEAR interpolation).
    auto grid_time_d = [&](std::int64_t k) -> double {
        if constexpr (std::is_same_v<Time, double>)
            return static_cast<double>(k) * bucket_width;
        else
            return static_cast<double>(k * bucket_width);
    };

    auto emit_real = [&](std::int64_t row) {
        for (std::int64_t c = 0; c < ncols; ++c) {
            const std::size_t oc = static_cast<std::size_t>(c);
            if (mode == GapFillMode::LINEAR && is_value[oc]) {
                if (ArrowArrayViewIsNull(lvals[oc].view, row))
                    b.append_null(oc);
                else
                    b.append_double(
                        oc, ArrowArrayViewGetDoubleUnsafe(lvals[oc].view, row));
            } else {
                append_value(b, oc, lvals[oc], row);
            }
        }
        b.end_row();
    };

    auto emit_generated = [&](std::int64_t rep, std::int64_t k,
                              std::int64_t prev_idx, std::int64_t next_idx,
                              const std::vector<Real<Time>>& reals) {
        for (std::int64_t c = 0; c < ncols; ++c) {
            const std::size_t oc = static_cast<std::size_t>(c);
            if (static_cast<std::uint32_t>(c) == time_col) {
                if constexpr (std::is_same_v<Time, double>) {
                    b.append_double(oc, static_cast<double>(k) * bucket_width);
                } else {
                    if (tkind == KeyKind::UNSIGNED)
                        b.append_uint64(
                            oc, static_cast<std::uint64_t>(k * bucket_width));
                    else
                        b.append_int64(oc, k * bucket_width);
                }
            } else if (is_part[oc]) {
                append_value(b, oc, lvals[oc], rep);
            } else if (is_value[oc]) {
                if (mode == GapFillMode::LOCF && prev_idx >= 0) {
                    append_value(b, oc, lvals[oc],
                                 reals[static_cast<std::size_t>(prev_idx)].row);
                } else if (mode == GapFillMode::LINEAR && prev_idx >= 0 &&
                           next_idx >= 0) {
                    const Real<Time>& lo =
                        reals[static_cast<std::size_t>(prev_idx)];
                    const Real<Time>& hi =
                        reals[static_cast<std::size_t>(next_idx)];
                    if (ArrowArrayViewIsNull(lvals[oc].view, lo.row) ||
                        ArrowArrayViewIsNull(lvals[oc].view, hi.row)) {
                        b.append_null(oc);
                    } else {
                        double v0 = ArrowArrayViewGetDoubleUnsafe(
                            lvals[oc].view, lo.row);
                        double v1 = ArrowArrayViewGetDoubleUnsafe(
                            lvals[oc].view, hi.row);
                        double t0 = static_cast<double>(lo.time);
                        double t1 = static_cast<double>(hi.time);
                        double g = grid_time_d(k);
                        b.append_double(oc,
                                        v0 + (v1 - v0) * (g - t0) / (t1 - t0));
                    }
                } else {
                    b.append_null(oc);
                }
            } else {
                b.append_null(oc);
            }
        }
        b.end_row();
    };

    std::vector<Real<Time>> reals;
    for (std::int64_t p = 0; p < N;) {
        std::int64_t q = p + 1;
        while (q < N && same_partition(idx[static_cast<std::size_t>(p)],
                                       idx[static_cast<std::size_t>(q)]))
            ++q;

        reals.clear();
        for (std::int64_t r = p; r < q; ++r) {
            const std::int64_t g = idx[static_cast<std::size_t>(r)];
            if (ArrowArrayViewIsNull(tview, g)) continue;
            Time t;
            if constexpr (std::is_same_v<Time, double>)
                t = ArrowArrayViewGetDoubleUnsafe(tview, g);
            else
                t = static_cast<Time>(read_time(tview, g, tkind));
            const std::int64_t bucket = floor_index(t, bucket_width);
            if (!reals.empty() && reals.back().bucket == bucket) continue;
            reals.push_back({g, t, bucket});
        }

        if (!reals.empty()) {
            std::int64_t start;
            std::int64_t end;
            if (has_range) {
                start = floor_index(range_start, bucket_width);
                end = floor_index(range_end, bucket_width);
            } else {
                start = reals.front().bucket;
                end = reals.back().bucket;
            }
            if (end >= start) {
                // Point count is the index span + 1; compute the span in
                // unsigned arithmetic so a full-domain range cannot overflow
                // the signed difference before the guard fires.
                const std::uint64_t span = static_cast<std::uint64_t>(end) -
                                           static_cast<std::uint64_t>(start);
                if (span >= static_cast<std::uint64_t>(GRID_MAX_POINTS)) {
                    throw DFTUtilsException(
                        ErrorCode::INVALID_ARGUMENT,
                        "gap_fill: grid exceeds 10,000,000 points; narrow the "
                        "range or widen the bucket");
                }
                const std::int64_t rep = reals.front().row;
                std::int64_t j = 0;
                for (std::int64_t k = start; k <= end; ++k) {
                    while (j < static_cast<std::int64_t>(reals.size()) &&
                           reals[static_cast<std::size_t>(j)].bucket < k)
                        ++j;
                    const bool has_real =
                        j < static_cast<std::int64_t>(reals.size()) &&
                        reals[static_cast<std::size_t>(j)].bucket == k;
                    if (has_real) {
                        emit_real(reals[static_cast<std::size_t>(j)].row);
                    } else {
                        const std::int64_t prev_idx = j - 1;
                        const std::int64_t next_idx =
                            j < static_cast<std::int64_t>(reals.size()) ? j
                                                                        : -1;
                        emit_generated(rep, k, prev_idx, next_idx, reals);
                    }
                }
            }
        }
        p = q;
    }

    return b.finish();
}

}  // namespace

ArrowExportResult gap_fill(const ArrowSchema* s, const ArrowArray* a,
                           const std::uint32_t* partition_cols,
                           std::uint32_t n_part, std::uint32_t time_col,
                           std::int64_t bucket_width,
                           const std::uint32_t* value_cols,
                           std::uint32_t n_value, GapFillMode mode,
                           bool has_range, std::int64_t range_start,
                           std::int64_t range_end) {
    return gap_fill_impl<std::int64_t>(s, a, partition_cols, n_part, time_col,
                                       bucket_width, value_cols, n_value, mode,
                                       has_range, range_start, range_end);
}

ArrowExportResult gap_fill(const ArrowSchema* s, const ArrowArray* a,
                           const std::uint32_t* partition_cols,
                           std::uint32_t n_part, std::uint32_t time_col,
                           double bucket_width, const std::uint32_t* value_cols,
                           std::uint32_t n_value, GapFillMode mode,
                           bool has_range, double range_start,
                           double range_end) {
    return gap_fill_impl<double>(s, a, partition_cols, n_part, time_col,
                                 bucket_width, value_cols, n_value, mode,
                                 has_range, range_start, range_end);
}

}  // namespace dftracer::utils::utilities::common::arrow

#endif  // DFTRACER_UTILS_ENABLE_ARROW
