#include <dftracer/utils/core/common/byte_view.h>
#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/core/common/constants.h>
#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/common/field_ref.h>
#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/common/hash/fnv1a.h>
#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/core/common/memory_budget.h>
#include <dftracer/utils/core/coro/when_all.h>
#include <dftracer/utils/core/coro/when_any.h>
#include <dftracer/utils/core/coro/yield.h>
#include <dftracer/utils/core/io/ops.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/agg_expr.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/map_grouping_arrow.h>
#include <dftracer/utils/plugins/map_join_arrow.h>
#include <dftracer/utils/plugins/map_unnest_arrow.h>
#include <dftracer/utils/plugins/utility_registry.h>
#include <dftracer/utils/trace/schema.h>
#include <dftracer/utils/trace/views/event_source.h>
#include <dftracer/utils/trace/views/fold_event.h>
#include <dftracer/utils/trace/views/native_row_fold.h>
#include <dftracer/utils/trace/views/view.h>
#include <dftracer/utils/utilities/common/serialization/binary_codec.h>
#include <dftracer/utils/utilities/common/statistics/ddsketch.h>
#include <dftracer/utils/utilities/fileio/compress/libdeflate_gzip.h>
#include <dftracer/utils/utilities/fileio/parallel/merge.h>
#include <simdjson.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef DFTRACER_UTILS_ENABLE_ARROW
#include <dftracer/utils/utilities/common/arrow/arrow_export.h>
#include <dftracer/utils/utilities/common/arrow/column_builder.h>
#endif
#ifdef DFTRACER_UTILS_ENABLE_ARROW_IPC
#include <dftracer/utils/utilities/common/arrow/ipc_reader.h>
#include <dftracer/utils/utilities/common/arrow/ipc_writer.h>
#endif

#include <dftracer/utils/plugins/fold_adapter/ext.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <numeric>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace dftracer::utils::plugins {
namespace {

namespace codec = utilities::common::serialization;

// SKETCH yields a quantile struct, not a scalar, so it has no scalar column;
// map_new_sketch materializes it to a count column plus one column per
// quantile.
bool monoid_has_scalar(dftu_monoid_kind kind) {
    return kind != DFTU_MONOID_SKETCH;
}

bool monoid_is_quantiles(dftu_monoid_kind kind) {
    return kind == DFTU_MONOID_SKETCH;
}

// Two-variable co-moment monoids fed by map_add_xy_at, not a single u64/f64
// add.
bool monoid_is_xy(dftu_monoid_kind kind) {
    switch (kind) {
        case DFTU_MONOID_CORR:
        case DFTU_MONOID_COVAR_POP:
        case DFTU_MONOID_COVAR_SAMP:
        case DFTU_MONOID_REGR_SLOPE:
        case DFTU_MONOID_REGR_INTERCEPT:
        case DFTU_MONOID_REGR_R2:
            return true;
        default:
            return false;
    }
}

// A component that map_add_row can drive: a scalar monoid fed by a single
// u64/f64 add. Excludes collections, arg-row, sketch, and the two-variable
// co-moment monoids.
bool monoid_is_fused_eligible(dftu_monoid_kind kind) {
    return monoid_has_scalar(kind) && !monoid_is_argrow(kind) &&
           !monoid_is_quantiles(kind) && !monoid_is_xy(kind);
}

// Column label for a requested quantile: p50, p90, p99, p99_9 (dot -> '_').
std::string quantile_column_name(double q) {
    double pct = q * 100.0;
    long long whole = static_cast<long long>(pct + 0.5);
    double diff = pct - static_cast<double>(whole);
    std::string name = "p";
    if (diff < 1e-9 && diff > -1e-9) {
        name += std::to_string(whole);
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.4g", pct);
        for (char* c = buf; *c; ++c) name += (*c == '.') ? '_' : *c;
    }
    return name;
}

bool monoid_is_f64(dftu_monoid_kind kind) {
    return kind == DFTU_MONOID_SUM_F64 || kind == DFTU_MONOID_MIN_F64 ||
           kind == DFTU_MONOID_MAX_F64;
}

bool monoid_is_set(dftu_monoid_kind kind) {
    return kind == DFTU_MONOID_SET_STR || kind == DFTU_MONOID_SET_I64;
}

bool monoid_is_list(dftu_monoid_kind kind) {
    return kind == DFTU_MONOID_LIST_STR || kind == DFTU_MONOID_LIST_I64;
}

// Also the TOPK/BOTTOMK/SAMPLE _STR kinds, which emit their kept ids as
// list<string>.
bool monoid_is_str_collection(dftu_monoid_kind kind) {
    return kind == DFTU_MONOID_SET_STR || kind == DFTU_MONOID_LIST_STR ||
           kind == DFTU_MONOID_TOPK_STR || kind == DFTU_MONOID_BOTTOMK_STR ||
           kind == DFTU_MONOID_SAMPLE_STR;
}

bool monoid_is_i64_collection(dftu_monoid_kind kind) {
    return kind == DFTU_MONOID_SET_I64 || kind == DFTU_MONOID_LIST_I64 ||
           kind == DFTU_MONOID_TOPK_I64 || kind == DFTU_MONOID_BOTTOMK_I64 ||
           kind == DFTU_MONOID_SAMPLE_I64;
}

// The fuse fan-out cap (fold.cpp: min(units, 16)); the budget is split across
// it so peak stays ~budget.
constexpr std::size_t MAP_SPILL_WORKERS = 16;
// Test the budget on a new key or every this-many adds, so the hot path pays a
// counter bump not a comparison.
constexpr std::size_t MAP_SPILL_CHECK_STRIDE = 4096;

// Write one partition to a sorted run file in view_spill's framing (big-endian
// length prefix per record, key ints then MonoidAccumulator::serialize),
// keeping one spill dialect. Throws on any I/O failure.
void write_run(const MapAccum& m, const MapAccum::EntriesMap& part,
               const std::string& path) {
    using Entry = std::pair<const std::vector<std::int64_t>,
                            std::vector<MonoidAccumulator>>;
    std::vector<const Entry*> ents;
    ents.reserve(part.size());
    for (const auto& kv : part) ents.push_back(&kv);
    std::sort(ents.begin(), ents.end(), [](const Entry* a, const Entry* b) {
        return a->first < b->first;
    });
    std::ofstream os(path, std::ios::binary);
    if (!os)
        throw dftracer::utils::DFTUtilsException(
            dftracer::utils::ErrorCode::IO,
            "plugin map spill: cannot open run file " + path);
    std::string rec, hdr;
    for (const Entry* e : ents) {
        rec.clear();
        for (std::uint32_t i = 0; i < m.key_n; ++i)
            codec::put_be64(rec, static_cast<std::uint64_t>(e->first[i]));
        for (const MonoidAccumulator& mon : e->second) mon.serialize(rec);
        hdr.clear();
        codec::put_be64(hdr, rec.size());
        os.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
        os.write(rec.data(), static_cast<std::streamsize>(rec.size()));
    }
    os.flush();
    if (!os)
        throw dftracer::utils::DFTUtilsException(
            dftracer::utils::ErrorCode::IO,
            "plugin map spill: write failed (disk full?) " + path);
}

// Stream one run back, merging each partial into m by key. partition_of(key) is
// deterministic, so a key spilled from partition p reloads into partition p.
void read_run_into(MapAccum& m, const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is)
        throw dftracer::utils::DFTUtilsException(
            dftracer::utils::ErrorCode::IO,
            "plugin map spill: cannot reopen run " + path);
    std::string buf;
    char hdr[8];
    while (is.read(hdr, 8)) {
        std::uint64_t len = 0;
        for (int i = 0; i < 8; ++i)
            len = (len << 8) | static_cast<std::uint8_t>(hdr[i]);
        buf.resize(len);
        if (len && !is.read(buf.data(), static_cast<std::streamsize>(len)))
            throw dftracer::utils::DFTUtilsException(
                dftracer::utils::ErrorCode::IO,
                "plugin map spill: truncated run " + path);
        codec::BinaryReader br(buf);
        std::vector<std::int64_t> key(m.key_n);
        for (std::uint32_t i = 0; i < m.key_n; ++i)
            key[i] = static_cast<std::int64_t>(br.be64());
        std::string_view rest = br.remaining();
        const std::byte* p = reinterpret_cast<const std::byte*>(rest.data());
        std::size_t n = rest.size();
        std::vector<MonoidAccumulator>& into = m.touch(key);
        for (std::size_t c = 0; c < m.value_kinds.size(); ++c) {
            MonoidAccumulator mon(m.value_kinds[c]);
            std::size_t used = mon.deserialize(p, n);
            if (used == 0)
                throw dftracer::utils::DFTUtilsException(
                    dftracer::utils::ErrorCode::PARSE,
                    "plugin map spill: corrupt run record in " + path);
            p += used;
            n -= used;
            if (c < into.size()) into[c].merge(mon);
        }
    }
}

// Fixed-width integer, float, STR, or BYTES; other ABI types are rejected at
// map creation.
bool key_type_supported(dftu_type t) {
    switch (t) {
        case DFTU_T_I8:
        case DFTU_T_I16:
        case DFTU_T_I32:
        case DFTU_T_I64:
        case DFTU_T_U8:
        case DFTU_T_U16:
        case DFTU_T_U32:
        case DFTU_T_U64:
        case DFTU_T_F32:
        case DFTU_T_F64:
        case DFTU_T_STR:
        case DFTU_T_BYTES:
            return true;
        default:
            return false;
    }
}

// STR and BYTES both ride the int64 key slot as an interned dftu_str id,
// differing only in the materialized column type (utf8 vs binary); every other
// path treats the slot as a plain int64 id.
bool key_type_interned(dftu_type t) {
    return t == DFTU_T_STR || t == DFTU_T_BYTES;
}

bool key_type_unsigned(dftu_type t) {
    return t == DFTU_T_U8 || t == DFTU_T_U16 || t == DFTU_T_U32 ||
           t == DFTU_T_U64;
}

bool key_type_float(dftu_type t) { return t == DFTU_T_F32 || t == DFTU_T_F64; }

// Recover a float key component bit-cast into the int64 slot (f32 from the low
// 32 bits, f64 from all 64).
double key_bits_to_double(dftu_type t, std::int64_t bits) {
    if (t == DFTU_T_F32) {
        std::uint32_t lo =
            static_cast<std::uint32_t>(static_cast<std::uint64_t>(bits));
        float f;
        std::memcpy(&f, &lo, sizeof(f));
        return static_cast<double>(f);
    }
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW
utilities::common::arrow::ColumnType key_column_type(dftu_type t) {
    namespace arr = utilities::common::arrow;
    switch (t) {
        case DFTU_T_STR:
            return arr::ColumnType::STRING;
        case DFTU_T_BYTES:
            return arr::ColumnType::BINARY;
        case DFTU_T_I8:
            return arr::ColumnType::INT8;
        case DFTU_T_I16:
            return arr::ColumnType::INT16;
        case DFTU_T_I32:
            return arr::ColumnType::INT32;
        case DFTU_T_U8:
            return arr::ColumnType::UINT8;
        case DFTU_T_U16:
            return arr::ColumnType::UINT16;
        case DFTU_T_U32:
            return arr::ColumnType::UINT32;
        case DFTU_T_U64:
            return arr::ColumnType::UINT64;
        case DFTU_T_F32:
            return arr::ColumnType::FLOAT32;
        case DFTU_T_F64:
            return arr::ColumnType::DOUBLE;
        default:
            return arr::ColumnType::INT64;
    }
}

// Append a non-STR key component, recovering unsigned and float values from the
// int64 slot bits.
void append_scalar_key(utilities::common::arrow::RecordBatchBuilder& builder,
                       std::uint32_t col, dftu_type t, std::int64_t bits) {
    if (key_type_float(t))
        builder.append_double(col, key_bits_to_double(t, bits));
    else if (key_type_unsigned(t))
        builder.append_uint64(col, static_cast<std::uint64_t>(bits));
    else
        builder.append_int64(col, bits);
}

// Typed min/max materialize at their element width; MIN/MAX_U64 keep the
// historical int64 column, f64-result monoids double, every other scalar int64.
utilities::common::arrow::ColumnType monoid_value_column_type(
    dftu_monoid_kind k) {
    namespace arr = utilities::common::arrow;
    switch (k) {
        case DFTU_MONOID_MIN_I8:
        case DFTU_MONOID_MAX_I8:
            return arr::ColumnType::INT8;
        case DFTU_MONOID_MIN_I16:
        case DFTU_MONOID_MAX_I16:
            return arr::ColumnType::INT16;
        case DFTU_MONOID_MIN_I32:
        case DFTU_MONOID_MAX_I32:
            return arr::ColumnType::INT32;
        case DFTU_MONOID_MIN_U8:
        case DFTU_MONOID_MAX_U8:
            return arr::ColumnType::UINT8;
        case DFTU_MONOID_MIN_U16:
        case DFTU_MONOID_MAX_U16:
            return arr::ColumnType::UINT16;
        case DFTU_MONOID_MIN_U32:
        case DFTU_MONOID_MAX_U32:
            return arr::ColumnType::UINT32;
        case DFTU_MONOID_MIN_F32:
        case DFTU_MONOID_MAX_F32:
            return arr::ColumnType::FLOAT32;
        case DFTU_MONOID_MEAN:
        case DFTU_MONOID_VARIANCE:
        case DFTU_MONOID_STDDEV:
        case DFTU_MONOID_SKEWNESS:
        case DFTU_MONOID_KURTOSIS:
        case DFTU_MONOID_CORR:
        case DFTU_MONOID_COVAR_POP:
        case DFTU_MONOID_COVAR_SAMP:
        case DFTU_MONOID_REGR_SLOPE:
        case DFTU_MONOID_REGR_INTERCEPT:
        case DFTU_MONOID_REGR_R2:
            return arr::ColumnType::DOUBLE;
        case DFTU_MONOID_ARGMIN_I64:
        case DFTU_MONOID_ARGMAX_I64:
            return arr::ColumnType::INT64;
        case DFTU_MONOID_ARGMIN_STR:
        case DFTU_MONOID_ARGMAX_STR:
            return arr::ColumnType::STRING;
        default:
            return monoid_is_f64(k) ? arr::ColumnType::DOUBLE
                                    : arr::ColumnType::INT64;
    }
}
#endif

#ifdef DFTRACER_UTILS_ENABLE_ARROW
// Shared column emission for the map materializers. materialize_map passes an
// empty prefix for byte-identical output; materialize_joined disambiguates its
// two value sides with "l_"/"r_".

// STR resolves to utf8, BYTES to binary, every other width to
// append_scalar_key.
void append_key_cell(utilities::common::arrow::RecordBatchBuilder& builder,
                     std::uint32_t col, dftu_type t, std::int64_t bits,
                     dftracer::utils::StringIntern& intern) {
    if (t == DFTU_T_STR)
        builder.append_string(col,
                              resolve_id(intern, static_cast<dftu_str>(bits)));
    else if (t == DFTU_T_BYTES)
        builder.append_binary(col,
                              resolve_id(intern, static_cast<dftu_str>(bits)));
    else
        append_scalar_key(builder, col, t, bits);
}

void append_key_specs(std::vector<utilities::common::arrow::ColumnSpec>& specs,
                      const std::vector<dftu_type>& key_types,
                      std::uint32_t key_n) {
    for (std::uint32_t i = 0; i < key_n; ++i)
        specs.push_back(
            {"k" + std::to_string(i), key_column_type(key_types[i])});
}

// An arg-row spreads across its payload columns and a sketch across its count +
// quantile columns; every other monoid is one.
std::uint32_t value_column_count(const std::vector<dftu_monoid_kind>& kinds,
                                 const std::vector<dftu_type>& payload_types,
                                 const std::vector<double>& quantile_qs,
                                 int only_comp = -1) {
    if (only_comp >= 0) return 1;  // a fused map emits one scalar per table
    std::uint32_t n = 0;
    for (dftu_monoid_kind k : kinds) {
        if (monoid_is_argrow(k))
            n += static_cast<std::uint32_t>(payload_types.size());
        else if (monoid_is_quantiles(k))
            n += 1u + static_cast<std::uint32_t>(quantile_qs.size());
        else
            n += 1u;
    }
    return n;
}

// Append the Arrow column specs for a flat value schema. `prefix` disambiguates
// a join's left/right value columns; "" reproduces materialize_map's naming.
void append_value_specs(
    std::vector<utilities::common::arrow::ColumnSpec>& specs,
    const std::vector<dftu_monoid_kind>& kinds,
    const std::vector<dftu_type>& payload_types,
    const std::vector<double>& quantile_qs, const std::string& prefix,
    int only_comp = -1) {
    namespace arr = utilities::common::arrow;
    if (only_comp >= 0) {  // a fused component: one scalar column named "value"
        specs.push_back(
            {prefix + "value", monoid_value_column_type(kinds[only_comp])});
        return;
    }
    const std::uint32_t value_n = static_cast<std::uint32_t>(kinds.size());
    for (std::uint32_t c = 0; c < value_n; ++c) {
        // arg-row expands to one column per payload component (p0..).
        if (monoid_is_argrow(kinds[c])) {
            for (std::uint32_t p = 0; p < payload_types.size(); ++p)
                specs.push_back({prefix + "p" + std::to_string(p),
                                 key_column_type(payload_types[p])});
            continue;
        }
        // sketch expands to a count column plus one column per quantile.
        if (monoid_is_quantiles(kinds[c])) {
            specs.push_back({prefix + "count", arr::ColumnType::INT64});
            for (double q : quantile_qs)
                specs.push_back({prefix + quantile_column_name(q),
                                 arr::ColumnType::DOUBLE});
            continue;
        }
        std::string col_name =
            prefix +
            (value_n == 1 ? std::string("value") : "v" + std::to_string(c));
        // APPROX_TOPK materializes to a list<struct<value, count>> column.
        if (monoid_is_approx_topk(kinds[c])) {
            arr::ColumnSpec spec;
            spec.name = std::move(col_name);
            spec.type = arr::ColumnType::STRUCT_LIST;
            spec.fields.push_back({"value",
                                   monoid_approx_topk_is_str(kinds[c])
                                       ? arr::ColumnType::STRING
                                       : arr::ColumnType::INT64,
                                   {}});
            spec.fields.push_back({"count", arr::ColumnType::INT64, {}});
            specs.push_back(std::move(spec));
            continue;
        }
        arr::ColumnType ct = arr::ColumnType::INT64;
        if (monoid_is_str_collection(kinds[c]))
            ct = arr::ColumnType::STRING_LIST;
        else if (monoid_is_i64_collection(kinds[c]))
            ct = arr::ColumnType::INT64_LIST;
        else
            ct = monoid_value_column_type(kinds[c]);
        specs.push_back({std::move(col_name), ct});
    }
}

// Emit one scalar monoid value cell at `col`.
void append_scalar_value(utilities::common::arrow::RecordBatchBuilder& builder,
                         std::uint32_t col, const MonoidAccumulator& mon,
                         dftu_monoid_kind kind,
                         dftracer::utils::StringIntern& intern) {
    namespace arr = utilities::common::arrow;
    dftu_monoid_value v = mon.to_value();
    switch (monoid_value_column_type(kind)) {
        case arr::ColumnType::FLOAT32:
        case arr::ColumnType::DOUBLE:
            builder.append_double(col, v.as.f64);
            break;
        case arr::ColumnType::UINT8:
        case arr::ColumnType::UINT16:
        case arr::ColumnType::UINT32:
            builder.append_uint64(col, v.as.u64);
            break;
        case arr::ColumnType::STRING:
            builder.append_string(
                col, resolve_id(intern, static_cast<dftu_str>(v.as.u64)));
            break;
        default:
            builder.append_int64(col, static_cast<std::int64_t>(v.as.u64));
            break;
    }
}

// Append the value cells of one row starting at `first_col`, one per flat value
// column. `mons` are this side's Monoids; `kinds`/`payload_types` its schema.
void append_value_cells(utilities::common::arrow::RecordBatchBuilder& builder,
                        std::uint32_t first_col,
                        const std::vector<MonoidAccumulator>& mons,
                        const std::vector<dftu_monoid_kind>& kinds,
                        const std::vector<dftu_type>& payload_types,
                        const std::vector<double>& quantile_qs,
                        dftracer::utils::StringIntern& intern,
                        int only_comp = -1) {
    namespace arr = utilities::common::arrow;
    if (only_comp >= 0) {  // a fused component: just its scalar at first_col
        append_scalar_value(builder, first_col, mons[only_comp],
                            kinds[only_comp], intern);
        return;
    }
    const std::uint32_t value_n = static_cast<std::uint32_t>(kinds.size());
    for (std::uint32_t c = 0; c < value_n; ++c) {
        // A never-added arg-row nulls every payload column.
        if (monoid_is_argrow(kinds[c])) {
            const MonoidAccumulator& mon = mons[c];
            const bool has = mon.argrow_has();
            const std::vector<std::int64_t>& pl = mon.argrow_payload();
            for (std::uint32_t p = 0; p < payload_types.size(); ++p) {
                const std::uint32_t col = first_col + c + p;
                if (!has) {
                    builder.append_null(col);
                    continue;
                }
                append_key_cell(builder, col, payload_types[p], pl[p], intern);
            }
            continue;
        }
        // sketch emits count, then the value at each requested quantile.
        if (monoid_is_quantiles(kinds[c])) {
            const MonoidAccumulator& mon = mons[c];
            std::uint32_t col = first_col + c;
            builder.append_int64(col++,
                                 static_cast<std::int64_t>(mon.sketch_count()));
            for (double q : quantile_qs)
                builder.append_double(col++, mon.sketch_quantile(q));
            continue;
        }
        if (monoid_is_approx_topk(kinds[c])) {
            const bool is_str = monoid_approx_topk_is_str(kinds[c]);
            std::vector<std::vector<arr::StructCell>> structs;
            for (const auto& [value, count] : mons[c].approx_topk_entries()) {
                std::vector<arr::StructCell> cells(2);
                if (is_str)
                    cells[0].str =
                        resolve_id(intern, static_cast<dftu_str>(value));
                else
                    cells[0].i64 = value;
                cells[1].i64 = static_cast<std::int64_t>(count);
                structs.push_back(std::move(cells));
            }
            builder.append_struct_list(first_col + c, structs);
            continue;
        }
        if (monoid_is_str_collection(kinds[c])) {
            std::vector<std::string_view> labels;
            auto emit_id = [&](std::int64_t id) {
                std::string_view s =
                    resolve_id(intern, static_cast<dftu_str>(id));
                if (!s.empty()) labels.push_back(s);
            };
            if (monoid_is_topk(kinds[c])) {
                for (std::int64_t id : mons[c].topk_elements()) emit_id(id);
            } else if (monoid_is_list(kinds[c])) {
                for (std::int64_t id : mons[c].ordered_elements()) emit_id(id);
            } else if (monoid_is_sample(kinds[c])) {
                for (std::int64_t id : mons[c].sample_items()) emit_id(id);
                std::sort(labels.begin(), labels.end());
            } else {
                for (std::int64_t id : mons[c].elements()) emit_id(id);
                std::sort(labels.begin(), labels.end());
            }
            builder.append_string_list(first_col + c, labels);
            continue;
        }
        if (monoid_is_i64_collection(kinds[c])) {
            std::vector<std::int64_t> vals;
            if (monoid_is_topk(kinds[c]))
                vals = mons[c].topk_elements();
            else if (monoid_is_list(kinds[c]))
                vals = mons[c].ordered_elements();
            else if (monoid_is_sample(kinds[c]))
                vals = mons[c].sample_items();
            else
                vals = mons[c].sorted_elements();
            builder.append_int64_list(first_col + c, vals);
            continue;
        }
        append_scalar_value(builder, first_col + c, mons[c], kinds[c], intern);
    }
}

// Null every value column of a flat schema (an outer-join row whose side had no
// key); append_null covers scalar, list, struct, and arg-row payload columns.
void null_value_cells(utilities::common::arrow::RecordBatchBuilder& builder,
                      std::uint32_t first_col,
                      const std::vector<dftu_monoid_kind>& kinds,
                      const std::vector<dftu_type>& payload_types,
                      const std::vector<double>& quantile_qs) {
    const std::uint32_t n =
        value_column_count(kinds, payload_types, quantile_qs);
    for (std::uint32_t i = 0; i < n; ++i) builder.append_null(first_col + i);
}
#endif

::dftu_map* host_map_new(void* h, const char* name, const dftu_type* key_types,
                         std::uint32_t key_n, dftu_monoid_kind value) {
    return reinterpret_cast<::dftu_map*>(
        static_cast<PluginFold*>(h)->map_get(name, key_types, key_n, value));
}

void host_map_add_u64(void* h, ::dftu_map* m, const std::int64_t* key,
                      std::uint64_t v) {
    static_cast<PluginFold*>(h)->map_add_u64(reinterpret_cast<MapAccum*>(m),
                                             key, v);
}

void host_map_add_f64(void* h, ::dftu_map* m, const std::int64_t* key,
                      double v) {
    static_cast<PluginFold*>(h)->map_add_f64(reinterpret_cast<MapAccum*>(m),
                                             key, v);
}

::dftu_map* host_map_new_product(void* h, const char* name,
                                 const dftu_type* key_types,
                                 std::uint32_t key_n,
                                 const dftu_monoid_kind* values,
                                 std::uint32_t value_n) {
    return reinterpret_cast<::dftu_map*>(
        static_cast<PluginFold*>(h)->map_get_product(name, key_types, key_n,
                                                     values, value_n));
}

void host_map_add_u64_at(void* h, ::dftu_map* m, const std::int64_t* key,
                         std::uint32_t comp, std::uint64_t v) {
    static_cast<PluginFold*>(h)->map_add_u64_at(reinterpret_cast<MapAccum*>(m),
                                                key, comp, v);
}

void host_map_add_f64_at(void* h, ::dftu_map* m, const std::int64_t* key,
                         std::uint32_t comp, double v) {
    static_cast<PluginFold*>(h)->map_add_f64_at(reinterpret_cast<MapAccum*>(m),
                                                key, comp, v);
}

void host_map_add_ordered_at(void* h, ::dftu_map* m, const std::int64_t* key,
                             std::uint32_t comp, std::int64_t order_key,
                             std::uint64_t element) {
    static_cast<PluginFold*>(h)->map_add_ordered_at(
        reinterpret_cast<MapAccum*>(m), key, comp, order_key, element);
}

void host_map_set_ordered(void* h, ::dftu_map* m, int ordered) {
    static_cast<PluginFold*>(h)->map_set_ordered(reinterpret_cast<MapAccum*>(m),
                                                 ordered);
}

::dftu_map* host_map_new_nested(void* h, const char* name,
                                const dftu_type* outer_key_types,
                                std::uint32_t outer_key_n,
                                const dftu_type* inner_key_types,
                                std::uint32_t inner_key_n,
                                const dftu_monoid_kind* values,
                                std::uint32_t value_n) {
    return reinterpret_cast<::dftu_map*>(
        static_cast<PluginFold*>(h)->map_get_nested(
            name, outer_key_types, outer_key_n, inner_key_types, inner_key_n,
            values, value_n));
}

void host_map_add_nested_u64(void* h, ::dftu_map* m,
                             const std::int64_t* outer_key,
                             const std::int64_t* inner_key, std::uint32_t comp,
                             std::uint64_t v) {
    static_cast<PluginFold*>(h)->map_add_nested_u64(
        reinterpret_cast<MapAccum*>(m), outer_key, inner_key, comp, v);
}

void host_map_add_nested_f64(void* h, ::dftu_map* m,
                             const std::int64_t* outer_key,
                             const std::int64_t* inner_key, std::uint32_t comp,
                             double v) {
    static_cast<PluginFold*>(h)->map_add_nested_f64(
        reinterpret_cast<MapAccum*>(m), outer_key, inner_key, comp, v);
}

void host_map_add_argby_at(void* h, ::dftu_map* m, const std::int64_t* key,
                           std::uint32_t comp, double by,
                           std::int64_t payload) {
    static_cast<PluginFold*>(h)->map_add_argby_at(
        reinterpret_cast<MapAccum*>(m), key, comp, by, payload);
}

void host_map_add_topk_at(void* h, ::dftu_map* m, const std::int64_t* key,
                          std::uint32_t comp, std::uint32_t k, double by,
                          std::int64_t payload) {
    static_cast<PluginFold*>(h)->map_add_topk_at(reinterpret_cast<MapAccum*>(m),
                                                 key, comp, k, by, payload);
}

void host_map_add_approx_topk_at(void* h, ::dftu_map* m,
                                 const std::int64_t* key, std::uint32_t comp,
                                 std::uint32_t k, std::int64_t value) {
    static_cast<PluginFold*>(h)->map_add_approx_topk_at(
        reinterpret_cast<MapAccum*>(m), key, comp, k, value);
}

void host_map_add_sample_at(void* h, ::dftu_map* m, const std::int64_t* key,
                            std::uint32_t comp, std::uint32_t k,
                            std::int64_t item) {
    static_cast<PluginFold*>(h)->map_add_sample_at(
        reinterpret_cast<MapAccum*>(m), key, comp, k, item);
}

::dftu_map* host_map_new_argrow(void* h, const char* name,
                                const dftu_type* key_types, std::uint32_t key_n,
                                int is_max, const dftu_type* payload_types,
                                std::uint32_t payload_n) {
    return reinterpret_cast<::dftu_map*>(
        static_cast<PluginFold*>(h)->map_get_argrow(
            name, key_types, key_n, is_max, payload_types, payload_n));
}

void host_map_add_argrow(void* h, ::dftu_map* m, const std::int64_t* key,
                         double by, const std::int64_t* payload,
                         std::uint32_t payload_n) {
    static_cast<PluginFold*>(h)->map_add_argrow(reinterpret_cast<MapAccum*>(m),
                                                key, by, payload, payload_n);
}

void host_map_declare_join(void* h, const char* out_name, const char* left_name,
                           const char* right_name, dftu_join_type type) {
    static_cast<PluginFold*>(h)->declare_join(out_name, left_name, right_name,
                                              type);
}

void host_map_add_xy_at(void* h, ::dftu_map* m, const std::int64_t* key,
                        std::uint32_t comp, double x, double y) {
    static_cast<PluginFold*>(h)->map_add_xy_at(reinterpret_cast<MapAccum*>(m),
                                               key, comp, x, y);
}

::dftu_map* host_map_new_sketch(void* h, const char* name,
                                const dftu_type* key_types, std::uint32_t key_n,
                                const double* qs, std::uint32_t nq) {
    return reinterpret_cast<::dftu_map*>(
        static_cast<PluginFold*>(h)->map_get_sketch(name, key_types, key_n, qs,
                                                    nq));
}

::dftu_map* host_map_new_fused(void* h, const char* name,
                               const dftu_type* key_types, std::uint32_t key_n,
                               const char* const* out_names,
                               const dftu_monoid_kind* values,
                               std::uint32_t value_n) {
    return reinterpret_cast<::dftu_map*>(
        static_cast<PluginFold*>(h)->map_get_fused(name, key_types, key_n,
                                                   out_names, values, value_n));
}

void host_map_add_row(void* h, ::dftu_map* m, const std::int64_t* key,
                      const ::dftu_row_val* vals, std::uint32_t n) {
    static_cast<PluginFold*>(h)->map_add_row(reinterpret_cast<MapAccum*>(m),
                                             key, vals, n);
}

// Iterator state for map_iter_*; a flat walk across MapAccum's radix
// partitions, since entries live in one EntriesMap per partition.
struct MapCursor {
    MapAccum* m;
    std::uint32_t part = 0;
    MapAccum::EntriesMap::const_iterator it{};

    explicit MapCursor(MapAccum* map) : m(map) {
        if (!m->parts.empty()) it = m->parts[0].begin();
    }
};

std::uint64_t host_map_size(void* h, ::dftu_map* m) {
    return static_cast<PluginFold*>(h)->map_size(
        reinterpret_cast<MapAccum*>(m));
}

int host_map_get(void* h, ::dftu_map* m, const std::int64_t* key,
                 std::uint32_t key_n, std::uint32_t comp,
                 dftu_monoid_value* out) {
    return static_cast<PluginFold*>(h)->map_lookup(
        reinterpret_cast<MapAccum*>(m), key, key_n, comp, out);
}

::dftu_map_cursor* host_map_iter_new(void* h, ::dftu_map* m) {
    return reinterpret_cast<::dftu_map_cursor*>(
        static_cast<PluginFold*>(h)->map_iter_new(
            reinterpret_cast<MapAccum*>(m)));
}

int host_map_iter_next(::dftu_map_cursor* cur_, std::int64_t* key_out,
                       std::uint32_t key_cap, std::uint32_t* key_n_out,
                       dftu_monoid_value* vals_out, std::uint32_t val_cap,
                       std::uint32_t* val_n_out) {
    auto* cur = reinterpret_cast<MapCursor*>(cur_);
    MapAccum* m = cur->m;
    while (cur->part < m->parts.size() &&
           cur->it == m->parts[cur->part].end()) {
        ++cur->part;
        if (cur->part < m->parts.size()) cur->it = m->parts[cur->part].begin();
    }
    if (cur->part >= m->parts.size()) return 0;
    const auto& kv = *cur->it;
    const std::uint32_t kn = static_cast<std::uint32_t>(kv.first.size());
    if (key_n_out) *key_n_out = kn;
    const std::uint32_t kcopy = std::min(kn, key_cap);
    for (std::uint32_t i = 0; i < kcopy; ++i) key_out[i] = kv.first[i];
    const std::uint32_t vn = static_cast<std::uint32_t>(kv.second.size());
    if (val_n_out) *val_n_out = vn;
    const std::uint32_t vcopy = std::min(vn, val_cap);
    for (std::uint32_t i = 0; i < vcopy; ++i)
        vals_out[i] = kv.second[i].to_value();
    ++cur->it;
    return 1;
}

void host_map_iter_free(::dftu_map_cursor* cur_) {
    delete reinterpret_cast<MapCursor*>(cur_);
}

const dftu_ext_map g_map = {host_map_new,
                            host_map_add_u64,
                            host_map_add_f64,
                            host_map_new_product,
                            host_map_add_u64_at,
                            host_map_add_f64_at,
                            host_map_add_ordered_at,
                            host_map_set_ordered,
                            host_map_new_nested,
                            host_map_add_nested_u64,
                            host_map_add_nested_f64,
                            host_map_add_argby_at,
                            host_map_add_topk_at,
                            host_map_add_approx_topk_at,
                            host_map_add_sample_at,
                            host_map_new_argrow,
                            host_map_add_argrow,
                            host_map_declare_join,
                            host_map_add_xy_at,
                            host_map_new_sketch,
                            host_map_new_fused,
                            host_map_add_row,
                            host_map_size,
                            host_map_get,
                            host_map_iter_new,
                            host_map_iter_next,
                            host_map_iter_free};

}  // namespace

const void* detail::map_ext_vtable() { return &g_map; }

// State grows per element (bounded to k for TOPK/BOTTOMK), so the footprint
// counter bumps on every add, not only on a new-key insert.
bool monoid_is_variable(dftu_monoid_kind kind) {
    return monoid_is_set(kind) || monoid_is_list(kind) ||
           monoid_is_topk(kind) || monoid_is_approx_topk(kind) ||
           monoid_is_sample(kind);
}

MapSpillEnv read_map_spill_env() {
    MapSpillEnv c;
    if (const char* s = std::getenv("DFTRACER_PLUGIN_MAP_STREAM");
        s && std::atoi(s) != 0)
        c.stream = true;
    std::size_t budget = 0;
    if (const char* b = std::getenv("DFTRACER_PLUGIN_MAP_MEM_BUDGET")) {
        budget = static_cast<std::size_t>(std::strtoull(b, nullptr, 10));
    } else if (const char* a = std::getenv("DFTRACER_PLUGIN_MAP_AUTO_SPILL");
               a && std::atoi(a) != 0) {
        budget = dftracer::utils::compute_memory_budget(0);
    }
    if (budget == 0) return c;
    c.enabled = true;
    c.share = std::max<std::size_t>(1, budget / MAP_SPILL_WORKERS);
    if (const char* d = std::getenv("DFTRACER_PLUGIN_MAP_SPILL_DIR"); d && *d)
        c.dir = d;
    return c;
}

void PluginFold::account_add(MapAccum& m, const std::vector<std::int64_t>& key,
                             std::uint32_t comp, bool inserted) {
    if (!map_spill_enabled_) return;
    const std::uint32_t p = m.partition_of(key);
    if (p >= m.part_bytes.size()) return;
    std::size_t delta = 0;
    if (inserted)
        delta += MapAccum::KEY_OVERHEAD_BYTES + sizeof(std::int64_t) * m.key_n +
                 m.value_base_bytes;
    if (comp < m.value_kinds.size() && monoid_is_variable(m.value_kinds[comp]))
        delta += 16;
    m.part_bytes[p] += delta;
    m.footprint += delta;
}

void PluginFold::spill_partition(MapAccum& m, std::uint32_t p) {
    if (p >= m.parts.size() || m.parts[p].empty()) return;
    const std::string& dir = ensure_spill_dir();
    std::string path = dir + "/" + std::to_string(map_spill_run_seq_++);
    write_run(m, m.parts[p], path);
    m.runs[p].push_back(std::move(path));
    m.footprint -= std::min(m.footprint, m.part_bytes[p]);
    m.part_bytes[p] = 0;
    m.parts[p].clear();
    ++map_spill_count_;
}

void PluginFold::note_and_maybe_spill(MapAccum& m, bool inserted) {
    if (!map_spill_enabled_ || map_spill_failed_) return;
    if (!inserted && ++m.adds_since_check < MAP_SPILL_CHECK_STRIDE) return;
    m.adds_since_check = 0;
    if (m.footprint <= map_spill_share_) return;
    // Hysteresis: drain the largest partitions down to a low-water mark (~75%
    // of the share) so one more add does not immediately re-trigger a spill.
    const std::size_t low = map_spill_share_ - map_spill_share_ / 4;
    try {
        while (m.footprint > low) {
            std::uint32_t best = m.partitions();
            std::size_t best_bytes = 0;
            for (std::uint32_t p = 0; p < m.parts.size(); ++p)
                if (!m.parts[p].empty() && m.part_bytes[p] >= best_bytes) {
                    best_bytes = m.part_bytes[p];
                    best = p;
                }
            if (best == m.partitions()) break;
            spill_partition(m, best);
        }
    } catch (const std::exception& e) {
        // Fail the run loudly rather than emit a partial result; materialize
        // refuses to emit after this.
        map_spill_failed_ = true;
        DFTRACER_UTILS_LOG_ERROR("Plugin map '%s' spill failed: %s",
                                 m.name.c_str(), e.what());
    }
}

void PluginFold::reload_runs(MapAccum& m) {
    if (!m.has_runs()) return;
    for (std::uint32_t p = 0; p < m.runs.size(); ++p) {
        for (const std::string& path : m.runs[p]) read_run_into(m, path);
        m.runs[p].clear();
    }
}

MapAccum* PluginFold::map_get(const char* name, const dftu_type* key_types,
                              std::uint32_t key_n, dftu_monoid_kind value) {
    return map_get_product(name, key_types, key_n, &value, 1);
}

MapAccum* PluginFold::map_get_product(const char* name,
                                      const dftu_type* key_types,
                                      std::uint32_t key_n,
                                      const dftu_monoid_kind* values,
                                      std::uint32_t value_n) {
    if (!name || (key_n && !key_types) || value_n == 0 || !values)
        return nullptr;
    for (std::uint32_t i = 0; i < key_n; ++i)
        if (!key_type_supported(key_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin map '%s' key component %u must be a fixed-width "
                "integer "
                "(I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    for (std::uint32_t i = 0; i < value_n; ++i) {
        // ARGMIN_ROW/ARGMAX_ROW are whole-value, created only via
        // map_new_argrow, never a product component.
        if (monoid_is_argrow(values[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin map '%s' value component %u is an arg-row monoid; "
                "create it via map_new_argrow, not as a product component",
                name, i);
            return nullptr;
        }
        if (!monoid_has_scalar(values[i]) && !monoid_is_set(values[i]) &&
            !monoid_is_list(values[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin map '%s' value component %u is not materializable "
                "(SKETCH)",
                name, i);
            return nullptr;
        }
    }
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(name);
    return maps_.get_or_create(key, [&] {
        MapAccum m;
        m.name = name;
        m.key_n = key_n;
        m.key_types.assign(key_types, key_types + key_n);
        m.value_kinds.assign(values, values + value_n);
        for (std::uint32_t i = 0; i < value_n; ++i)
            m.value_base_bytes += MonoidAccumulator(values[i]).state_bytes();
        m.set_part_bits(map_part_bits_);
        return m;
    });
}

MapAccum* PluginFold::map_get_nested(const char* name,
                                     const dftu_type* outer_key_types,
                                     std::uint32_t outer_key_n,
                                     const dftu_type* inner_key_types,
                                     std::uint32_t inner_key_n,
                                     const dftu_monoid_kind* values,
                                     std::uint32_t value_n) {
    if (!name || outer_key_n == 0 || inner_key_n == 0 || !outer_key_types ||
        !inner_key_types || value_n == 0 || !values)
        return nullptr;
    for (std::uint32_t i = 0; i < outer_key_n; ++i)
        if (!key_type_supported(outer_key_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin nested map '%s' outer key component %u must be a "
                "fixed-width integer (I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    for (std::uint32_t i = 0; i < inner_key_n; ++i)
        if (!key_type_supported(inner_key_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin nested map '%s' inner key component %u must be a "
                "fixed-width integer (I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    for (std::uint32_t i = 0; i < value_n; ++i)
        if (!monoid_has_scalar(values[i]) || monoid_is_set(values[i]) ||
            monoid_is_list(values[i]) || monoid_is_topk(values[i]) ||
            monoid_is_approx_topk(values[i]) || monoid_is_sample(values[i]) ||
            monoid_is_argrow(values[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin nested map '%s' inner value component %u must be a "
                "scalar or product monoid (collections, ordered lists, top-k, "
                "approx-top-k, sample, arg-row and SKETCH are not supported "
                "this phase)",
                name, i);
            return nullptr;
        }
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(name);
    return maps_.get_or_create(key, [&] {
        MapAccum m;
        m.name = name;
        m.key_n = outer_key_n + inner_key_n;
        m.key_types.assign(outer_key_types, outer_key_types + outer_key_n);
        m.key_types.insert(m.key_types.end(), inner_key_types,
                           inner_key_types + inner_key_n);
        m.value_kinds.assign(values, values + value_n);
        for (std::uint32_t i = 0; i < value_n; ++i)
            m.value_base_bytes += MonoidAccumulator(values[i]).state_bytes();
        m.nested_inner_n = inner_key_n;
        m.inner_key_types.assign(inner_key_types,
                                 inner_key_types + inner_key_n);
        m.set_part_bits(map_part_bits_);
        return m;
    });
}

void PluginFold::map_add_nested_u64(MapAccum* m, const std::int64_t* outer_key,
                                    const std::int64_t* inner_key,
                                    std::uint32_t comp, std::uint64_t v) {
    if (!m || m->nested_inner_n == 0 || !outer_key || !inner_key) return;
    try {
        const std::uint32_t outer_n = m->key_n - m->nested_inner_n;
        std::vector<std::int64_t> key;
        key.reserve(m->key_n);
        key.insert(key.end(), outer_key, outer_key + outer_n);
        key.insert(key.end(), inner_key, inner_key + m->nested_inner_n);
        map_add_u64_at(m, key.data(), comp, v);
    } catch (...) {
    }
}

void PluginFold::map_add_nested_f64(MapAccum* m, const std::int64_t* outer_key,
                                    const std::int64_t* inner_key,
                                    std::uint32_t comp, double v) {
    if (!m || m->nested_inner_n == 0 || !outer_key || !inner_key) return;
    try {
        const std::uint32_t outer_n = m->key_n - m->nested_inner_n;
        std::vector<std::int64_t> key;
        key.reserve(m->key_n);
        key.insert(key.end(), outer_key, outer_key + outer_n);
        key.insert(key.end(), inner_key, inner_key + m->nested_inner_n);
        map_add_f64_at(m, key.data(), comp, v);
    } catch (...) {
    }
}

void PluginFold::map_add_u64(MapAccum* m, const std::int64_t* key,
                             std::uint64_t v) {
    map_add_u64_at(m, key, 0, v);
}

void PluginFold::map_add_f64(MapAccum* m, const std::int64_t* key, double v) {
    map_add_f64_at(m, key, 0, v);
}

void PluginFold::map_add_u64_at(MapAccum* m, const std::int64_t* key,
                                std::uint32_t comp, std::uint64_t v) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        m->touch(k, inserted)[comp].add_u64(v);
        account_add(*m, k, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_f64_at(MapAccum* m, const std::int64_t* key,
                                std::uint32_t comp, double v) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        m->touch(k, inserted)[comp].add_f64(v, 1.0);
        account_add(*m, k, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_ordered_at(MapAccum* m, const std::int64_t* key,
                                    std::uint32_t comp, std::int64_t order_key,
                                    std::uint64_t element) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        m->touch(k, inserted)[comp].add_ordered(
            order_key, static_cast<std::int64_t>(element));
        account_add(*m, k, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_argby_at(MapAccum* m, const std::int64_t* key,
                                  std::uint32_t comp, double by,
                                  std::int64_t payload) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        m->touch(k, inserted)[comp].add_argby(by, payload);
        account_add(*m, k, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_xy_at(MapAccum* m, const std::int64_t* key,
                               std::uint32_t comp, double x, double y) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        m->touch(k, inserted)[comp].add_xy(x, y);
        account_add(*m, k, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

MapAccum* PluginFold::map_get_argrow(const char* name,
                                     const dftu_type* key_types,
                                     std::uint32_t key_n, int is_max,
                                     const dftu_type* payload_types,
                                     std::uint32_t payload_n) {
    if (!name || (key_n && !key_types) || payload_n == 0 || !payload_types)
        return nullptr;
    for (std::uint32_t i = 0; i < key_n; ++i)
        if (!key_type_supported(key_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin arg-row map '%s' key component %u must be a "
                "fixed-width "
                "integer (I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    // The payload row reuses the key encoding, so it accepts the same supported
    // component types.
    for (std::uint32_t i = 0; i < payload_n; ++i)
        if (!key_type_supported(payload_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin arg-row map '%s' payload component %u must be a "
                "fixed-width integer (I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(name);
    return maps_.get_or_create(key, [&] {
        const dftu_monoid_kind vk =
            is_max ? DFTU_MONOID_ARGMAX_ROW : DFTU_MONOID_ARGMIN_ROW;
        MapAccum m;
        m.name = name;
        m.key_n = key_n;
        m.key_types.assign(key_types, key_types + key_n);
        m.value_kinds.assign(1, vk);
        m.payload_types.assign(payload_types, payload_types + payload_n);
        m.value_base_bytes += MonoidAccumulator(vk).state_bytes();
        m.set_part_bits(map_part_bits_);
        return m;
    });
}

MapAccum* PluginFold::map_get_sketch(const char* name,
                                     const dftu_type* key_types,
                                     std::uint32_t key_n, const double* qs,
                                     std::uint32_t nq) {
    if (!name || (key_n && !key_types) || nq == 0 || !qs) return nullptr;
    for (std::uint32_t i = 0; i < key_n; ++i)
        if (!key_type_supported(key_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin sketch map '%s' key component %u must be a fixed-width "
                "integer (I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    for (std::uint32_t i = 0; i < nq; ++i)
        if (!(qs[i] >= 0.0 && qs[i] <= 1.0)) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin sketch map '%s' quantile %u = %f is outside [0, 1]",
                name, i, qs[i]);
            return nullptr;
        }
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(name);
    return maps_.get_or_create(key, [&] {
        MapAccum m;
        m.name = name;
        m.key_n = key_n;
        m.key_types.assign(key_types, key_types + key_n);
        m.value_kinds.assign(1, DFTU_MONOID_SKETCH);
        m.quantile_qs.assign(qs, qs + nq);
        m.value_base_bytes +=
            MonoidAccumulator(DFTU_MONOID_SKETCH).state_bytes();
        m.set_part_bits(map_part_bits_);
        return m;
    });
}

MapAccum* PluginFold::map_get_fused(const char* name,
                                    const dftu_type* key_types,
                                    std::uint32_t key_n,
                                    const char* const* out_names,
                                    const dftu_monoid_kind* values,
                                    std::uint32_t value_n) {
    if (!name || (key_n && !key_types) || value_n == 0 || !values || !out_names)
        return nullptr;
    for (std::uint32_t i = 0; i < key_n; ++i)
        if (!key_type_supported(key_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin fused map '%s' key component %u must be a fixed-width "
                "integer (I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    for (std::uint32_t i = 0; i < value_n; ++i) {
        if (!monoid_is_fused_eligible(values[i]) || !out_names[i]) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin fused map '%s' component %u is not a fused-eligible "
                "scalar monoid",
                name, i);
            return nullptr;
        }
    }
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(name);
    return maps_.get_or_create(key, [&] {
        MapAccum m;
        m.name = name;
        m.key_n = key_n;
        m.key_types.assign(key_types, key_types + key_n);
        m.value_kinds.assign(values, values + value_n);
        for (std::uint32_t i = 0; i < value_n; ++i) {
            m.value_base_bytes += MonoidAccumulator(values[i]).state_bytes();
            m.fused_out_names.emplace_back(out_names[i]);
        }
        m.set_part_bits(map_part_bits_);
        return m;
    });
}

void PluginFold::map_add_row(MapAccum* m, const std::int64_t* key,
                             const ::dftu_row_val* vals, std::uint32_t n) {
    if (!m || (m->key_n && !key) || (n && !vals)) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        // One lookup for the whole row; components are updated in place, so no
        // spill occurs until after the row is applied and the reference is
        // done.
        std::vector<MonoidAccumulator>& into = m->touch(k, inserted);
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint32_t comp = vals[i].comp;
            if (comp >= into.size()) continue;
            if (vals[i].is_f64)
                into[comp].add_f64(vals[i].value.f, 1.0);
            else
                into[comp].add_u64(vals[i].value.u);
            account_add(*m, k, comp, inserted && i == 0);
        }
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_argrow(MapAccum* m, const std::int64_t* key, double by,
                                const std::int64_t* payload,
                                std::uint32_t payload_n) {
    if (!m || (m->key_n && !key) || m->value_kinds.empty() ||
        (payload_n && !payload))
        return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        m->touch(k, inserted)[0].add_argrow(by, payload, payload_n);
        account_add(*m, k, 0, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_topk_at(MapAccum* m, const std::int64_t* key,
                                 std::uint32_t comp, std::uint32_t k, double by,
                                 std::int64_t payload) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> kk(key, key + m->key_n);
        m->touch(kk, inserted)[comp].add_topk(k, by, payload);
        account_add(*m, kk, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_approx_topk_at(MapAccum* m, const std::int64_t* key,
                                        std::uint32_t comp, std::uint32_t k,
                                        std::int64_t value) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> kk(key, key + m->key_n);
        m->touch(kk, inserted)[comp].add_approx_topk(k, value);
        account_add(*m, kk, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_sample_at(MapAccum* m, const std::int64_t* key,
                                   std::uint32_t comp, std::uint32_t k,
                                   std::int64_t item) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> kk(key, key + m->key_n);
        m->touch(kk, inserted)[comp].add_sample(k, item);
        account_add(*m, kk, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_set_ordered(MapAccum* m, int ordered) {
    if (m) m->ordered = ordered != 0;
}

void PluginFold::declare_join(const char* out_name, const char* left_name,
                              const char* right_name, dftu_join_type type) {
    if (!out_name || !left_name || !right_name) return;
    for (const DeclaredJoin& j : joins_)
        if (j.out_name == out_name) return;
    joins_.push_back({out_name, left_name, right_name, type});
}

std::uint64_t PluginFold::map_size(MapAccum* m) {
    if (!m) return 0;
    reload_runs(*m);
    return static_cast<std::uint64_t>(m->total_entries());
}

int PluginFold::map_lookup(MapAccum* m, const std::int64_t* key,
                           std::uint32_t key_n, std::uint32_t comp,
                           ::dftu_monoid_value* out) {
    if (!m || !key || !out) return 0;
    reload_runs(*m);
    if (m->parts.empty()) return 0;
    std::vector<std::int64_t> k(key, key + key_n);
    const MapAccum::EntriesMap& part = m->parts[m->partition_of(k)];
    auto it = part.find(k);
    if (it == part.end() || comp >= it->second.size()) return 0;
    *out = it->second[comp].to_value();
    return 1;
}

void* PluginFold::map_iter_new(MapAccum* m) {
    if (!m) return nullptr;
    reload_runs(*m);
    return new MapCursor(m);
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW
void PluginFold::materialize_map(
    MapAccum& m, std::size_t max_rows_per_batch,
    std::vector<utilities::common::arrow::ArrowExportResult>& out,
    int only_comp) {
    namespace arr = utilities::common::arrow;
    using NestedEntryRef = const std::pair<const std::vector<std::int64_t>,
                                           std::vector<MonoidAccumulator>>*;
    // Compare key components [off, off+n): STR by resolved label, float by
    // value, unsigned by magnitude, signed by value. Returns -1/0/1.
    auto cmp_range = [&](const MapAccum& acc,
                         const std::vector<std::int64_t>& a,
                         const std::vector<std::int64_t>& b, std::uint32_t off,
                         std::uint32_t n) -> int {
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint32_t k = off + i;
            const dftu_type t = acc.key_types[k];
            if (key_type_interned(t)) {
                // STR and BYTES both sort byte-lexicographically on their
                // resolved span.
                std::string_view sa =
                    resolve_id(*intern_, static_cast<dftu_str>(a[k]));
                std::string_view sb =
                    resolve_id(*intern_, static_cast<dftu_str>(b[k]));
                if (sa != sb) return sa < sb ? -1 : 1;
            } else if (a[k] != b[k]) {
                if (key_type_float(t)) {
                    double da = key_bits_to_double(t, a[k]);
                    double db = key_bits_to_double(t, b[k]);
                    if (da != db) return da < db ? -1 : 1;
                    return a[k] < b[k] ? -1 : 1;
                }
                if (key_type_unsigned(t))
                    return static_cast<std::uint64_t>(a[k]) <
                                   static_cast<std::uint64_t>(b[k])
                               ? -1
                               : 1;
                return a[k] < b[k] ? -1 : 1;
            }
        }
        return 0;
    };

    // One Arrow batch per chunk of output rows; max_rows == 0 emits a single
    // batch. append_unit fills output row u, nunits is the output-row count
    // (groups for a nested map). A fresh builder per chunk avoids reset()-reuse
    // across the list/struct column types.
    auto emit_chunks = [&](const std::vector<arr::ColumnSpec>& specs,
                           std::size_t nunits, std::size_t max_rows,
                           const auto& append_unit) {
        auto emit = [&](std::size_t lo, std::size_t hi) {
            auto builder = std::make_unique<arr::RecordBatchBuilder>();
            builder->declare_schema(specs);
            builder->reserve(hi - lo);
            for (std::size_t u = lo; u < hi; ++u) append_unit(*builder, u);
            arr::ArrowExportResult res = builder->finish();
            if (res.valid()) out.push_back(std::move(res));
        };
        if (max_rows == 0 || nunits == 0) {
            emit(0, nunits);
        } else {
            for (std::size_t lo = 0; lo < nunits; lo += max_rows)
                emit(lo, std::min(nunits, lo + max_rows));
        }
    };

    // Reload any spilled runs so the map materializes identically to the
    // never-spilled path.
    reload_runs(m);
    const std::size_t max = max_rows_per_batch;
    if (m.nested_inner_n > 0) {
        const std::uint32_t inner_n = m.nested_inner_n;
        const std::uint32_t outer_n = m.key_n - inner_n;
        const std::uint32_t nvalue =
            static_cast<std::uint32_t>(m.value_kinds.size());
        std::vector<arr::ColumnSpec> specs;
        specs.reserve(outer_n + 1);
        append_key_specs(specs, m.key_types, outer_n);
        arr::ColumnSpec nested;
        nested.name = "value";
        nested.type = arr::ColumnType::STRUCT_LIST;
        for (std::uint32_t j = 0; j < inner_n; ++j)
            nested.fields.push_back({"ik" + std::to_string(j),
                                     key_column_type(m.inner_key_types[j]),
                                     {}});
        for (std::uint32_t c = 0; c < nvalue; ++c)
            nested.fields.push_back(
                {nvalue == 1 ? std::string("value") : "v" + std::to_string(c),
                 monoid_value_column_type(m.value_kinds[c]),
                 {}});
        specs.push_back(std::move(nested));

        std::vector<NestedEntryRef> rows;
        rows.reserve(m.total_entries());
        for (const auto& part : m.parts)
            for (const auto& kv : part) rows.push_back(&kv);
        // Always sort (outer then inner), not only when m.ordered: a nested
        // list has no intrinsic element order, so sorting is what makes the
        // rows reproducible across the merge's arbitrary hash order.
        std::sort(
            rows.begin(), rows.end(), [&](NestedEntryRef a, NestedEntryRef b) {
                int c = cmp_range(m, a->first, b->first, 0, outer_n);
                if (c != 0) return c < 0;
                return cmp_range(m, a->first, b->first, outer_n, inner_n) < 0;
            });
        // One output row per outer key. Chunk by whole groups so an outer group
        // is never split across batches.
        std::vector<std::pair<std::size_t, std::size_t>> groups;
        for (std::size_t gi = 0; gi < rows.size();) {
            std::size_t gj = gi;
            while (gj < rows.size() &&
                   cmp_range(m, rows[gi]->first, rows[gj]->first, 0, outer_n) ==
                       0)
                ++gj;
            groups.push_back({gi, gj});
            gi = gj;
        }
        auto append_group = [&](arr::RecordBatchBuilder& builder,
                                std::size_t g) {
            const std::size_t gi = groups[g].first;
            const std::size_t gj = groups[g].second;
            const std::vector<std::int64_t>& okey = rows[gi]->first;
            for (std::uint32_t o = 0; o < outer_n; ++o)
                append_key_cell(builder, o, m.key_types[o], okey[o], *intern_);
            std::vector<std::vector<arr::StructCell>> inner;
            inner.reserve(gj - gi);
            for (std::size_t r = gi; r < gj; ++r) {
                const std::vector<std::int64_t>& key = rows[r]->first;
                const std::vector<MonoidAccumulator>& mons = rows[r]->second;
                std::vector<arr::StructCell> cells(inner_n + nvalue);
                for (std::uint32_t jj = 0; jj < inner_n; ++jj) {
                    const dftu_type t = m.inner_key_types[jj];
                    const std::int64_t bits = key[outer_n + jj];
                    arr::StructCell& cell = cells[jj];
                    if (key_type_interned(t))
                        // STR and BYTES both resolve to a byte span; the field
                        // column type (utf8 vs binary) decides output.
                        cell.str =
                            resolve_id(*intern_, static_cast<dftu_str>(bits));
                    else if (key_type_float(t))
                        cell.f64 = key_bits_to_double(t, bits);
                    else if (key_type_unsigned(t))
                        cell.u64 = static_cast<std::uint64_t>(bits);
                    else
                        cell.i64 = bits;
                }
                for (std::uint32_t c = 0; c < nvalue; ++c) {
                    dftu_monoid_value v = mons[c].to_value();
                    arr::StructCell& cell = cells[inner_n + c];
                    switch (monoid_value_column_type(m.value_kinds[c])) {
                        case arr::ColumnType::FLOAT32:
                        case arr::ColumnType::DOUBLE:
                            cell.f64 = v.as.f64;
                            break;
                        case arr::ColumnType::UINT8:
                        case arr::ColumnType::UINT16:
                        case arr::ColumnType::UINT32:
                            cell.u64 = v.as.u64;
                            break;
                        case arr::ColumnType::STRING:
                            cell.str = resolve_id(
                                *intern_, static_cast<dftu_str>(v.as.u64));
                            break;
                        default:
                            cell.i64 = static_cast<std::int64_t>(v.as.u64);
                            break;
                    }
                }
                inner.push_back(std::move(cells));
            }
            builder.append_struct_list(outer_n, inner);
            builder.end_row();
        };
        emit_chunks(specs, groups.size(), max, append_group);
        return;
    }
    const std::uint32_t value_n =
        static_cast<std::uint32_t>(m.value_kinds.size());
    std::vector<arr::ColumnSpec> specs;
    specs.reserve(m.key_n + value_n);
    append_key_specs(specs, m.key_types, m.key_n);
    append_value_specs(specs, m.value_kinds, m.payload_types, m.quantile_qs, "",
                       only_comp);
    using EntryRef = const std::pair<const std::vector<std::int64_t>,
                                     std::vector<MonoidAccumulator>>*;
    std::vector<EntryRef> rows;
    rows.reserve(m.total_entries());
    for (const auto& part : m.parts)
        for (const auto& kv : part) rows.push_back(&kv);
    if (m.ordered) {
        // Resolve each interned key component once, then sort rows: interned by
        // resolved bytes, I64 by value.
        std::vector<std::vector<std::string_view>> labels(rows.size());
        for (std::size_t r = 0; r < rows.size(); ++r) {
            labels[r].resize(m.key_n);
            for (std::uint32_t i = 0; i < m.key_n; ++i)
                if (key_type_interned(m.key_types[i]))
                    labels[r][i] = resolve_id(
                        *intern_, static_cast<dftu_str>(rows[r]->first[i]));
        }
        std::vector<std::size_t> idx(rows.size());
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            const auto& ka = rows[a]->first;
            const auto& kb = rows[b]->first;
            for (std::uint32_t i = 0; i < m.key_n; ++i) {
                if (key_type_interned(m.key_types[i])) {
                    if (labels[a][i] != labels[b][i])
                        return labels[a][i] < labels[b][i];
                } else if (ka[i] != kb[i]) {
                    if (key_type_float(m.key_types[i])) {
                        double da = key_bits_to_double(m.key_types[i], ka[i]);
                        double db = key_bits_to_double(m.key_types[i], kb[i]);
                        if (da != db) return da < db;
                        // Equal value, differing bits (e.g. -0.0): fall back to
                        // bit order for determinism.
                        return ka[i] < kb[i];
                    }
                    if (key_type_unsigned(m.key_types[i]))
                        return static_cast<std::uint64_t>(ka[i]) <
                               static_cast<std::uint64_t>(kb[i]);
                    return ka[i] < kb[i];
                }
            }
            return false;
        });
        std::vector<EntryRef> sorted;
        sorted.reserve(rows.size());
        for (std::size_t r : idx) sorted.push_back(rows[r]);
        rows.swap(sorted);
    }

    auto append_row = [&](arr::RecordBatchBuilder& builder, std::size_t ri) {
        EntryRef row = rows[ri];
        const std::vector<std::int64_t>& key = row->first;
        const std::vector<MonoidAccumulator>& mons = row->second;
        for (std::uint32_t i = 0; i < m.key_n; ++i)
            append_key_cell(builder, i, m.key_types[i], key[i], *intern_);
        append_value_cells(builder, m.key_n, mons, m.value_kinds,
                           m.payload_types, m.quantile_qs, *intern_, only_comp);
        builder.end_row();
    };
    emit_chunks(specs, rows.size(), max, append_row);
}

// Column order: key columns k0.., then LEFT value columns ("l_"), then RIGHT
// ("r_"); the prefixes stop the two sides colliding on a shared value name.
// Rows follow the JoinedMap's key order; an absent OUTER-join side nulls its
// values.
static JoinType to_join_type(dftu_join_type t) {
    switch (t) {
        case DFTU_JOIN_LEFT:
            return JoinType::LEFT;
        case DFTU_JOIN_RIGHT:
            return JoinType::RIGHT;
        case DFTU_JOIN_FULL:
            return JoinType::FULL;
        case DFTU_JOIN_INNER:
        default:
            return JoinType::INNER;
    }
}

utilities::common::arrow::ArrowExportResult materialize_joined(
    const JoinedMap& j, dftracer::utils::StringIntern& intern) {
    namespace arr = utilities::common::arrow;
    arr::ArrowExportResult empty;
    if (!j.valid) return empty;

    std::vector<arr::ColumnSpec> specs;
    append_key_specs(specs, j.key_types, j.key_n);
    append_value_specs(specs, j.left_schema.value_kinds,
                       j.left_schema.payload_types, j.left_schema.quantile_qs,
                       "l_");
    if (!j.left_only)
        append_value_specs(specs, j.right_schema.value_kinds,
                           j.right_schema.payload_types,
                           j.right_schema.quantile_qs, "r_");

    const std::uint32_t left_first = j.key_n;
    const std::uint32_t right_first =
        left_first + value_column_count(j.left_schema.value_kinds,
                                        j.left_schema.payload_types,
                                        j.left_schema.quantile_qs);

    arr::RecordBatchBuilder builder;
    builder.declare_schema(specs);
    builder.reserve(j.rows.size());
    for (const JoinedRow& row : j.rows) {
        for (std::uint32_t i = 0; i < j.key_n; ++i)
            append_key_cell(builder, i, j.key_types[i], row.key[i], intern);
        if (row.left_present)
            append_value_cells(
                builder, left_first, row.left_values, j.left_schema.value_kinds,
                j.left_schema.payload_types, j.left_schema.quantile_qs, intern);
        else
            null_value_cells(builder, left_first, j.left_schema.value_kinds,
                             j.left_schema.payload_types,
                             j.left_schema.quantile_qs);
        if (!j.left_only) {
            if (row.right_present)
                append_value_cells(builder, right_first, row.right_values,
                                   j.right_schema.value_kinds,
                                   j.right_schema.payload_types,
                                   j.right_schema.quantile_qs, intern);
            else
                null_value_cells(
                    builder, right_first, j.right_schema.value_kinds,
                    j.right_schema.payload_types, j.right_schema.quantile_qs);
        }
        builder.end_row();
    }
    return builder.finish();
}

utilities::common::arrow::ArrowExportResult materialize_exploded(
    const ExplodedRows& e, dftracer::utils::StringIntern& intern) {
    namespace arr = utilities::common::arrow;
    arr::ArrowExportResult empty;
    if (!e.valid) return empty;

    std::vector<arr::ColumnSpec> specs;
    append_key_specs(specs, e.key_types, e.key_n);
    // Nested maps reject a SKETCH value, so the exploded rows carry no
    // quantiles.
    const std::vector<double> no_qs;
    append_value_specs(specs, e.value_kinds, e.payload_types, no_qs, "");
    const std::uint32_t elem_col =
        e.key_n + value_column_count(e.value_kinds, e.payload_types, no_qs);
    specs.push_back({e.elem_name, key_column_type(e.elem_type)});

    arr::RecordBatchBuilder builder;
    builder.declare_schema(specs);
    builder.reserve(e.rows.size());
    for (const ExplodedRow& row : e.rows) {
        for (std::uint32_t i = 0; i < e.key_n; ++i)
            append_key_cell(builder, i, e.key_types[i], row.key[i], intern);
        append_value_cells(builder, e.key_n, row.kept_values, e.value_kinds,
                           e.payload_types, no_qs, intern);
        if (row.elem_null)
            builder.append_null(elem_col);
        else
            append_key_cell(builder, elem_col, e.elem_type, row.elem, intern);
        builder.end_row();
    }
    return builder.finish();
}

utilities::common::arrow::ArrowExportResult materialize_grouping_sets(
    const std::vector<MapAccum>& groupings,
    const std::vector<std::vector<std::uint32_t>>& keep_sets,
    const MapAccum& original, dftracer::utils::StringIntern& intern) {
    namespace arr = utilities::common::arrow;
    arr::ArrowExportResult empty;
    if (groupings.size() != keep_sets.size() || original.key_n == 0)
        return empty;

    std::vector<arr::ColumnSpec> specs;
    append_key_specs(specs, original.key_types, original.key_n);
    append_value_specs(specs, original.value_kinds, original.payload_types,
                       original.quantile_qs, "");
    const std::uint32_t value_first = original.key_n;
    const std::uint32_t gid_col =
        original.key_n + value_column_count(original.value_kinds,
                                            original.payload_types,
                                            original.quantile_qs);
    // grouping_id is the keep-set index, not a SQL GROUPING() null-mask
    // bitmask.
    specs.push_back({"grouping_id", arr::ColumnType::INT64});

    arr::RecordBatchBuilder builder;
    builder.declare_schema(specs);
    std::size_t total = 0;
    for (const MapAccum& g : groupings) total += g.total_entries();
    builder.reserve(total);

    using Entry = std::pair<const std::vector<std::int64_t>,
                            std::vector<MonoidAccumulator>>;
    std::vector<std::int32_t> slot(original.key_n);
    for (std::size_t i = 0; i < groupings.size(); ++i) {
        const MapAccum& g = groupings[i];
        const std::vector<std::uint32_t>& ks = keep_sets[i];
        // slot[d] = the regrouped-key position of original component d, or -1
        // when this set dropped d (its column is nulled).
        for (std::uint32_t d = 0; d < original.key_n; ++d) slot[d] = -1;
        for (std::uint32_t j = 0; j < ks.size(); ++j)
            if (ks[j] < original.key_n)
                slot[ks[j]] = static_cast<std::int32_t>(j);

        std::vector<const Entry*> ents;
        ents.reserve(g.total_entries());
        for (const MapAccum::EntriesMap& part : g.parts)
            for (const auto& e : part) ents.push_back(&e);
        std::sort(ents.begin(), ents.end(), [](const Entry* a, const Entry* b) {
            return a->first < b->first;
        });

        for (const Entry* e : ents) {
            for (std::uint32_t d = 0; d < original.key_n; ++d) {
                if (slot[d] < 0)
                    builder.append_null(d);
                else
                    append_key_cell(builder, d, original.key_types[d],
                                    e->first[slot[d]], intern);
            }
            append_value_cells(builder, value_first, e->second,
                               original.value_kinds, original.payload_types,
                               original.quantile_qs, intern);
            builder.append_int64(gid_col, static_cast<std::int64_t>(i));
            builder.end_row();
        }
    }
    return builder.finish();
}

void PluginFold::collect_map_batches_streaming(
    MapAccum& m,
    std::vector<utilities::common::arrow::ArrowExportResult>& out) {
    // Ordered/nested maps must order globally, so reload the whole map and
    // stream the sorted result in row-capped chunks. A key (flat) or outer
    // group (nested) lives in exactly one partition, so the union never
    // re-merges a monoid and a global sort matches merging the K sorted
    // partition streams.
    if (m.ordered || m.nested_inner_n > 0) {
        const std::size_t chunk = map_stream_chunk_rows_
                                      ? map_stream_chunk_rows_
                                      : MAP_STREAM_CHUNK_ROWS;
        materialize_map(m, chunk, out);
        if (m.partitions() > map_stream_max_resident_parts_)
            map_stream_max_resident_parts_ = m.partitions();
        return;
    }
    // Unordered: each key lives in one partition, so materialize one at a time
    // and free it before the next. Row order across partitions is unspecified.
    for (std::uint32_t p = 0; p < m.partitions(); ++p) {
        if (m.parts[p].empty() && m.runs[p].empty()) continue;
        MapAccum tmp;
        tmp.name = m.name;
        tmp.key_n = m.key_n;
        tmp.key_types = m.key_types;
        tmp.value_kinds = m.value_kinds;
        tmp.payload_types = m.payload_types;
        tmp.quantile_qs = m.quantile_qs;
        tmp.fused_out_names = m.fused_out_names;
        tmp.value_base_bytes = m.value_base_bytes;
        tmp.set_part_bits(0);
        tmp.parts[0] = std::move(m.parts[p]);
        tmp.runs[0] = std::move(m.runs[p]);
        materialize_map(tmp, 0, out);
        m.parts[p].clear();
        m.runs[p].clear();
        if (map_stream_max_resident_parts_ < 1)
            map_stream_max_resident_parts_ = 1;
    }
}
#endif

void PluginFold::materialize_maps() {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
    if (!named_results_) return;
    if (map_spill_failed_) {
        DFTRACER_UTILS_LOG_ERROR(
            "Plugin map spill failed earlier; refusing to emit maps rather "
            "than "
            "a silently partial result");
        return;
    }
    namespace arr = utilities::common::arrow;
    map_stream_batches_ = 0;
    map_stream_max_resident_parts_ = 0;
    // Emit one collected map (as `name`): a single batch as an Arrow table,
    // multiple as a streamed RecordBatchReader (see result_to_py).
    auto emit_named = [&](const char* name,
                          std::vector<arr::ArrowExportResult>& batches) {
        std::vector<arr::ArrowExportResult> valid;
        valid.reserve(batches.size());
        for (auto& b : batches)
            if (b.valid()) valid.push_back(std::move(b));
        if (valid.empty()) return;
        map_stream_batches_ += valid.size();
        if (valid.size() == 1) {
            named_results_->emit_arrow(name, valid[0].get_array(),
                                       valid[0].get_schema());
        } else {
            OwnedArrowBatches ob;
            ob.batches.reserve(valid.size());
            for (auto& b : valid) {
                OwnedArrow o;
                ArrowArrayMove(b.get_array(), &o.array);
                ArrowSchemaMove(b.get_schema(), &o.schema);
                ob.batches.push_back(std::move(o));
            }
            named_results_->emit_arrow_batches(name, std::move(ob));
        }
    };
    for (MapAccum& m : maps_) {
        try {
            // A fused map splits into one named table per component, so the
            // caller sees the separate maps it declared, never the product.
            if (!m.fused_out_names.empty()) {
                for (std::uint32_t c = 0; c < m.fused_out_names.size(); ++c) {
                    std::vector<arr::ArrowExportResult> batches;
                    materialize_map(m, 0, batches, static_cast<int>(c));
                    emit_named(m.fused_out_names[c].c_str(), batches);
                }
                continue;
            }
            std::vector<arr::ArrowExportResult> batches;
            if (map_stream_enabled_)
                collect_map_batches_streaming(m, batches);
            else
                materialize_map(m, 0, batches);
            emit_named(m.name.c_str(), batches);
        } catch (...) {
        }
    }
    auto find_map = [&](const std::string& name) -> MapAccum* {
        return maps_.find(dftracer::utils::hash::fnv1a_hash(name));
    };
    for (const DeclaredJoin& dj : joins_) {
        try {
            MapAccum* left = find_map(dj.left_name);
            MapAccum* right = find_map(dj.right_name);
            if (!left || !right) {
                DFTRACER_UTILS_LOG_ERROR(
                    "Plugin join '%s' skipped: input map '%s' not found",
                    dj.out_name.c_str(),
                    !left ? dj.left_name.c_str() : dj.right_name.c_str());
                continue;
            }
            JoinedMap jm = join_maps(*left, *right, to_join_type(dj.type));
            if (!jm.valid) {
                DFTRACER_UTILS_LOG_ERROR(
                    "Plugin join '%s' skipped: maps '%s' and '%s' do not share "
                    "a key schema",
                    dj.out_name.c_str(), dj.left_name.c_str(),
                    dj.right_name.c_str());
                continue;
            }
            arr::ArrowExportResult res = materialize_joined(jm, *intern_);
            if (res.valid())
                named_results_->emit_arrow(dj.out_name.c_str(), res.get_array(),
                                           res.get_schema());
        } catch (...) {
        }
    }
#endif
}

}  // namespace dftracer::utils::plugins
