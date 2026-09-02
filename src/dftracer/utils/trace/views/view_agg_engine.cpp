#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/rocksdb/database.h>
#include <dftracer/utils/dataframe/agg.h>
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/trace/views/aggfold.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/native_row_fold.h>
#include <dftracer/utils/trace/views/rollup_store.h>
#include <dftracer/utils/trace/views/view_agg_engine.h>
#include <dftracer/utils/trace/views/view_aggregate.h>
#include <dftracer/utils/trace/views/view_executor.h>
#include <dftracer/utils/trace/views/view_scan.h>
#include <dftracer/utils/trace/views/view_source.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace dftracer::utils::trace::views::detail {

namespace dataframe = dftracer::utils::dataframe;

namespace {

// The frame column name the raw scan produces for `field` fed as a value/by
// column: a top-level field keeps its name, an arg field becomes args.<name>
// (canonical_row_column_name), so the group_by references the column the scan
// emits. Empty field stays empty (Count()'s neutral value column).
std::string value_col_name(const std::string& field) {
    return field.empty() ? std::string() : canonical_row_column_name(field);
}

// A field agg_field_typed_t/agg_field_t derives instead of reading straight
// (size = io-cat byte size, te = ts+dur). The raw scan builds it as a typed
// derived column (AGG_DERIVED_PREFIX / derived_agg_column) rather than reading
// a stored field.
bool is_derived_field(const std::string& f) { return f == "size" || f == "te"; }

// The frame column name a value/by field's group agg reads: a derived field
// (size/te) keeps its own name (build_row_frame emits it under that name from
// the derived token), any other field maps through value_col_name.
std::string value_col(const std::string& field) {
    return is_derived_field(field) ? field : value_col_name(field);
}

// The raw-scan select token that produces a value/by field's column: a derived
// field routes through AGG_DERIVED_PREFIX (a typed derived column), any other
// field is selected by its own name. Empty field stays empty.
std::string value_select_token(const std::string& field) {
    if (field.empty()) return std::string();
    return is_derived_field(field) ? std::string(AGG_DERIVED_PREFIX) + field
                                   : field;
}

// ts/dur/te accumulate in the unrounded time_scale domain (agg_fold.h's
// field_scaled); a value agg over one needs the engine-side rescale path.
bool is_scaled_field(const std::string& f) {
    return f == "ts" || f == "dur" || f == "te";
}

bool agg_op_engine_supported(AggOp op) {
    switch (op) {
        case AggOp::Count:
        case AggOp::Sum:
        case AggOp::Min:
        case AggOp::Max:
        case AggOp::Mean:
        case AggOp::Var:
        case AggOp::Std:
        case AggOp::Skew:
        case AggOp::Kurt:
        case AggOp::SumSq:
        case AggOp::Pct:
        case AggOp::Hist:
        case AggOp::ArgMax:
        case AggOp::SetUnion:
        case AggOp::Busy:
        case AggOp::Concurrency:
        case AggOp::Utilization:
        case AggOp::Active:
            return true;
    }
    return false;
}

dataframe::Agg to_engine_agg(AggOp op) {
    switch (op) {
        case AggOp::Count:
            return dataframe::Agg::Count;
        case AggOp::Sum:
            return dataframe::Agg::Sum;
        case AggOp::Min:
            return dataframe::Agg::Min;
        case AggOp::Max:
            return dataframe::Agg::Max;
        case AggOp::Mean:
            return dataframe::Agg::Mean;
        case AggOp::Var:
            return dataframe::Agg::Var;
        case AggOp::Std:
            return dataframe::Agg::Std;
        case AggOp::Skew:
            return dataframe::Agg::Skew;
        case AggOp::Kurt:
            return dataframe::Agg::Kurt;
        case AggOp::SumSq:
            return dataframe::Agg::SumSq;
        case AggOp::Pct:
            return dataframe::Agg::Pct;
        case AggOp::Hist:
            return dataframe::Agg::Hist;
        case AggOp::ArgMax:
            return dataframe::Agg::ArgMax;
        case AggOp::SetUnion:
            return dataframe::Agg::SetUnion;
        case AggOp::Busy:
            return dataframe::Agg::Busy;
        case AggOp::Concurrency:
            return dataframe::Agg::Concurrency;
        case AggOp::Utilization:
            return dataframe::Agg::Utilization;
        case AggOp::Active:
            return dataframe::Agg::Active;
    }
    throw DFTUtilsException::cat(ErrorCode::INTERNAL,
                                 "agg engine: unknown aggregate op");
}

dataframe::GroupAgg to_group_agg(const AggSpec& spec) {
    dataframe::GroupAgg g;
    g.op = to_engine_agg(spec.op);
    g.out = agg_col_name(spec);
    g.param = spec.q;
    if (spec.op == AggOp::ArgMax) {
        g.column = value_col(spec.field);
        g.by = value_col(spec.by);
    } else if (is_occupancy_op(spec.op)) {
        // Occupancy has no value field; it reads the raw (ts, dur) pair, with
        // the endpoint-snap tolerance carried in param.
        g.column = "ts";
        g.by = "dur";
    } else if (spec.op == AggOp::Count) {
        // Count() is the group row count; Count(field) counts only the
        // field-present rows, which the engine's CountValid reads from the
        // field's per-group stat (FieldStat::n).
        if (!spec.field.empty()) {
            g.op = dataframe::Agg::CountValid;
            g.column = value_col(spec.field);
        }
    } else {
        g.column = value_col(spec.field);
    }
    return g;
}

// to_batch's group key column is always a String (the fold's key is text);
// agg_finalize instead keeps a non-string key's native domain collapsed to
// Int64. Render that back to the same decimal text the GroupMap path would
// have produced (plain integer text - the group key values reachable here are
// pid/tid, always small enough that the int64/uint64 bit pattern round-trips
// through decimal identically).
dataframe::Series key_column_to_string(const dataframe::Series& col) {
    using dataframe::TypeId;
    if (col.type() == TypeId::String) return col.share();
    if (col.type() == TypeId::Int64) {
        const std::int64_t n = col.length();
        std::vector<std::string> vals(static_cast<std::size_t>(n));
        const std::int64_t* d = col.data<std::int64_t>();
        for (std::int64_t i = 0; i < n; ++i)
            vals[static_cast<std::size_t>(i)] = std::to_string(d[i]);
        return dataframe::Series::strings(vals);
    }
    throw DFTUtilsException::cat(
        ErrorCode::INTERNAL, "agg engine: unexpected group-key column type");
}

// Keys whose group column is an opaque identifier the post-aggregation re-key
// pass relabels to a human-readable name: fhash/hhash for resolved-name keys,
// pid for Rank (the PR-metadata rank map keys on pid).
bool key_is_resolved(GroupKey::Kind kind) {
    return kind == GroupKey::Kind::FilePath ||
           kind == GroupKey::Kind::FileName ||
           kind == GroupKey::Kind::HostName || kind == GroupKey::Kind::Rank;
}

// The raw scan/group-by field a key groups on: fhash/hhash for a resolved
// name key (the fold groups on the hash, a bijection, and relabels to the
// resolved name only after aggregation), pid for Rank (the rank map keys on
// pid, matching agg_fold.h's append_group_dim), a group-key-string sentinel for
// an Arg/Field key (rendered like the GroupMap fold, then relabeled to
// group_col_name after aggregation), else the key's own column.
std::string key_group_field(const GroupKey& gk) {
    switch (gk.kind) {
        case GroupKey::Kind::FilePath:
        case GroupKey::Kind::FileName:
            return "fhash";
        case GroupKey::Kind::HostName:
            return "hhash";
        case GroupKey::Kind::Rank:
            return "pid";
        case GroupKey::Kind::Arg:
            return std::string(AGG_KEY_ARG_PREFIX) + gk.arg;
        case GroupKey::Kind::Field:
            return std::string(AGG_KEY_FIELD_PREFIX) + gk.arg;
        default:
            return group_col_name(gk);
    }
}

// Resolve one already-stringified hash group-key column to its resolved name,
// matching resolve_group_keys/resolve_group_value exactly (same GroupResolver,
// same FileName-from-FilePath basename derivation).
dataframe::Series resolve_key_column(const dataframe::Series& hashes,
                                     const GroupResolver& resolver,
                                     GroupKey::Kind kind) {
    const std::int64_t n = hashes.length();
    std::vector<std::string> vals(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        vals[static_cast<std::size_t>(i)] = resolve_group_value(
            resolver, kind, std::string(hashes.string_at(i)));
    return dataframe::Series::strings(vals);
}

// One raw group-key column cell rendered exactly as the GroupMap fold builds
// its key: a String cell verbatim, an integer cell as decimal, a null cell as
// the empty string (append_arg emits nothing for a missing value).
std::string cell_to_key_string(const dataframe::Series& col, std::int64_t r) {
    using dataframe::TypeId;
    if (col.is_null(r)) return std::string();
    switch (col.type()) {
        case TypeId::String:
            return std::string(col.string_at(r));
        case TypeId::Int64:
            return std::to_string(col.data<std::int64_t>()[r]);
        case TypeId::Uint64:
            return std::to_string(col.data<std::uint64_t>()[r]);
        default:
            throw DFTUtilsException::cat(
                ErrorCode::INTERNAL,
                "agg engine: unexpected transform key-column type");
    }
}

// The pre-transform value of group key `gk` for one raw cell, matching the
// GroupMap path (resolve_group_keys: resolve_group_value after the fold's key
// rendering). A resolved-name key resolves its hash (or keeps the raw hash when
// no resolver is loaded); cat is lowercased like agg_fold.h's append_group_dim;
// the rest keep their rendered value.
std::string transform_key_base(const GroupKey& gk, std::string raw,
                               const GroupResolver* resolver) {
    if (key_is_resolved(gk.kind))
        return resolver ? resolve_group_value(*resolver, gk.kind, raw) : raw;
    if (gk.kind == GroupKey::Kind::Cat)
        for (char& ch : raw)
            ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
    return raw;
}

// Rank is a query-time side channel: the pid -> rank map lives in PR metadata
// records, not the index or the event columns the engine streams. Harvest it
// with the same AggFold logic the GroupMap path uses (identical records over
// the same trace, so the map is byte-identical) and feed it to the shared
// resolver the post-aggregation re-key reads.
coro::CoroTask<void> harvest_ranks(const ViewPlan& plan) {
    ViewPlan hp = plan;
    hp.group_by.assign(1, GroupKey::rank());
    hp.agg.clear();
    hp.numeric_arg_aggs.clear();
    hp.auto_numeric_metrics = false;
    hp.time_bucket_us = 0;
    hp.bucket_origin_us = 0;
    hp.bucket_origin_min = false;
    hp.materialize = false;
    hp.sort_col.clear();
    hp.topk_col.clear();
    hp.offset = 0;
    hp.limit = 0;
    hp.select.clear();
    hp.schema.reset();
    ensure_schema(hp);
    ViewDefinition vdef = make_vdef(hp, /*for_aggregation=*/true);
    dftracer::utils::StringIntern intern;
    AggFold agg(hp, intern);
    std::array<Fold*, 1> folds{&agg};
    co_await fuse(hp, vdef, folds, intern);
    apply_ranks(plan, agg.ranks());
}

// The numeric args auto_numeric_metrics discovers are data-dependent (the arg
// set is only known after a scan; see docs/plans
// lazyframe_async_unification 3.3 / 5.4). Run the SAME GroupMap fold that the
// legacy path folds with (a single group, no per-arg sketch) and read the
// sorted union of the arg names it saw - byte-identical to the set to_batch
// would emit (fold_numeric_args_t discovers, reserved/pre-agg-filtered, plus
// the io-cat-derived "size"). Its finish_map is discarded; only the names are
// kept, then fed back as engine value columns.
coro::CoroTask<std::vector<std::string>> harvest_numeric_arg_names(
    const ViewPlan& plan) {
    ViewPlan hp = plan;
    hp.group_by.clear();
    hp.agg.clear();
    hp.auto_numeric_metrics = true;
    hp.numeric_arg_aggs.clear();
    hp.time_bucket_us = 0;
    hp.bucket_origin_us = 0;
    hp.bucket_origin_min = false;
    hp.materialize = false;
    hp.sort_col.clear();
    hp.topk_col.clear();
    hp.offset = 0;
    hp.limit = 0;
    hp.select.clear();
    hp.schema.reset();
    ensure_schema(hp);
    ViewDefinition vdef = make_vdef(hp, /*for_aggregation=*/true);
    dftracer::utils::StringIntern intern;
    AggFold agg(hp, intern);
    std::array<Fold*, 1> folds{&agg};
    co_await fuse(hp, vdef, folds, intern);
    std::set<std::string> names;  // to_batch unions the dyn keys via std::set
    GroupMap m = agg.finish_map();
    for (const auto& [k, a] : m) {
        (void)k;
        for (const auto& [name, stat] : a.dyn) {
            (void)stat;
            names.insert(name);
        }
    }
    co_return std::vector<std::string>(names.begin(), names.end());
}

// The shared engine-aggregation tail (key rendering, busy_cell_us, resolver
// relabel, dyn fixes), matching to_batch byte-for-byte. Reused by the streaming
// serve path and finalize_engine_result.
dataframe::DataFrame finalize_engine_frame(dataframe::DataFrame r,
                                           const ViewPlan& plan,
                                           const std::vector<DynFix>& dyn) {
    const std::size_t ng = plan.group_by.size();
    const std::size_t off = plan.time_bucket_us > 0 ? 1 : 0;
    std::vector<std::string> key_names(ng);
    std::vector<char> key_transformed(ng, 0);
    for (std::size_t j = 0; j < ng; ++j) {
        key_names[j] = group_col_name(plan.group_by[j]);
        key_transformed[j] =
            plan.group_by[j].transform != GroupKey::Transform::None ? 1 : 0;
    }
    std::optional<std::size_t> cat_pos;
    for (std::size_t j = 0; j < ng; ++j)
        if (plan.group_by[j].kind == GroupKey::Kind::Cat && !key_transformed[j])
            cat_pos = j;
    std::size_t n_plan_value = 0;
    for (const auto& s : plan.agg)
        if (s.op != AggOp::ArgMax && s.op != AggOp::SetUnion) ++n_plan_value;
    if (plan.agg.empty()) n_plan_value = 1;
    const std::size_t n_value_cols = n_plan_value + dyn.size();

    if (off) r.names[0] = "time_bucket";
    if (cat_pos) r.names[off + *cat_pos] = "cat";
    for (std::size_t j = 0; j < ng; ++j) {
        const GroupKey::Kind k = plan.group_by[j].kind;
        if (key_transformed[j] || k == GroupKey::Kind::Arg ||
            k == GroupKey::Kind::Field)
            r.names[off + j] = key_names[j];
    }
    for (std::size_t i = 0; i < off + ng; ++i)
        r.columns[i] = key_column_to_string(r.columns[i]);

    const bool occ_cell_col =
        std::any_of(plan.agg.begin(), plan.agg.end(), [](const AggSpec& s) {
            return s.op == AggOp::Busy || s.op == AggOp::Concurrency ||
                   s.op == AggOp::Utilization;
        });
    if (occ_cell_col) {
        const std::int64_t nrows = r.num_rows();
        std::vector<std::int64_t> cell(
            static_cast<std::size_t>(nrows),
            static_cast<std::int64_t>(plan.occ_cell_us));
        const std::size_t at = off + ng + n_value_cols;
        r.names.insert(r.names.begin() + static_cast<std::ptrdiff_t>(at),
                       "busy_cell_us");
        r.columns.insert(r.columns.begin() + static_cast<std::ptrdiff_t>(at),
                         dataframe::Series::flat_i64(cell.data(), nrows));
    }

    bool needs_resolver = false;
    for (std::size_t j = 0; j < ng; ++j)
        needs_resolver =
            needs_resolver ||
            (key_is_resolved(plan.group_by[j].kind) && !key_transformed[j]);
    if (needs_resolver) {
        const GroupResolver* resolver = ensure_resolver(plan);
        for (std::size_t j = 0; j < ng; ++j) {
            if (!key_is_resolved(plan.group_by[j].kind) || key_transformed[j])
                continue;
            const std::size_t idx = off + j;
            r.columns[idx] = resolve_key_column(r.columns[idx], *resolver,
                                                plan.group_by[j].kind);
            r.names[idx] = key_names[j];
        }
    }

    for (const DynFix& d : dyn) {
        if (!d.pct) continue;
        const std::int64_t ci = r.column_index(d.out);
        if (ci < 0) continue;
        dataframe::Series& col = r.columns[static_cast<std::size_t>(ci)];
        if (col.type() != dataframe::TypeId::Float64) continue;
        const std::int64_t n = col.length();
        const double* src = col.data<double>();
        std::vector<double> vals(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
            vals[static_cast<std::size_t>(i)] =
                std::isnan(src[i]) ? 0.0 : src[i];
        col = dataframe::Series::flat_f64(vals.data(), n);
    }
    for (const DynFix& d : dyn) {
        if (!d.count) continue;
        const std::int64_t ci = r.column_index(d.out);
        if (ci < 0) continue;
        dataframe::Series& col = r.columns[static_cast<std::size_t>(ci)];
        if (col.type() != dataframe::TypeId::Int64) continue;
        const std::int64_t n = col.length();
        const std::int64_t* src = col.data<std::int64_t>();
        std::vector<double> vals(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
            vals[static_cast<std::size_t>(i)] = static_cast<double>(src[i]);
        col = dataframe::Series::flat_f64(vals.data(), n);
    }
    return r;
}

}  // namespace

dataframe::DataFrame finalize_engine_result(const dataframe::AggState& st,
                                            const ViewPlan& plan) {
    std::vector<std::string> names;
    names.reserve(1 + plan.group_by.size());
    if (plan.time_bucket_us > 0) names.push_back("time_bucket");
    for (const auto& gk : plan.group_by) names.push_back(group_col_name(gk));
    dataframe::DataFrame r = dataframe::agg_finalize(st, names);

    // Recover the dyn (auto_numeric_metrics) columns from the stored specs:
    // they sit between the plan's value specs and its text specs. dyn Count was
    // built as CountValid; dyn Pct stays Pct.
    const std::vector<dataframe::AggSpec>& specs = dataframe::agg_specs(st);
    std::size_t n_plan_value = 0, n_plan_text = 0;
    for (const auto& s : plan.agg) {
        if (s.op == AggOp::ArgMax || s.op == AggOp::SetUnion)
            ++n_plan_text;
        else
            ++n_plan_value;
    }
    if (plan.agg.empty()) n_plan_value = 1;
    std::vector<DynFix> dyn;
    for (std::size_t i = n_plan_value; i + n_plan_text < specs.size(); ++i)
        dyn.push_back({specs[i].out, specs[i].op == dataframe::AggOp::Pct,
                       specs[i].op == dataframe::AggOp::CountValid});
    return finalize_engine_frame(std::move(r), plan, dyn);
}

bool agg_engine_eligible(const ViewPlan& plan) {
    if (plan.group_by.empty() && plan.time_bucket_us == 0) return false;
    // Every GroupKey::Kind, transform, key/value field (top-level, arg, nested,
    // derived size/te), reduction, occupancy (per-column raw ts/dur alongside
    // scaled value aggs), and numeric-arg reduction is now served by
    // run_collect_via_engine, so the only gate left is on an unsupported
    // aggregate op. Note that the engine intentionally diverges from GroupMap
    // on one point: a domain-sensitive reduction (Sum/Min/Max/SumSq) over an
    // int field absent from an entire group keeps the engine's stable
    // per-column type instead of GroupMap's data-dependent Float64 widening.
    for (const auto& spec : plan.agg)
        if (!agg_op_engine_supported(spec.op)) return false;
    return true;
}

coro::CoroTask<EnginePrep> prepare_engine_group(const ViewPlan& plan) {
    // A Rank key groups on pid and relabels to the PR-metadata rank; harvest
    // that map into the resolver before the scan, mirroring run_scan_aggregate.
    if (std::any_of(
            plan.group_by.begin(), plan.group_by.end(),
            [](const GroupKey& gk) { return gk.kind == GroupKey::Kind::Rank; }))
        co_await harvest_ranks(plan);

    const bool has_bucket = plan.time_bucket_us > 0;
    const bool has_occ =
        std::any_of(plan.agg.begin(), plan.agg.end(),
                    [](const AggSpec& s) { return is_occupancy_op(s.op); });

    std::vector<std::string> key_fields;
    key_fields.reserve(plan.group_by.size());
    for (const GroupKey& gk : plan.group_by)
        key_fields.push_back(key_group_field(gk));

    // Scaled-field renaming: a scaled value field (ts/dur/te) routes through a
    // hidden pre-scaled column instead of the raw field name. Needed whenever
    // time_scale is non-identity and the raw scan is read unscaled (a bucket,
    // or any non-occupancy value agg over a scaled field). Occupancy always
    // reads raw ts/dur, so it is excluded here and never rescaled.
    static constexpr const char* SCALED_TS_COL = "__view_agg_engine_scaled_ts";
    static constexpr const char* SCALED_DUR_COL =
        "__view_agg_engine_scaled_dur";
    static constexpr const char* SCALED_TE_COL = "__view_agg_engine_scaled_te";
    const bool has_scaled_value_agg =
        std::any_of(plan.agg.begin(), plan.agg.end(), [](const AggSpec& s) {
            return !is_occupancy_op(s.op) &&
                   (is_scaled_field(s.field) || is_scaled_field(s.by));
        });
    const bool needs_value_scale =
        plan.time_scale != 1.0 && (has_bucket || has_scaled_value_agg);
    auto scaled_name = [&](const std::string& f) -> std::string {
        if (!needs_value_scale) return f;
        if (f == "ts") return SCALED_TS_COL;
        if (f == "dur") return SCALED_DUR_COL;
        if (f == "te") return SCALED_TE_COL;
        return f;
    };

    // Auto-discovered numeric-arg reductions (the auto_numeric_metrics / dyn
    // path). The arg set is only known after a scan, so discover it first, then
    // feed each (arg, reduction) to the engine as a Float64 value column.
    // Output column names and order match to_batch's dyn columns exactly (arg
    // outer, reduction inner; legacy empty-reductions = one bare-named per-arg
    // mean).
    struct DynAgg {
        std::string column;  // AGG_NUM_ARG_PREFIX + arg (the frame column name)
        std::string out;  // dyn_col_name(spec, arg), or the bare arg (legacy)
        AggOp op;
        double q;
    };
    std::vector<DynAgg> dyn_aggs;
    if (plan.auto_numeric_metrics) {
        std::vector<std::string> names =
            co_await harvest_numeric_arg_names(plan);
        for (const std::string& name : names) {
            const std::string col = std::string(AGG_NUM_ARG_PREFIX) + name;
            if (plan.numeric_arg_aggs.empty()) {
                dyn_aggs.push_back({col, name, AggOp::Mean, 0.0});
            } else {
                for (const AggSpec& r : plan.numeric_arg_aggs)
                    dyn_aggs.push_back({col, dyn_col_name(r, name), r.op, r.q});
            }
        }
    }

    // to_batch lays out value columns as [named non-text/hist aggs, dyn cols]
    // then text columns (ArgMax/SetUnion) then hist. The engine emits columns
    // in gaggs order, so partition named specs into value vs text and slot the
    // dyn columns between them to reproduce that order.
    std::vector<dataframe::GroupAgg> gaggs;
    std::vector<dataframe::GroupAgg> text_gaggs;
    if (plan.agg.empty()) {
        dataframe::GroupAgg g;
        g.op = dataframe::Agg::Count;
        g.out = agg_col_name(AggSpec(AggOp::Count));
        gaggs.push_back(std::move(g));
    } else {
        for (const auto& spec : plan.agg) {
            dataframe::GroupAgg g = to_group_agg(spec);
            // Occupancy reads raw ts/dur (never rescaled); every other value
            // agg over a scaled field routes to its pre-scaled column.
            if (is_occupancy_op(spec.op)) {
                g.param = plan.occ_cell_us;
            } else {
                g.column = scaled_name(g.column);
                g.by = scaled_name(g.by);
            }
            if (spec.op == AggOp::ArgMax || spec.op == AggOp::SetUnion)
                text_gaggs.push_back(std::move(g));
            else
                gaggs.push_back(std::move(g));
        }
    }
    for (const DynAgg& d : dyn_aggs) {
        dataframe::GroupAgg g;
        // A dyn Count is the arg-present count (FieldStat::n), not the group's
        // row count, so it maps to CountValid over the sentinel column; the
        // Int64 result is cast to Float64 below to match the dyn column type.
        g.op = d.op == AggOp::Count ? dataframe::Agg::CountValid
                                    : to_engine_agg(d.op);
        g.out = d.out;
        g.param = d.q;
        g.column = d.column;
        gaggs.push_back(std::move(g));
    }
    gaggs.insert(gaggs.end(), std::make_move_iterator(text_gaggs.begin()),
                 std::make_move_iterator(text_gaggs.end()));

    // A fixed select list on the raw scan (instead of the default, per-batch-
    // discovered schema) guarantees every streamed morsel carries the same
    // columns in the same order: the streaming group_by resolves key/value
    // columns once against the Source's schema, so a morsel with a different
    // column layout would silently misalign otherwise. Group keys select the
    // field they fold on (key_fields), which for a resolved name key is the
    // raw hash, not the resolved-name output column.
    std::vector<std::string> select = key_fields;
    auto add_field = [&](const std::string& f) {
        if (f.empty()) return;
        if (std::find(select.begin(), select.end(), f) == select.end())
            select.push_back(f);
    };
    for (const auto& spec : plan.agg) {
        // A derived value/by field (size/te) selects its typed derived column.
        add_field(value_select_token(spec.field));
        add_field(value_select_token(spec.by));
    }
    for (const DynAgg& d : dyn_aggs) add_field(d.column);
    if (has_bucket) add_field("ts");
    if (has_occ) {
        add_field("ts");
        add_field("dur");
    }

    auto next = std::make_shared<ViewPlan>(plan);
    next->group_by.clear();
    next->agg.clear();
    next->auto_numeric_metrics = false;
    next->numeric_arg_aggs.clear();
    next->sort_col.clear();
    next->topk_col.clear();
    next->offset = 0;
    next->limit = 0;
    next->select = select;
    next->schema.reset();
    next->resolver.reset();
    next->time_bucket_us = 0;
    next->bucket_origin_us = 0;
    next->bucket_origin_min = false;
    // The raw scan (native_row_fold.cpp) pre-scales+rounds ts/dur to the
    // nearest integer via time_scale before this function ever sees them; the
    // GroupMap fold instead applies time_scale as an unrounded double, once,
    // at bucket/aggregate time (agg_fold.h). Bucketing needs the exact
    // fold formula, so pull the raw (unscaled) values here and reapply
    // time_scale ourselves below, byte-for-byte like the fold.
    // Occupancy is a time-native reduction over raw ts/dur (agg_fold.h reads
    // them unscaled); the raw scan must not pre-scale them. A scaled value agg
    // (ts/dur/te) is rescaled the same unrounded way below, so it too reads
    // raw.
    if (has_bucket || has_occ || needs_value_scale) next->time_scale = 1.0;

    View raw(std::move(next));
    dataframe::LazyFrame lf =
        dataframe::LazyFrame::scan(std::make_shared<ViewSource>(raw))
            .memory_budget(plan.memory_budget);

    // Group-key transforms (dirname/basename/lower/bucket) coarsen the key:
    // many raw values fold to one, so they must be applied BEFORE the group-by,
    // not by relabeling the finalized result (which cannot re-merge). The
    // engine has no dirname/basename/bucket string expr, so materialize the raw
    // scan and build each transformed key column in C++, matching
    // resolve_group_keys (resolve_group_value then apply_group_transform) byte
    // for byte, then group over the in-memory frame.
    std::vector<char> key_transformed(plan.group_by.size(), 0);
    std::vector<std::string> tf_col(plan.group_by.size());
    const bool needs_transform = std::any_of(
        plan.group_by.begin(), plan.group_by.end(), [](const GroupKey& gk) {
            return gk.transform != GroupKey::Transform::None;
        });
    if (needs_transform) {
        dataframe::DataFrame frame = co_await lf.collect();
        const bool wants_resolver = std::any_of(
            plan.group_by.begin(), plan.group_by.end(), [](const GroupKey& gk) {
                return gk.transform != GroupKey::Transform::None &&
                       key_is_resolved(gk.kind);
            });
        const GroupResolver* resolver =
            wants_resolver ? ensure_resolver(plan) : nullptr;
        const std::int64_t n = frame.num_rows();
        for (std::size_t i = 0; i < plan.group_by.size(); ++i) {
            const GroupKey& gk = plan.group_by[i];
            if (gk.transform == GroupKey::Transform::None) continue;
            const std::int64_t ci = frame.column_index(key_fields[i]);
            const dataframe::Series& src =
                frame.columns[static_cast<std::size_t>(ci)];
            std::vector<std::string> vals(static_cast<std::size_t>(n));
            for (std::int64_t r = 0; r < n; ++r)
                vals[static_cast<std::size_t>(r)] = apply_group_transform(
                    gk, transform_key_base(gk, cell_to_key_string(src, r),
                                           resolver));
            tf_col[i] = "__view_agg_engine_tf_" + std::to_string(i);
            frame.names.push_back(tf_col[i]);
            frame.columns.push_back(dataframe::Series::strings(vals));
            key_transformed[i] = 1;
        }
        lf =
            dataframe::lazy(std::move(frame)).memory_budget(plan.memory_budget);
    }

    // cat is a computed key: the GroupMap path lowercases it for grouping only
    // (agg_fold.h's lower_ascii) while a value agg (e.g. SetUnion(cat)) still
    // sees the raw-case text, so the lowered key is materialized into a hidden
    // column rather than overwriting "cat" in place.
    static constexpr const char* CAT_KEY_COL = "__view_agg_engine_cat_key";
    static constexpr const char* BUCKET_KEY_COL =
        "__view_agg_engine_time_bucket";
    std::vector<std::string> group_key_names = key_fields;
    for (std::size_t i = 0; i < plan.group_by.size(); ++i)
        if (key_transformed[i]) group_key_names[i] = tf_col[i];
    for (std::size_t i = 0; i < plan.group_by.size(); ++i) {
        if (plan.group_by[i].kind != GroupKey::Kind::Cat) continue;
        if (key_transformed[i]) continue;  // transform path lowercased it
        group_key_names[i] = CAT_KEY_COL;
        lf = lf.with_column(CAT_KEY_COL,
                            dataframe::expr_lower(dataframe::expr_col(
                                static_cast<std::int32_t>(i))));
    }

    // Bucket key: match agg_fold.h's fold_event_over exactly. `ts` there is the
    // raw (unscaled) timestamp; the fold applies time_scale itself, floors
    // (ts*scale - origin)/interval toward -inf, then rescales by the interval
    // width and shifts back by origin. All in int64 once floored, so the
    // multiply-back and origin add cannot introduce float error the fold does
    // not also have.
    if (has_bucket) {
        const auto ts_it = std::find(select.begin(), select.end(), "ts");
        const auto ts_idx = static_cast<std::int32_t>(ts_it - select.begin());
        const double interval = static_cast<double>(plan.time_bucket_us);
        const auto w = static_cast<std::int64_t>(plan.time_bucket_us);
        const auto origin = static_cast<std::int64_t>(plan.bucket_origin_us);
        dataframe::Expr ts_d = dataframe::expr_cast(
            dataframe::TypeId::Float64, dataframe::expr_col(ts_idx));
        dataframe::Expr rel = ts_d * dataframe::expr_lit(plan.time_scale) -
                              dataframe::expr_lit(static_cast<double>(origin));
        dataframe::Expr floored = dataframe::expr_unary(
            static_cast<std::int32_t>(dataframe::UnaryOp::Floor),
            rel / dataframe::expr_lit(interval));
        dataframe::Expr bucket =
            dataframe::expr_cast(dataframe::TypeId::Int64, floored) *
                dataframe::expr_lit(w) +
            dataframe::expr_lit(origin);
        lf = lf.with_column(BUCKET_KEY_COL, bucket);
        group_key_names.insert(group_key_names.begin(), BUCKET_KEY_COL);
    }

    // Value fields that need the same unrounded time_scale (agg_fold.h's
    // field_scaled: ts/dur/te). The raw scan left them unscaled (time_scale
    // reset above), so multiply each referenced one by time_scale exactly as
    // the fold does. Only materialize the columns an agg spec references, keyed
    // by the select token (te reads its derived column).
    if (needs_value_scale) {
        bool need_ts = false, need_dur = false, need_te = false;
        for (const auto& spec : plan.agg) {
            if (is_occupancy_op(spec.op)) continue;
            need_ts = need_ts || spec.field == "ts" || spec.by == "ts";
            need_dur = need_dur || spec.field == "dur" || spec.by == "dur";
            need_te = need_te || spec.field == "te" || spec.by == "te";
        }
        auto scale_col = [&](const std::string& tok, const char* out) {
            const auto it = std::find(select.begin(), select.end(), tok);
            if (it == select.end()) return;
            const auto idx = static_cast<std::int32_t>(it - select.begin());
            lf = lf.with_column(out,
                                dataframe::expr_cast(dataframe::TypeId::Float64,
                                                     dataframe::expr_col(idx)) *
                                    dataframe::expr_lit(plan.time_scale));
        };
        if (need_ts) scale_col("ts", SCALED_TS_COL);
        if (need_dur) scale_col("dur", SCALED_DUR_COL);
        if (need_te) scale_col(value_select_token("te"), SCALED_TE_COL);
    }

    std::vector<DynFix> dynfix;
    dynfix.reserve(dyn_aggs.size());
    for (const DynAgg& d : dyn_aggs)
        dynfix.push_back({d.out, d.op == AggOp::Pct, d.op == AggOp::Count});

    co_return EnginePrep{std::move(lf), std::move(group_key_names),
                         std::move(gaggs), std::move(dynfix)};
}

coro::CoroTask<dataframe::DataFrame> run_collect_via_engine(
    const ViewPlan& plan_in) {
    const ViewPlan plan = resolve_bucket_origin(plan_in);
    ensure_schema(plan);

    // The rollup carries occupancy delta-maps (bootstrap/tier do not), so it is
    // tried first and is the only fast path that can serve occupancy.
    if (auto df = try_serve_rollup(plan)) co_return std::move(*df);
    {
        GroupMap served;
        if (co_await try_serve_aggregate_no_scan(plan, served))
            co_return finalize_collect_batch(served, plan);
    }

    EnginePrep ep = co_await prepare_engine_group(plan);

    // materialize() persists the AggState partials as a rollup (opt-in); a
    // paginated result is never cached.
    if (plan.materialize && !plan.limit && !plan.offset) {
        auto state =
            co_await ep.lf->collect_group_state(ep.group_key_names, ep.gaggs);
        const std::string rdir = rollup_index_path(plan);
        if (!rdir.empty()) {
            try {
                auto db = open_rollup_db(
                    rdir, rocksdb::RocksDatabase::OpenMode::ReadWrite);
                if (db)
                    persist_rollup(*db, plan_signature(plan),
                                   rest_signature(plan), plan.time_bucket_us,
                                   plan.group_by, *state);
            } catch (const std::exception& e) {
                DFTRACER_UTILS_LOG_WARN("rollup materialize skipped: %s",
                                        e.what());
            }
        }
        co_return finalize_engine_result(*state, plan);
    }

    dataframe::DataFrame r =
        co_await ep.lf->group_by(ep.group_key_names, ep.gaggs).collect();
    co_return finalize_engine_frame(std::move(r), plan, ep.dynfix);
}

coro::CoroTask<dataframe::AggStatePtr> build_engine_agg_state(
    const ViewPlan& plan_in) {
    const ViewPlan plan = resolve_bucket_origin(plan_in);
    ensure_schema(plan);
    EnginePrep ep = co_await prepare_engine_group(plan);
    co_return co_await ep.lf->collect_group_state(ep.group_key_names, ep.gaggs);
}

}  // namespace dftracer::utils::trace::views::detail
