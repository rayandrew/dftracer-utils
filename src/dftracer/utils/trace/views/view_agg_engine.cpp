#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/dataframe/agg.h>
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/trace/views/aggfold.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/native_row_fold.h>
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

// The always-present top-level numeric fields (plus name/cat). A stable field's
// scanned column carries a value in every group, so its accumulation domain is
// uniform; a possibly-absent arg field is not.
bool is_stable_field(const std::string& f) {
    return f.empty() || f == "name" || f == "cat" || f == "pid" || f == "tid" ||
           f == "ts" || f == "dur";
}

// A field agg_field_typed_t/agg_field_t derives instead of reading straight
// (size = io-cat byte size, te = ts+dur). A plain arg/top-level column cannot
// reproduce these, so an agg over one stays on the GroupMap path.
bool is_derived_field(const std::string& f) { return f == "size" || f == "te"; }

// A single-segment field name (no nested dot/bracket path). The GroupMap fold
// resolves a nested value field (args.n.v) through number_typed, but the raw
// scan feeds a value column only by a flat top-level or arg name, so a nested
// value/by field stays on the GroupMap path.
bool is_simple_field(const std::string& f) {
    return f.find('.') == std::string::npos && f.find('[') == std::string::npos;
}

// Sum/Min/Max/SumSq keep an integer field's exact domain, but the GroupMap path
// demotes a group with zero present values to the Float64 default, so the whole
// column widens to Float64 when a field is absent from any group. The engine's
// per-column domain cannot reproduce that data-dependent widening, so a domain-
// sensitive reduction stays on the stable (always-present) fields.
bool is_domain_sensitive(AggOp op) {
    return op == AggOp::Sum || op == AggOp::Min || op == AggOp::Max ||
           op == AggOp::SumSq;
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
        g.column = value_col_name(spec.field);
        g.by = value_col_name(spec.by);
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
            g.column = value_col_name(spec.field);
        }
    } else {
        g.column = value_col_name(spec.field);
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

}  // namespace

bool agg_engine_eligible(const ViewPlan& plan) {
    if (plan.group_by.empty() && plan.time_bucket_us == 0) return false;
    // Every GroupKey::Kind and every transform is now handled by
    // run_collect_via_engine (a resolved/computed/arg key folds on its scanned
    // column; a transform materializes its coarsened key pre-group), so the
    // only remaining gates are on the aggregate specs below.
    for (const AggSpec& r : plan.numeric_arg_aggs)
        if (is_derived_field(r.field)) return false;

    for (const auto& spec : plan.agg) {
        if (!agg_op_engine_supported(spec.op)) return false;
        // Occupancy reads raw ts/dur; under a non-identity time_scale the
        // GroupMap path splits raw-occupancy from scaled value aggs per field,
        // which the single-scan engine path cannot reproduce, so defer to it.
        if (is_occupancy_op(spec.op) && plan.time_scale != 1.0) return false;
        // size/te are derived at fold time (agg_field_typed_t); a plain scanned
        // column cannot reproduce them, so an agg over one stays on GroupMap.
        if (is_derived_field(spec.field) || is_derived_field(spec.by))
            return false;
        // A domain-sensitive reduction over a possibly-absent arg field cannot
        // match GroupMap's per-group integer/float demotion (see above).
        if (is_domain_sensitive(spec.op) && !is_stable_field(spec.field))
            return false;
        // A nested value/by path is resolved by the fold but not by the flat
        // value-column scan, so it stays on the GroupMap path.
        if (!is_simple_field(spec.field) || !is_simple_field(spec.by))
            return false;
    }
    return true;
}

coro::CoroTask<dataframe::DataFrame> run_collect_via_engine(
    const ViewPlan& plan_in) {
    const ViewPlan plan = resolve_bucket_origin(plan_in);
    ensure_schema(plan);

    {
        GroupMap served;
        if (co_await try_serve_aggregate_no_scan(plan, served))
            co_return finalize_collect_batch(served, plan);
    }

    // A Rank key groups on pid and relabels to the PR-metadata rank; harvest
    // that map into the resolver before the scan, mirroring run_scan_aggregate.
    if (std::any_of(
            plan.group_by.begin(), plan.group_by.end(),
            [](const GroupKey& gk) { return gk.kind == GroupKey::Kind::Rank; }))
        co_await harvest_ranks(plan);

    const bool has_bucket = plan.time_bucket_us > 0;

    std::vector<std::string> key_names;
    std::vector<std::string> key_fields;
    key_names.reserve(plan.group_by.size());
    key_fields.reserve(plan.group_by.size());
    for (const GroupKey& gk : plan.group_by) {
        key_names.push_back(group_col_name(gk));
        key_fields.push_back(key_group_field(gk));
    }

    // Scaled-field renaming (bucket only, see below): ts/dur route through a
    // hidden pre-scaled column instead of the raw field name.
    static constexpr const char* SCALED_TS_COL = "__view_agg_engine_scaled_ts";
    static constexpr const char* SCALED_DUR_COL =
        "__view_agg_engine_scaled_dur";
    const bool needs_value_scale = has_bucket && plan.time_scale != 1.0;
    auto scaled_name = [&](const std::string& f) -> std::string {
        if (!needs_value_scale) return f;
        if (f == "ts") return SCALED_TS_COL;
        if (f == "dur") return SCALED_DUR_COL;
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
            g.column = scaled_name(g.column);
            g.by = scaled_name(g.by);
            if (is_occupancy_op(spec.op)) g.param = plan.occ_cell_us;
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
    // Value columns precede the text (ArgMax/SetUnion) columns in to_batch;
    // busy_cell_us (below) slots in right after them, so capture the count now.
    const std::size_t n_value_cols = gaggs.size();
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
        add_field(spec.field);
        add_field(spec.by);
    }
    for (const DynAgg& d : dyn_aggs) add_field(d.column);
    if (has_bucket) add_field("ts");
    const bool has_occ =
        std::any_of(plan.agg.begin(), plan.agg.end(),
                    [](const AggSpec& s) { return is_occupancy_op(s.op); });
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
    // them unscaled); the raw scan must not pre-scale them.
    if (has_bucket || has_occ) next->time_scale = 1.0;

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
    std::optional<std::size_t> cat_pos;
    for (std::size_t i = 0; i < plan.group_by.size(); ++i) {
        if (plan.group_by[i].kind != GroupKey::Kind::Cat) continue;
        if (key_transformed[i]) continue;  // transform path lowercased it
        cat_pos = i;
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

        // Value fields that need the same unrounded time_scale (agg_fold.h's
        // field_scaled: ts/dur; te is derived, never engine-eligible). Only
        // materialize the ones an agg spec actually references.
        if (needs_value_scale) {
            bool need_ts = false, need_dur = false;
            for (const auto& spec : plan.agg) {
                need_ts = need_ts || spec.field == "ts" || spec.by == "ts";
                need_dur = need_dur || spec.field == "dur" || spec.by == "dur";
            }
            auto scale_col = [&](const std::string& field, const char* out) {
                const auto it = std::find(select.begin(), select.end(), field);
                const auto idx = static_cast<std::int32_t>(it - select.begin());
                lf = lf.with_column(
                    out, dataframe::expr_cast(dataframe::TypeId::Float64,
                                              dataframe::expr_col(idx)) *
                             dataframe::expr_lit(plan.time_scale));
            };
            if (need_ts) scale_col("ts", SCALED_TS_COL);
            if (need_dur) scale_col("dur", SCALED_DUR_COL);
        }
    }

    dataframe::DataFrame r =
        co_await lf.group_by(group_key_names, gaggs).collect();
    const std::size_t off = has_bucket ? 1 : 0;
    if (has_bucket) r.names[0] = "time_bucket";
    if (cat_pos) r.names[off + *cat_pos] = "cat";
    // An Arg/Field key (or any transformed key, grouped on a hidden column)
    // relabels to its user-facing group_col_name (== key_names[j]).
    for (std::size_t j = 0; j < plan.group_by.size(); ++j) {
        const GroupKey::Kind k = plan.group_by[j].kind;
        if (key_transformed[j] || k == GroupKey::Kind::Arg ||
            k == GroupKey::Kind::Field)
            r.names[off + j] = key_names[j];
    }
    for (std::size_t i = 0; i < off + key_names.size(); ++i)
        r.columns[i] = key_column_to_string(r.columns[i]);

    // to_batch appends a busy_cell_us column (the effective occ_cell tolerance)
    // right after the value columns whenever a Busy/Concurrency/Utilization
    // spec is present (Active alone does not emit it). Reproduce it as an Int64
    // constant so the two paths match column-for-column.
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
        const std::size_t at = off + key_names.size() + n_value_cols;
        r.names.insert(r.names.begin() + static_cast<std::ptrdiff_t>(at),
                       "busy_cell_us");
        r.columns.insert(r.columns.begin() + static_cast<std::ptrdiff_t>(at),
                         dataframe::Series::flat_i64(cell.data(), nrows));
    }

    // Post-aggregation re-key: relabel each resolved-name key column (grouped
    // on its raw hash, a bijection) to the resolved name, same as
    // resolve_group_keys/resolve_group_value in the GroupMap path. Runs on the
    // small distinct-group result, never per event.
    bool needs_resolver = false;
    for (std::size_t j = 0; j < plan.group_by.size(); ++j)
        needs_resolver =
            needs_resolver ||
            (key_is_resolved(plan.group_by[j].kind) && !key_transformed[j]);
    if (needs_resolver) {
        const GroupResolver* resolver = ensure_resolver(plan);
        for (std::size_t j = 0; j < plan.group_by.size(); ++j) {
            // A transformed resolved key already holds the resolved+transformed
            // string (built pre-group); do not re-resolve it here.
            if (!key_is_resolved(plan.group_by[j].kind) || key_transformed[j])
                continue;
            const std::size_t idx = off + j;
            r.columns[idx] = resolve_key_column(r.columns[idx], *resolver,
                                                plan.group_by[j].kind);
            r.names[idx] = key_names[j];
        }
    }

    // A dyn Pct on a group where the arg never appeared: the engine reads an
    // empty per-group DDSketch, whose quantile is NaN, while the GroupMap dyn
    // path emits 0.0 (dyn_sketches has no entry for the arg). Rewrite that NaN
    // to 0.0 so the two paths match. Only an empty sketch yields NaN, so this
    // touches exactly the absent-arg groups.
    for (const DynAgg& d : dyn_aggs) {
        if (d.op != AggOp::Pct) continue;
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

    // A dyn Count is emitted as Int64 by CountValid but the GroupMap dyn path
    // renders every dyn column Float64 (reduce_dyn returns a double); cast to
    // match column-for-column.
    for (const DynAgg& d : dyn_aggs) {
        if (d.op != AggOp::Count) continue;
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
    co_return r;
}

}  // namespace dftracer::utils::trace::views::detail
