// DFTU_EXT_AGG: a columnar fold accumulates each batch's key + value columns
// into the dataframe engine's cross-batch AggState, merged across worker slices
// and finalized to a native dataframe result (no Arrow hop).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/plugins/fold_adapter.h>

#include <memory>
// After fold_adapter.h so nanoarrow is set up before dataframe/abi.h's
// arrow_abi.
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/agg_op_codes.h>
#include <dftracer/utils/plugins/plugin.h>
#include <dftracer/utils/trace/schema.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/fold_event.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

using dftracer::utils::CoroScope;
using dftracer::utils::Runtime;
using dftracer::utils::StringIntern;
using dftracer::utils::plugins::AggCol;
using dftracer::utils::plugins::Host;
using dftracer::utils::plugins::NamedResultRegistry;
using dftracer::utils::plugins::OwnedDataFrame;
using dftracer::utils::plugins::PluginFold;
using dftracer::utils::trace::RecordPhase;
namespace agg = dftracer::utils::plugins::agg;
namespace coro = dftracer::utils::coro;
namespace views = dftracer::utils::trace::views;
using views::detail::FoldBatch;
using views::detail::FoldEvent;
using views::detail::ScanUnit;

namespace {

FoldEvent evt(StringIntern& intern, const char* cat, const char* name,
              std::uint64_t dur) {
    FoldEvent fe;
    fe.cat_id = intern.get_or_insert(cat);
    fe.name_id = intern.get_or_insert(name);
    fe.phase = RecordPhase::COMPLETE;
    fe.ts = 1;
    fe.dur = dur;
    fe.has_dur = true;
    return fe;
}

// A raw columnar plugin: on_batch_columns builds one dft.ext.agg accumulator
// keyed by "cat" with sum(dur), p50(dur) and set_union(name), and folds each
// batch's columns into it. The host merges and finalizes it to a native
// dataframe named "by_cat".
::dftu_task* agg_on_batch_columns(void* slice, const dftu_dataframe* df,
                                  const dftu_host* host) {
    (void)slice;
    const auto* agg = static_cast<const dftu_ext_agg*>(
        host->get_extension(host->h, DFTU_EXT_AGG));
    if (!agg) return nullptr;
    const dftu_agg_col specs[] = {
        {DFTU_AGG_SUM, "dur", "sum_dur", 0.0, nullptr},
        {DFTU_AGG_PCT, "dur", "p50_dur", 0.5, nullptr},
        {DFTU_AGG_SET_UNION, "name", "names", 0.0, nullptr},
    };
    const char* keys[] = {"cat"};
    dftu_agg* a = agg->agg_new(host->h, "by_cat", keys, 1, specs, 3);
    if (a) agg->agg_accumulate(host->h, a, df);
    return nullptr;
}

// Same accumulator, built through the agg:: factories + Host::agg instead of
// raw dftu_agg_col structs, proving each factory produces the same wire spec.
::dftu_task* agg_factory_on_batch_columns(void* slice, const dftu_dataframe* df,
                                          const dftu_host* host) {
    (void)slice;
    Host h(host);
    const auto acc =
        h.agg("by_cat_factory", {"cat"},
              {agg::mean("dur", "mean_dur"), agg::pct("dur", "p50_dur", 0.5)});
    if (acc) acc.accumulate(df);
    return nullptr;
}

dftu_plugin make_agg_factory_plugin() {
    dftu_plugin p{};
    p.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    p.needs = [](void*) -> std::uint32_t { return 0; };
    p.plan_query = [](void*) -> const char* { return nullptr; };
    p.make_slice = [](void*) -> void* {
        static int sentinel;
        return &sentinel;
    };
    p.merge = [](void*, void*) {};
    p.on_finalize = [](void*, const dftu_host*) -> ::dftu_task* {
        return nullptr;
    };
    p.destroy_slice = [](void*) {};
    p.destroy = [](void*) {};
    p.on_batch_columns = agg_factory_on_batch_columns;
    return p;
}

dftu_plugin make_agg_plugin() {
    dftu_plugin p{};
    p.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    p.needs = [](void*) -> std::uint32_t { return 0; };
    p.plan_query = [](void*) -> const char* { return nullptr; };
    // This plugin holds no per-slice state (the accumulator lives in the host's
    // PluginFold), so a non-null sentinel is all PluginFold::step needs - no
    // per-slice allocation.
    p.make_slice = [](void*) -> void* {
        static int sentinel;
        return &sentinel;
    };
    p.merge = [](void*, void*) {};
    p.on_finalize = [](void*, const dftu_host*) -> ::dftu_task* {
        return nullptr;
    };
    p.destroy_slice = [](void*) {};
    p.destroy = [](void*) {};
    p.on_batch_columns = agg_on_batch_columns;
    return p;
}

void finalize_now(PluginFold& f) {
    Runtime rt(1);
    rt.scope("fin", [&](CoroScope&) -> coro::CoroTask<void> {
          co_await f.finalize(views::detail::CoverageSet{});
      }).wait();
    rt.shutdown();
}

// One flat numeric cell as a double (Int64/UInt64/Float64 result columns).
double num_at(const dftu_series* col, std::int64_t r) {
    dftu_series* flat = dftu_series_materialize(col);
    const void* d = dftu_series_data(flat);
    double out = 0.0;
    switch (dftu_series_type(flat)) {
        case DFTU_TYPE_INT64:
            out = static_cast<double>(static_cast<const std::int64_t*>(d)[r]);
            break;
        case DFTU_TYPE_UINT64:
            out = static_cast<double>(static_cast<const std::uint64_t*>(d)[r]);
            break;
        case DFTU_TYPE_FLOAT64:
            out = static_cast<const double*>(d)[r];
            break;
        default:
            break;
    }
    dftu_series_free(flat);
    return out;
}

std::string str_at(const dftu_series* col, std::int64_t r) {
    dftu_series* flat = dftu_series_materialize(col);
    const std::int32_t* off = dftu_series_offsets(flat);
    const char* data = static_cast<const char*>(dftu_series_data(flat));
    std::string out(data + off[r],
                    static_cast<std::size_t>(off[r + 1] - off[r]));
    dftu_series_free(flat);
    return out;
}

}  // namespace

TEST_CASE("DFTU_EXT_AGG: cross-batch grouped aggregation, engine-backed") {
    StringIntern intern;
    dftu_plugin p = make_agg_plugin();
    NamedResultRegistry named;
    PluginFold master(&p, intern, nullptr, &named);

    // Two slices (two workers), each one batch, merged into the master.
    std::vector<std::vector<FoldEvent>> slices = {
        {evt(intern, "POSIX", "read", 10), evt(intern, "POSIX", "write", 20),
         evt(intern, "STDIO", "open", 5)},
        {evt(intern, "POSIX", "read", 30), evt(intern, "STDIO", "close", 7)},
    };
    for (const auto& evs : slices) {
        auto slice = master.slice();
        auto* pf = static_cast<PluginFold*>(slice.get());
        ScanUnit unit{};
        pf->step(FoldBatch{std::span<const FoldEvent>(evs), unit, {}});
        master.merge(*pf);
    }
    finalize_now(master);

    auto it = named.results().find("by_cat");
    REQUIRE(it != named.results().end());
    REQUIRE(std::holds_alternative<OwnedDataFrame>(it->second));
    dftu_dataframe* out = std::get<OwnedDataFrame>(it->second).handle;
    REQUIRE(out != nullptr);
    const std::int64_t n = dftu_dataframe_num_rows(out);
    REQUIRE(n == 2);

    dftu_series* cat = dftu_dataframe_column(out, "cat");
    dftu_series* sum = dftu_dataframe_column(out, "sum_dur");
    dftu_series* p50 = dftu_dataframe_column(out, "p50_dur");
    dftu_series* names = dftu_dataframe_column(out, "names");
    REQUIRE(cat);
    REQUIRE(sum);
    REQUIRE(p50);
    REQUIRE(names);

    std::map<std::string, double> got_sum;
    std::map<std::string, double> got_p50;
    std::map<std::string, std::string> got_names;
    for (std::int64_t r = 0; r < n; ++r) {
        std::string k = str_at(cat, r);
        got_sum[k] = num_at(sum, r);
        got_p50[k] = num_at(p50, r);
        got_names[k] = str_at(names, r);
    }

    // Sums are exact across the merged slices.
    CHECK(got_sum["POSIX"] == doctest::Approx(60.0));  // 10 + 20 + 30
    CHECK(got_sum["STDIO"] == doctest::Approx(12.0));  // 5 + 7

    // Pct is a mergeable DDSketch quantile; assert it lands in range and near
    // the true median (relative accuracy, not exact).
    CHECK(got_p50["POSIX"] == doctest::Approx(20.0).epsilon(0.05));
    CHECK(got_p50["STDIO"] >= 5.0);
    CHECK(got_p50["STDIO"] <= 7.0);

    // SetUnion is the distinct value set the map path could not express.
    CHECK(got_names["POSIX"].find("read") != std::string::npos);
    CHECK(got_names["POSIX"].find("write") != std::string::npos);
    CHECK(got_names["STDIO"].find("open") != std::string::npos);
    CHECK(got_names["STDIO"].find("close") != std::string::npos);

    dftu_series_free(cat);
    dftu_series_free(sum);
    dftu_series_free(p50);
    dftu_series_free(names);

    // Parity: the same grouped sum via the dftu_dataframe_group_by primitive
    // the legacy DFTU_EXT_MAP vfold codegen used, over the concatenated rows.
    const char* cats[] = {"POSIX", "POSIX", "STDIO", "POSIX", "STDIO"};
    const std::uint64_t durs[] = {10, 20, 5, 30, 7};
    std::vector<std::int32_t> offs;
    std::string blob;
    offs.push_back(0);
    for (const char* c : cats) {
        blob += c;
        offs.push_back(static_cast<std::int32_t>(blob.size()));
    }
    dftu_series* keycol = dftu_series_new_string(DFTU_TYPE_STRING, offs.data(),
                                                 blob.data(), 5, nullptr);
    dftu_series* valcol =
        dftu_series_new_flat(DFTU_TYPE_UINT64, durs, 5, nullptr);
    dftu_series* gk = nullptr;
    dftu_series* gv[1] = {nullptr};
    std::int32_t nout =
        dftu_dataframe_group_by(keycol, valcol, DFTU_REDUCE_SUM, &gk, gv, 1);
    REQUIRE(nout == 1);
    REQUIRE(gk);
    REQUIRE(gv[0]);
    std::map<std::string, double> ref_sum;
    for (std::int64_t r = 0; r < dftu_series_length(gk); ++r)
        ref_sum[str_at(gk, r)] = num_at(gv[0], r);
    CHECK(ref_sum["POSIX"] == doctest::Approx(got_sum["POSIX"]));
    CHECK(ref_sum["STDIO"] == doctest::Approx(got_sum["STDIO"]));

    dftu_series_free(keycol);
    dftu_series_free(valcol);
    dftu_series_free(gk);
    dftu_series_free(gv[0]);
}

TEST_CASE(
    "agg:: factories build the same dftu_agg_col wire fields as the raw "
    "builder") {
    const AggCol mean_c = agg::mean("dur", "mean_dur");
    const dftu_agg_col mean_raw = mean_c.raw();
    CHECK(mean_raw.op == DFTU_AGG_MEAN);
    CHECK(std::string(mean_raw.value) == "dur");
    CHECK(std::string(mean_raw.out) == "mean_dur");
    CHECK(mean_raw.by == nullptr);

    const dftu_agg_col pct_raw = agg::pct("dur", "p50_dur", 0.5).raw();
    CHECK(pct_raw.op == DFTU_AGG_PCT);
    CHECK(std::string(pct_raw.value) == "dur");
    CHECK(pct_raw.param == doctest::Approx(0.5));
    CHECK(pct_raw.by == nullptr);

    const dftu_agg_col count_raw = agg::count("n").raw();
    CHECK(count_raw.op == DFTU_AGG_COUNT);
    CHECK(count_raw.value == nullptr);
    CHECK(std::string(count_raw.out) == "n");

    const dftu_agg_col argmax_raw =
        agg::argmax("name", "name_at_max", "dur").raw();
    CHECK(argmax_raw.op == DFTU_AGG_ARGMAX);
    CHECK(std::string(argmax_raw.value) == "name");
    CHECK(std::string(argmax_raw.by) == "dur");

    const dftu_agg_col busy_raw = agg::busy("ts", "dur", "busy_us", 2.0).raw();
    CHECK(busy_raw.op == DFTU_AGG_BUSY);
    CHECK(std::string(busy_raw.value) == "ts");
    CHECK(std::string(busy_raw.by) == "dur");
    CHECK(busy_raw.param == doctest::Approx(2.0));
}

TEST_CASE(
    "agg:: factories through Host::agg accumulate the same as raw dftu_agg_col "
    "specs") {
    StringIntern intern;
    dftu_plugin p = make_agg_factory_plugin();
    NamedResultRegistry named;
    PluginFold master(&p, intern, nullptr, &named);

    std::vector<std::vector<FoldEvent>> slices = {
        {evt(intern, "POSIX", "read", 10), evt(intern, "POSIX", "write", 20),
         evt(intern, "STDIO", "open", 5)},
        {evt(intern, "POSIX", "read", 30), evt(intern, "STDIO", "close", 7)},
    };
    for (const auto& evs : slices) {
        auto slice = master.slice();
        auto* pf = static_cast<PluginFold*>(slice.get());
        ScanUnit unit{};
        pf->step(FoldBatch{std::span<const FoldEvent>(evs), unit, {}});
        master.merge(*pf);
    }
    finalize_now(master);

    auto it = named.results().find("by_cat_factory");
    REQUIRE(it != named.results().end());
    REQUIRE(std::holds_alternative<OwnedDataFrame>(it->second));
    dftu_dataframe* out = std::get<OwnedDataFrame>(it->second).handle;
    REQUIRE(out != nullptr);
    const std::int64_t n = dftu_dataframe_num_rows(out);
    REQUIRE(n == 2);

    dftu_series* cat = dftu_dataframe_column(out, "cat");
    dftu_series* mean_dur = dftu_dataframe_column(out, "mean_dur");
    dftu_series* p50_dur = dftu_dataframe_column(out, "p50_dur");
    REQUIRE(cat);
    REQUIRE(mean_dur);
    REQUIRE(p50_dur);

    std::map<std::string, double> got_mean;
    std::map<std::string, double> got_p50;
    for (std::int64_t r = 0; r < n; ++r) {
        std::string k = str_at(cat, r);
        got_mean[k] = num_at(mean_dur, r);
        got_p50[k] = num_at(p50_dur, r);
    }

    // POSIX: durs {10, 20, 30}; STDIO: durs {5, 7}.
    CHECK(got_mean["POSIX"] == doctest::Approx(20.0));
    CHECK(got_mean["STDIO"] == doctest::Approx(6.0));
    CHECK(got_p50["POSIX"] == doctest::Approx(20.0).epsilon(0.05));
    CHECK(got_p50["STDIO"] >= 5.0);
    CHECK(got_p50["STDIO"] <= 7.0);

    dftu_series_free(cat);
    dftu_series_free(mean_dur);
    dftu_series_free(p50_dur);
}
