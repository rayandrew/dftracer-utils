// DFTU_EXT_OPS: exposes the dataframe engine's op registry to a plugin by
// name, so it can run any registered column/aggregate/frame op on a
// Series/DataFrame it already holds without linking the dataframe C ABI.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/fold_adapter.h>

#include <memory>
// After fold_adapter.h so nanoarrow is set up before dataframe/abi.h's
// arrow_abi.
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/plugins/plugin.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

using dftracer::utils::StringIntern;
using dftracer::utils::plugins::Host;
using dftracer::utils::plugins::PluginFold;

namespace {

struct TrivialSlice {
    explicit TrivialSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch&, Host) {}
    void merge(TrivialSlice&) {}
    void finalize(Host) {}
};

struct HostFixture {
    StringIntern intern;
    dftu_plugin* plugin =
        dftracer::utils::plugins::make_plugin<TrivialSlice>(nullptr);
    std::unique_ptr<PluginFold> fold =
        std::make_unique<PluginFold>(plugin, intern);
    dftu_host& host() { return fold->host(); }
    ~HostFixture() {
        fold.reset();
        if (plugin && plugin->destroy) plugin->destroy(plugin->self);
    }
};

}  // namespace

TEST_CASE("plugin ops: run_op runs a registered column op by name") {
    HostFixture fx;
    Host h{&fx.host()};

    const std::int64_t a_vals[] = {1, 2, 3};
    const std::int64_t b_vals[] = {10, 20, 30};
    dftu_series* a = dftu_series_new_flat(DFTU_TYPE_INT64, a_vals, 3, nullptr);
    dftu_series* b = dftu_series_new_flat(DFTU_TYPE_INT64, b_vals, 3, nullptr);
    REQUIRE(a);
    REQUIRE(b);

    dftu_series* sum = h.run_op("add", {a, b});
    REQUIRE(sum);
    REQUIRE(dftu_series_length(sum) == 3);
    const auto* sum_data =
        static_cast<const std::int64_t*>(dftu_series_data(sum));
    REQUIRE(sum_data);
    for (std::int64_t i = 0; i < 3; ++i)
        CHECK(sum_data[i] == a_vals[i] + b_vals[i]);

    dftu_series_free(sum);
    dftu_series_free(a);
    dftu_series_free(b);
}

TEST_CASE("plugin ops: run_op reports unknown names and bad arity") {
    HostFixture fx;
    Host h{&fx.host()};

    CHECK(h.run_op("dftu.test.nonexistent_op", {}) == nullptr);

    const std::int64_t a_vals[] = {1, 2, 3};
    dftu_series* a = dftu_series_new_flat(DFTU_TYPE_INT64, a_vals, 3, nullptr);
    REQUIRE(a);
    // "add" takes two series operands; one is an arity mismatch.
    CHECK(h.run_op("add", {a}) == nullptr);
    dftu_series_free(a);
}

TEST_CASE("plugin ops: find_op resolves a built-in by name") {
    HostFixture fx;
    Host h{&fx.host()};

    const dftu_op_desc* desc = h.find_op("add");
    REQUIRE(desc);
    CHECK(dftu_op_kind_of(desc->sig) == DFTU_OP_KIND_SERIES);
    CHECK(h.find_op("dftu.test.nonexistent_op") == nullptr);
}
