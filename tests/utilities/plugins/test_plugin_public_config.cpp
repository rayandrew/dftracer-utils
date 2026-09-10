// Builder::add(path, ConfigTree) takes a type only reachable through the
// private src/ header, so a caller with only <dftracer/utils/plugins/
// plugins.h> could not build config for it. This file includes ONLY the
// public header (no plugins/config.h) to prove the JSON-string overload
// closes that gap: a config literal, loaded and its values checked through a
// real plugin run, and a malformed literal failing at build() rather than
// add().

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/plugins/plugins.h>
#include <dftracer/utils/trace/internal/utils.h>
#include <dftracer/utils/trace/views/view.h>
#include <doctest/doctest.h>
#include <testing_utilities.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

using dftracer::utils::CoroScope;
using dftracer::utils::Runtime;
using dftracer::utils::plugins::PluginRun;
using dftracer::utils::plugins::Plugins;
using dftracer::utils::trace::internal::determine_index_path;
using View = dftracer::utils::trace::views::View;
using ViewFile = dftracer::utils::trace::views::ViewFile;
using ExportSink = dftracer::utils::trace::views::ExportSink;
namespace coro = dftracer::utils::coro;
namespace fs = std::filesystem;

#ifndef CONFIG_KEYS_PLUGIN_PATH
#error "CONFIG_KEYS_PLUGIN_PATH must be defined by CMake"
#endif

namespace {

constexpr int NUM_EVENTS = 5;

ViewFile make_trace(dftu_utils_test::TestEnvironment& env) {
    const std::string pfw = env.get_dir() + "/public_config.pfw";
    std::ofstream ofs(pfw);
    for (int i = 0; i < NUM_EVENTS; ++i) {
        ofs << R"({"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,"ts":)"
            << (1000 + i * 100) << R"(,"dur":10,"args":{}})"
            << "\n";
    }
    ofs.close();

    const std::string gz = pfw + ".gz";
    dftu_utils_test::compress_file_to_gzip(pfw, gz);
    fs::remove(pfw);

    const std::string idx = determine_index_path(gz, "");
    struct Sink : ExportSink {
        void write(std::string_view) override {}
    } sink;
    View::from_file(gz, idx).metadata(false).export_json(sink).get();
    return ViewFile{gz, idx};
}

PluginRun run_set(const Plugins& set, const View& view) {
    PluginRun out;
    Runtime rt(2);
    auto task = dftracer::utils::run_coro_scope(
        rt.executor(), [&](CoroScope&) -> coro::CoroTask<void> {
            auto run = co_await set.run(view);
            REQUIRE(run.has_value());
            out = std::move(*run);
            co_return;
        });
    rt.submit(std::move(task), "public-config").wait();
    rt.shutdown();
    return out;
}

std::string result_text(PluginRun& run, const char* name) {
    auto it = run.results.results().find(name);
    if (it == run.results.results().end()) return "(missing)";
    const auto& bytes = std::get<std::vector<std::byte>>(it->second);
    return std::string(reinterpret_cast<const char*>(bytes.data()),
                       bytes.size());
}

}  // namespace

TEST_CASE(
    "add(path, json) builds config from only the public header and the "
    "plugin receives it") {
    auto set = Plugins::builder()
                   .add(CONFIG_KEYS_PLUGIN_PATH,
                        std::string(R"({"label": "rows", "stride": 3})"))
                   .build();
    INFO((set.has_value() ? std::string{} : set.error().message));
    REQUIRE(set.has_value());

    dftu_utils_test::TestEnvironment env(0);
    REQUIRE(env.is_valid());
    View view = View::from_files({make_trace(env)});
    PluginRun run = run_set(*set, view);

    // rows = NUM_EVENTS * stride; a wrong stride (the config default is 1)
    // would report NUM_EVENTS instead.
    CHECK(result_text(run, "config_keys_plugin.rows") ==
          std::to_string(NUM_EVENTS * 3));
}

TEST_CASE("add(path, json) fails at build(), not at add()") {
    Plugins::Builder builder = Plugins::builder();
    REQUIRE_NOTHROW(
        builder.add(CONFIG_KEYS_PLUGIN_PATH, std::string("{not json")));

    auto set = builder.build();
    REQUIRE(!set.has_value());
    CHECK(set.error().message.find("does not parse") != std::string::npos);
}
