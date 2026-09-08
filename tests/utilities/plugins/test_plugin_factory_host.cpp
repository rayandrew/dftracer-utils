// The factory is the plugin's init: it receives a host and may register with
// it. What it may do there is enforced, not documented - the build-phase host
// answers only the registration groups, and a factory that reaches past them
// fails the load instead of running on a host that is not there.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/plugins.h>
#include <dftracer/utils/trace/internal/utils.h>
#include <dftracer/utils/trace/views/view.h>
#include <doctest/doctest.h>
#include <testing_utilities.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>
#include <vector>

using dftracer::utils::CoroScope;
using dftracer::utils::Runtime;
using dftracer::utils::plugins::NamedResult;
using dftracer::utils::plugins::PluginRun;
using dftracer::utils::plugins::Plugins;
using dftracer::utils::trace::internal::determine_index_path;
using View = dftracer::utils::trace::views::View;
using ViewFile = dftracer::utils::trace::views::ViewFile;
using ExportSink = dftracer::utils::trace::views::ExportSink;
namespace coro = dftracer::utils::coro;
namespace fs = std::filesystem;

#ifndef FACTORY_REGISTERS_OP_PLUGIN_PATH
#error "FACTORY_REGISTERS_OP_PLUGIN_PATH must be defined by CMake"
#endif
#ifndef FACTORY_ESCAPES_PLUGIN_PATH
#error "FACTORY_ESCAPES_PLUGIN_PATH must be defined by CMake"
#endif

namespace {

constexpr int EVENTS = 64;

ViewFile make_trace(dftu_utils_test::TestEnvironment& env,
                    const std::string& tag) {
    const std::string dir = env.get_dir() + "/" + tag;
    fs::create_directories(dir);
    const std::string pfw = dir + "/trace.pfw";
    std::ofstream ofs(pfw);
    for (int i = 0; i < EVENTS; ++i)
        ofs << R"({"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,"ts":)"
            << (1000 + i * 100) << R"(,"dur":)" << (10 + i) << R"(,"args":{}})"
            << "\n";
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
    rt.submit(std::move(task), "factory-host").wait();
    rt.shutdown();
    return out;
}

}  // namespace

TEST_CASE("an op registered by the factory is live before the scan") {
    auto set = Plugins::builder().add(FACTORY_REGISTERS_OP_PLUGIN_PATH).build();
    INFO((set.has_value() ? std::string{} : set.error().message));
    REQUIRE(set.has_value());

    // Registration happened at load, so the op resolves before any batch has
    // been produced. That is the whole point of handing the factory a host.
    const dftu_op_desc* op = dftu_op_find("factory_registers_op.double_rows");
    REQUIRE(op != nullptr);

    dftu_utils_test::TestEnvironment env(0);
    REQUIRE(env.is_valid());
    View view = View::from_files({make_trace(env, "regop")});
    PluginRun run = run_set(*set, view);

    auto it = run.results.results().find("doubled_rows");
    REQUIRE(it != run.results.results().end());
    const auto& bytes = std::get<std::vector<std::byte>>(it->second);
    REQUIRE(bytes.size() == sizeof(std::int64_t));
    std::int64_t total = 0;
    std::memcpy(&total, bytes.data(), sizeof(total));
    CHECK(total == 2 * EVENTS);
}

TEST_CASE("a factory reaching past the registration surface fails the load") {
    auto set = Plugins::builder().add(FACTORY_ESCAPES_PLUGIN_PATH).build();

    REQUIRE_FALSE(set.has_value());
    CHECK(set.error().message.find(DFTU_EXT_IO) != std::string::npos);
    CHECK(set.error().message.find("build-phase host") != std::string::npos);
    CHECK(set.error().message.find(FACTORY_ESCAPES_PLUGIN_PATH) !=
          std::string::npos);
}
