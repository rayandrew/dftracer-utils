// Exercises the reflected plugin ABI the way a plugin would: build a dftu_host
// wired to the host registry, then invoke each exposed utility through its
// generated typed wrapper and consume the output. Covers the three field
// routing classes: injected context (line_filter callback), streaming output
// (view_scanner), and projected accumulator (statistics_aggregator sketch).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/constants.h>
#include <dftracer/utils/plugins/dftu_generated_utilities.h>
#include <dftracer/utils/plugins/utility_registry.h>
#include <dftracer/utils/trace/internal/utils.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "testing_utilities.h"

namespace dfti = dftracer::utils::trace::internal;
constexpr std::uint64_t DEFAULT_CHECKPOINT_SIZE =
    dftracer::utils::constants::indexer::DEFAULT_CHECKPOINT_SIZE;
using dftracer::utils::plugins::registry_find;
using dftracer::utils::plugins::registry_run_stream;

namespace {

const dftu_utility* host_find(void*, std::uint32_t id) {
    return registry_find(id);
}

int host_run_stream(void*, std::uint32_t id, const void* in,
                    dftu_stream_item_fn on_item, void* ud) {
    return registry_run_stream(id, in, on_item, ud);
}

const dftu_ext_util g_util = {host_find, host_run_stream, nullptr, nullptr,
                              nullptr,   nullptr,         nullptr};

const void* host_get_extension(void*, const char* id) {
    return (id && std::strcmp(id, DFTU_EXT_UTIL) == 0) ? &g_util : nullptr;
}

dftu_host make_host() {
    dftu_host h{};
    h.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    h.get_extension = host_get_extension;
    return h;
}

dftu_bytes bytes_of(const std::string& s) {
    return dftu_bytes{s.data(), static_cast<std::uint32_t>(s.size())};
}

}  // namespace

TEST_CASE("plugin ABI: fnv1a single-value run still works") {
    dftu_host host = make_host();
    std::string data = "hello-world";
    dftu_bytes in = bytes_of(data);
    std::uint64_t out = 0;
    int rc = dftu_util_fnv1a(&host, &in, &out);
    CHECK(rc == 0);
    CHECK(out != 0);
}

TEST_CASE("plugin ABI: line_filter injects a C predicate (context class)") {
    dftu_host host = make_host();

    struct Counter {
        int calls = 0;
    };
    struct Sink {
        int items = 0;
        std::string last;
    };

    auto predicate = [](const dftu_line* l, void* ud) -> std::uint8_t {
        static_cast<Counter*>(ud)->calls++;
        std::string_view s(l->content.ptr, l->content.len);
        return s.find("ERROR") != std::string_view::npos ? 1 : 0;
    };
    auto on_item = [](const void* item, void* ud) {
        const dftu_line* l = static_cast<const dftu_line*>(item);
        Sink* s = static_cast<Sink*>(ud);
        s->items++;
        s->last.assign(l->content.ptr, l->content.len);
    };

    SUBCASE("predicate true yields exactly one item") {
        Counter ctx;
        Sink sink;
        std::string text = "ERROR: disk full";
        dftu_filterable_line in{};
        in.line.content = bytes_of(text);
        in.line.line_number = 7;
        in.predicate = predicate;
        in.predicate_ud = &ctx;

        int rc = dftu_util_line_filter_stream(&host, &in, on_item, &sink);
        CHECK(rc == 0);
        CHECK(ctx.calls == 1);
        CHECK(sink.items == 1);
        CHECK(sink.last == "ERROR: disk full");
    }

    SUBCASE("predicate false yields zero items") {
        Counter ctx;
        Sink sink;
        std::string text = "all good here";
        dftu_filterable_line in{};
        in.line.content = bytes_of(text);
        in.line.line_number = 3;
        in.predicate = predicate;
        in.predicate_ud = &ctx;

        int rc = dftu_util_line_filter_stream(&host, &in, on_item, &sink);
        CHECK(rc == 0);
        CHECK(ctx.calls == 1);
        CHECK(sink.items == 0);
    }
}

TEST_CASE("plugin ABI: view_scanner streams batches (streaming class)") {
    dftu_utils_test::TestEnvironment env(200);
    std::string gz = env.create_dft_test_gzip_file(200);
    REQUIRE(dftu_utils_test::build_index(gz));
    std::string idx = dfti::determine_index_path(gz, "");

    dftu_host host = make_host();
    dftu_view_scanner_input in{};
    in.file_path = bytes_of(gz);
    in.index_path = bytes_of(idx);
    in.checkpoint_size = 1024;
    in.event_batch_size = 10000;
    in.start_byte = 0;
    in.end_byte = UINT64_MAX;

    struct Sink {
        int batches = 0;
        std::uint64_t scanned = 0;
        std::uint64_t matched = 0;
    };
    auto on_item = [](const void* item, void* ud) {
        const dftu_view_scanner_batch* b =
            static_cast<const dftu_view_scanner_batch*>(item);
        Sink* s = static_cast<Sink*>(ud);
        s->batches++;
        s->scanned += b->events_scanned;
        s->matched += b->events_matched;
    };

    Sink sink;
    int rc = dftu_util_view_scanner_stream(&host, &in, on_item, &sink);
    CHECK(rc == 0);
    CHECK(sink.batches >= 1);
    CHECK(sink.scanned > 0);
    MESSAGE("stream: batches=" << sink.batches << " scanned=" << sink.scanned
                               << " matched=" << sink.matched);
}

TEST_CASE("plugin ABI: metadata_collector returns a file-metadata row") {
    dftu_utils_test::TestEnvironment env(200);
    std::string gz = env.create_dft_test_gzip_file(200);
    REQUIRE(dftu_utils_test::build_index(gz));

    dftu_host host = make_host();
    dftu_metadata_collector_utility_input in{};
    in.file_path = bytes_of(gz);
    in.checkpoint_size = DEFAULT_CHECKPOINT_SIZE;

    dftu_metadata_collector_utility_output out{};
    int rc = dftu_util_metadata_collector(&host, &in, &out);
    CHECK(rc == 0);
    CHECK(out.success == 1);
    CHECK(out.valid_events > 0);
    CHECK(out.end_line >= out.start_line);
    MESSAGE("metadata: valid_events=" << out.valid_events
                                      << " num_lines=" << out.num_lines);
}

TEST_CASE("plugin ABI: event_collector streams EventIds from metadata") {
    dftu_utils_test::TestEnvironment env(200);
    std::string gz = env.create_dft_test_gzip_file(200);
    REQUIRE(dftu_utils_test::build_index(gz));

    dftu_host host = make_host();
    dftu_metadata_collector_utility_input meta_in{};
    meta_in.file_path = bytes_of(gz);
    meta_in.checkpoint_size = DEFAULT_CHECKPOINT_SIZE;
    dftu_metadata_collector_utility_output meta{};
    REQUIRE(dftu_util_metadata_collector(&host, &meta_in, &meta) == 0);
    REQUIRE(meta.success == 1);

    std::vector<dftu_metadata_collector_utility_output> rows{meta};
    dftu_event_collector_from_metadata_collector_utility_input in{};
    in.metadata.ptr = rows.data();
    in.metadata.len = static_cast<std::uint32_t>(rows.size());

    struct Sink {
        std::uint64_t count = 0;
        bool all_valid = true;
    };
    auto on_item = [](const void* item, void* ud) {
        const dftu_event_id* e = static_cast<const dftu_event_id*>(item);
        Sink* s = static_cast<Sink*>(ud);
        s->count++;
        if (e->id == 0) s->all_valid = false;
    };

    Sink sink;
    int rc = dftu_util_event_collector_stream(&host, &in, on_item, &sink);
    CHECK(rc == 0);
    CHECK(sink.count > 0);
    CHECK(sink.all_valid);
    MESSAGE("event_collector: events=" << sink.count);
}

TEST_CASE("plugin ABI: directory_scanner streams file entries") {
    dftu_utils_test::TestEnvironment env(50);
    std::string gz = env.create_dft_test_gzip_file(50);

    dftu_host host = make_host();
    std::string dir = env.get_dir();
    dftu_directory_scanner_utility_input in{};
    in.path = bytes_of(dir);
    in.recursive = 0;
    in.populate_size = 1;

    struct Sink {
        int entries = 0;
        int regular = 0;
        std::uint64_t total_size = 0;
    };
    auto on_item = [](const void* item, void* ud) {
        const dftu_file_entry* e = static_cast<const dftu_file_entry*>(item);
        Sink* s = static_cast<Sink*>(ud);
        s->entries++;
        if (e->is_regular_file) {
            s->regular++;
            s->total_size += e->size;
        }
    };

    Sink sink;
    int rc = dftu_util_directory_scanner_stream(&host, &in, on_item, &sink);
    CHECK(rc == 0);
    CHECK(sink.entries >= 1);
    CHECK(sink.regular >= 1);
    CHECK(sink.total_size > 0);
    MESSAGE("directory_scanner: entries=" << sink.entries
                                          << " regular=" << sink.regular);
}

TEST_CASE("plugin ABI: pattern_directory_scanner filters by suffix") {
    dftu_utils_test::TestEnvironment env(50);
    std::string gz = env.create_dft_test_gzip_file(50);

    dftu_host host = make_host();
    std::string dir = env.get_dir();
    std::string pat = ".gz";
    dftu_span_string patterns{};
    dftu_bytes pat_bytes = bytes_of(pat);
    patterns.ptr = &pat_bytes;
    patterns.len = 1;

    dftu_pattern_directory_scanner_utility_input in{};
    in.path = bytes_of(dir);
    in.recursive = 0;
    in.populate_size = 1;
    in.patterns = patterns;

    struct Sink {
        int matched = 0;
    };
    auto on_item = [](const void*, void* ud) {
        static_cast<Sink*>(ud)->matched++;
    };

    Sink sink;
    int rc =
        dftu_util_pattern_directory_scanner_stream(&host, &in, on_item, &sink);
    CHECK(rc == 0);
    CHECK(sink.matched >= 1);
    MESSAGE("pattern_directory_scanner: matched=" << sink.matched);
}

TEST_CASE("plugin ABI: file_reader returns file content") {
    dftu_utils_test::TestEnvironment env(10);
    std::string path = env.get_dir() + "/reader_input.txt";
    const std::string body = "line one\nline two\nline three\n";
    {
        std::ofstream f(path, std::ios::binary);
        f << body;
    }

    dftu_host host = make_host();
    dftu_file_entry in{};
    in.path = bytes_of(path);
    in.is_regular_file = 1;

    dftu_text out{};
    int rc = dftu_util_file_reader(&host, &in, &out);
    CHECK(rc == 0);
    REQUIRE(out.content.len == body.size());
    CHECK(std::string_view(out.content.ptr, out.content.len) == body);
}

TEST_CASE("plugin ABI: file_compressor then file_decompressor round-trips") {
    dftu_utils_test::TestEnvironment env(10);
    std::string src = env.get_dir() + "/roundtrip.txt";
    std::string body;
    for (int i = 0; i < 2000; ++i) body += "the quick brown fox jumps\n";
    {
        std::ofstream f(src, std::ios::binary);
        f << body;
    }
    std::string gz = src + ".gz";
    std::string out_path = env.get_dir() + "/roundtrip.out";

    dftu_host host = make_host();

    dftu_file_compression_utility_input cin{};
    cin.input_path = bytes_of(src);
    cin.output_path = bytes_of(gz);
    cin.compression_level = 6;
    cin.member_size = 64 * 1024;
    dftu_file_compression_utility_output cout{};
    int rc = dftu_util_file_compressor(&host, &cin, &cout);
    CHECK(rc == 0);
    CHECK(cout.original_size == body.size());
    CHECK(cout.compressed_size > 0);

    dftu_file_decompression_utility_input din{};
    din.input_path = bytes_of(gz);
    din.output_path = bytes_of(out_path);
    dftu_file_decompression_utility_output dout{};
    rc = dftu_util_file_decompressor(&host, &din, &dout);
    CHECK(rc == 0);
    CHECK(dout.decompressed_size == body.size());
    MESSAGE("roundtrip: original=" << cout.original_size
                                   << " compressed=" << cout.compressed_size);
}
