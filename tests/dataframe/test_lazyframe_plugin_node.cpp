// dftu_node_register/dftu_lazyframe_op: a plugin plan node is a Cursor the
// engine drives inside the same pull chain lower_cursor_chain builds for
// every built-in op, not a second protocol. These tests drive hand-written
// dftu_node_vt fixtures through the public C ABI directly (no dlopen), the
// same style test_provider_schema_types.cpp uses for dftu_source_vt.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/core/coro/task_abi.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/provider_source.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using dftracer::utils::coro::CoroTask;
using dftracer::utils::dataframe::col;
using dftracer::utils::dataframe::DataFrame;
using dftracer::utils::dataframe::LazyFrame;
using dftracer::utils::dataframe::make_provider_source;
using dftracer::utils::dataframe::OpArgs;
using dftracer::utils::dataframe::Schema;
using dftracer::utils::dataframe::Source;
using dftracer::utils::dataframe::TimeUnit;
using dftracer::utils::dataframe::TypeId;

namespace {

DataFrame run(CoroTask<DataFrame> t) {
    return dftracer::utils::default_runtime().submit(std::move(t)).get();
}

// ---- A resident int64 source, one morsel per collect. ---------------------

struct IntSource {
    std::vector<std::string> names;
    std::vector<std::int64_t> col0;
};

int32_t src_schema(void* self, const char* const** out_names) {
    auto* s = static_cast<IntSource*>(self);
    static thread_local std::vector<const char*> storage;
    storage.clear();
    for (const std::string& n : s->names) storage.push_back(n.c_str());
    *out_names = storage.data();
    return static_cast<int32_t>(storage.size());
}

struct SrcCursorState {
    dftu_dataframe* frame;
    bool done = false;
};

dftu_task* src_cursor_next(void* self, std::int64_t, dftu_result_frame* out) {
    auto* cs = static_cast<SrcCursorState*>(self);
    out->ok = 1;
    if (cs->done) {
        out->u.value = nullptr;
    } else {
        out->u.value = cs->frame;
        cs->frame = nullptr;
        cs->done = true;
    }
    return nullptr;
}

void src_cursor_destroy(void* self) {
    auto* cs = static_cast<SrcCursorState*>(self);
    if (cs->frame) dftu_dataframe_free(cs->frame);
    delete cs;
}

const dftu_cursor_vt SRC_CURSOR_VT = {src_cursor_next, src_cursor_destroy};

void* src_scan(void* self, const dftu_scan_request*, int32_t*,
               void** out_cursor_self, const dftu_cursor_vt** out_vt) {
    auto* s = static_cast<IntSource*>(self);
    dftu_series* c = dftu_series_new_flat(
        DFTU_TYPE_INT64, s->col0.data(),
        static_cast<std::int64_t>(s->col0.size()), nullptr);
    const char* names[1] = {s->names[0].c_str()};
    dftu_series* cols[1] = {c};
    auto* cs = new SrcCursorState{dftu_dataframe_new(names, cols, 1)};
    *out_cursor_self = cs;
    *out_vt = &SRC_CURSOR_VT;
    return cs;
}

void src_destroy(void*) {}

// The Timestamp column carries a timezone (a parameterized type) so a node
// declaring it via dftu_schema_copy_field must preserve it, not just the id.
void src_schema_types(void* self, dftu_schema* out) {
    auto* s = static_cast<IntSource*>(self);
    (void)s;
    dftu_schema_add_field(out, "val", DFTU_TYPE_INT64, 0, DFTU_TIME_UNIT_MICRO,
                          nullptr, 0, 0, 0);
    dftu_schema_add_field(out, "ts", DFTU_TYPE_TIMESTAMP, 1,
                          DFTU_TIME_UNIT_MILLI, "UTC", 0, 0, 0);
}

std::shared_ptr<Source> make_int_source(IntSource& fx, bool typed) {
    dftu_source_vt vt{};
    vt.schema = src_schema;
    vt.scan = src_scan;
    vt.destroy = src_destroy;
    if (typed) vt.schema_types = src_schema_types;
    return make_provider_source(vt, &fx);
}

// ---- "test.double": a streaming node doubling column 0. --------------------

CoroTask<void> pull_and_double(void* in_self, const dftu_cursor_vt* in_vt,
                               std::int64_t max_rows, dftu_result_frame* out) {
    dftu_result_frame in_out{};
    if (dftu_task* t = in_vt->next(in_self, max_rows, &in_out)) {
        auto* task = reinterpret_cast<CoroTask<void>*>(t);
        co_await *task;
        delete task;
    }
    if (!DFTU_RESULT_OK(in_out)) {
        *out = in_out;
        co_return;
    }
    dftu_dataframe* frame = DFTU_RESULT_VALUE(in_out);
    if (!frame) {
        out->ok = 1;
        out->u.value = nullptr;
        co_return;
    }
    dftu_series* c0 = dftu_dataframe_column(frame, "val");
    const std::int64_t n = dftu_series_length(c0);
    const auto* data = static_cast<const std::int64_t*>(dftu_series_data(c0));
    std::vector<std::int64_t> doubled(data, data + n);
    for (std::int64_t& v : doubled) v *= 2;
    dftu_series_free(c0);
    dftu_series* new_col =
        dftu_series_new_flat(DFTU_TYPE_INT64, doubled.data(), n, nullptr);
    const char* names[1] = {"val"};
    dftu_series* cols[1] = {new_col};
    dftu_dataframe* result = dftu_dataframe_new(names, cols, 1);
    dftu_dataframe_free(frame);
    out->ok = 1;
    out->u.value = result;
}

struct DoubleCursorState {
    void* in_self;
    const dftu_cursor_vt* in_vt;
};

dftu_task* double_cursor_next(void* self, std::int64_t max_rows,
                              dftu_result_frame* out) {
    auto* st = static_cast<DoubleCursorState*>(self);
    return dftracer::utils::task_to_abi(
        pull_and_double(st->in_self, st->in_vt, max_rows, out));
}

void double_cursor_destroy(void* self) {
    auto* st = static_cast<DoubleCursorState*>(self);
    if (st->in_vt->destroy) st->in_vt->destroy(st->in_self);
    delete st;
}

const dftu_cursor_vt DOUBLE_CURSOR_VT = {double_cursor_next,
                                         double_cursor_destroy};

void* double_node_open(void*, void* in_self, const dftu_cursor_vt* in_vt,
                       const dftu_op_arg*, void** out_self,
                       const dftu_cursor_vt** out_vt) {
    auto* st = new DoubleCursorState{in_self, in_vt};
    *out_self = st;
    *out_vt = &DOUBLE_CURSOR_VT;
    return st;
}

// Pass every input field through unchanged - doubling a value never changes
// its declared type, including a parameterized one (Timestamp + timezone).
void double_node_schema(void*, const dftu_schema* in, const dftu_op_arg*,
                        dftu_schema* out) {
    const int32_t n = dftu_schema_field_count(in);
    for (int32_t i = 0; i < n; ++i) dftu_schema_copy_field(out, in, i);
}

void double_node_destroy(void*) {}

dftu_node_vt make_double_vt() {
    dftu_node_vt vt{};
    vt.output_schema = double_node_schema;
    vt.open = double_node_open;
    vt.destroy = double_node_destroy;
    return vt;
}

// ---- "test.open_fails": open() always returns the NULL sentinel. ----------

void*  // NOLINT(readability-non-const-parameter)
failing_node_open(void*, void*, const dftu_cursor_vt*, const dftu_op_arg*,
                  void**, const dftu_cursor_vt**) {
    return nullptr;
}

void failing_node_schema(void*, const dftu_schema*, const dftu_op_arg*,
                         dftu_schema*) {}

// ---- A node recording only whether it was ever opened. ---------------------

struct RecordingNode {
    bool opened = false;
};

void recording_node_schema(void*, const dftu_schema* in, const dftu_op_arg*,
                           dftu_schema* out) {
    const int32_t n = dftu_schema_field_count(in);
    for (int32_t i = 0; i < n; ++i) dftu_schema_copy_field(out, in, i);
}

// Reuses the doubler's cursor machinery (values end up doubled); the test
// below only checks that open() ran and where the node sits in explain(),
// not the values.
void* recording_node_open(void* self, void* in_self,
                          const dftu_cursor_vt* in_vt, const dftu_op_arg*,
                          void** out_self, const dftu_cursor_vt** out_vt) {
    static_cast<RecordingNode*>(self)->opened = true;
    auto* st = new DoubleCursorState{in_self, in_vt};
    *out_self = st;
    *out_vt = &DOUBLE_CURSOR_VT;
    return st;
}

}  // namespace

TEST_SUITE("lazyframe plugin node") {
    TEST_CASE("a streaming node doubles values through the pull chain") {
        REQUIRE(dftu_node_register("test.double", nullptr, nullptr) != 0);
        dftu_node_vt vt = make_double_vt();
        REQUIRE(dftu_node_register("test.double.a", &vt, nullptr) == 0);

        IntSource fx{{"val"}, {1, 2, 3, 4}};
        LazyFrame lf = LazyFrame::scan(make_int_source(fx, false))
                           .op("test.double.a", OpArgs());
        DataFrame out = run(lf.collect());
        REQUIRE(out.columns.size() == 1);
        REQUIRE(out.num_rows() == 4);
        CHECK(out.columns[0].data<std::int64_t>()[0] == 2);
        CHECK(out.columns[0].data<std::int64_t>()[1] == 4);
        CHECK(out.columns[0].data<std::int64_t>()[2] == 6);
        CHECK(out.columns[0].data<std::int64_t>()[3] == 8);

        CHECK(dftu_node_unregister("test.double.a") == 0);
        CHECK(dftu_node_unregister("test.double.a") != 0);
    }

    TEST_CASE(
        "re-registering the same name is refused; unregister then "
        "re-register works") {
        dftu_node_vt vt = make_double_vt();
        REQUIRE(dftu_node_register("test.double.b", &vt, nullptr) == 0);
        CHECK(dftu_node_register("test.double.b", &vt, nullptr) != 0);
        REQUIRE(dftu_node_unregister("test.double.b") == 0);
        CHECK(dftu_node_register("test.double.b", &vt, nullptr) == 0);
        CHECK(dftu_node_unregister("test.double.b") == 0);
    }

    TEST_CASE(
        "output_schema over a plan reports the node's declared types, "
        "including a parameterized type") {
        dftu_node_vt vt = make_double_vt();
        REQUIRE(dftu_node_register("test.double.c", &vt, nullptr) == 0);

        // Schema-only: output_schema() never scans, so the source need not
        // actually carry a "ts" column, only declare it.
        IntSource fx{{"val", "ts"}, {1, 2, 3}};
        LazyFrame lf = LazyFrame::scan(make_int_source(fx, true))
                           .op("test.double.c", OpArgs());
        Schema s = lf.output_schema();
        REQUIRE(s.fields.size() == 2);
        CHECK(s.fields[0].name == "val");
        CHECK(s.fields[0].type.id == TypeId::Int64);
        CHECK(s.fields[1].name == "ts");
        CHECK(s.fields[1].type.id == TypeId::Timestamp);
        CHECK(s.fields[1].type.time_unit == TimeUnit::Milli);
        CHECK(s.fields[1].type.timezone == "UTC");

        CHECK(dftu_node_unregister("test.double.c") == 0);
    }

    TEST_CASE("explain names the node") {
        dftu_node_vt vt = make_double_vt();
        REQUIRE(dftu_node_register("test.double.d", &vt, nullptr) == 0);

        IntSource fx{{"val"}, {1}};
        LazyFrame lf = LazyFrame::scan(make_int_source(fx, false))
                           .op("test.double.d", OpArgs());
        std::string explained = lf.explain();
        CHECK(explained.find("op test.double.d") != std::string::npos);

        CHECK(dftu_node_unregister("test.double.d") == 0);
    }

    TEST_CASE("an unknown node name is a clear error") {
        IntSource fx{{"val"}, {1}};
        LazyFrame base = LazyFrame::scan(make_int_source(fx, false));
        CHECK_THROWS_AS(base.op("no.such.node", OpArgs()),
                        std::invalid_argument);

        dftu_lazyframe* base_c =
            dftu_lazyframe_from_provider("no.such.provider");
        CHECK(base_c == nullptr);
    }

    TEST_CASE(
        "a node whose open() returns NULL is a clear error at drive "
        "time") {
        dftu_node_vt vt{};
        vt.output_schema = failing_node_schema;
        vt.open = failing_node_open;
        REQUIRE(dftu_node_register("test.fails", &vt, nullptr) == 0);

        IntSource fx{{"val"}, {1, 2}};
        LazyFrame lf = LazyFrame::scan(make_int_source(fx, false))
                           .op("test.fails", OpArgs());
        CHECK_THROWS_AS(run(lf.collect()), std::runtime_error);

        CHECK(dftu_node_unregister("test.fails") == 0);
    }

    TEST_CASE("a filter placed after a node is not pushed through it") {
        dftu_node_vt vt = make_double_vt();
        REQUIRE(dftu_node_register("test.double.e", &vt, nullptr) == 0);

        IntSource fx{{"val"}, {1, 2, 3, 4, 5}};
        LazyFrame lf = LazyFrame::scan(make_int_source(fx, false))
                           .op("test.double.e", OpArgs())
                           .filter(col(0) > std::int64_t{5});
        std::string explained = lf.explain();
        // Structural: the node line still precedes the filter line, so
        // pushdown never hoisted the filter across the opaque node.
        std::size_t op_pos = explained.find("op test.double.e");
        std::size_t filter_pos = explained.find("filter");
        REQUIRE(op_pos != std::string::npos);
        REQUIRE(filter_pos != std::string::npos);
        CHECK(op_pos < filter_pos);

        // Functional: the filter must see doubled values (it runs on the
        // node's OUTPUT), not the source's original column.
        DataFrame out = run(lf.collect());
        REQUIRE(out.num_rows() == 3);  // doubled: 2,4,6,8,10 -> >5: 6,8,10
        CHECK(out.columns[0].data<std::int64_t>()[0] == 6);
        CHECK(out.columns[0].data<std::int64_t>()[1] == 8);
        CHECK(out.columns[0].data<std::int64_t>()[2] == 10);

        CHECK(dftu_node_unregister("test.double.e") == 0);
    }

    TEST_CASE(
        "a select placed after a node is not pushed into the scan "
        "across it") {
        RecordingNode rec;
        dftu_node_vt vt{};
        vt.output_schema = recording_node_schema;
        vt.open = recording_node_open;
        REQUIRE(dftu_node_register("test.double.f", &vt, &rec) == 0);

        IntSource fx{{"val"}, {10, 20}};
        LazyFrame lf = LazyFrame::scan(make_int_source(fx, false))
                           .op("test.double.f", OpArgs())
                           .select({"val"});
        std::string explained = lf.explain();
        std::size_t scan_pos = explained.find("scan");
        std::size_t op_pos = explained.find("op test.double.f");
        std::size_t select_pos = explained.find("select");
        REQUIRE(scan_pos != std::string::npos);
        REQUIRE(op_pos != std::string::npos);
        REQUIRE(select_pos != std::string::npos);
        // If pushdown had (incorrectly) hoisted the select across the node,
        // it would appear as a projection right after "scan", before "op".
        CHECK(scan_pos < op_pos);
        CHECK(op_pos < select_pos);

        DataFrame out = run(lf.collect());
        REQUIRE(out.num_rows() == 2);
        CHECK(rec.opened);

        CHECK(dftu_node_unregister("test.double.f") == 0);
    }

    TEST_CASE("a breaker node drains its input and emits one frame") {
        struct CountState {
            void* in_self;
            const dftu_cursor_vt* in_vt;
            bool done = false;
        };

        static const dftu_cursor_vt* count_vt_holder = nullptr;
        (void)count_vt_holder;

        struct Impl {
            static CoroTask<void> drain_and_count(CountState* st,
                                                  dftu_result_frame* out) {
                std::int64_t total = 0;
                for (;;) {
                    dftu_result_frame in_out{};
                    if (dftu_task* t =
                            st->in_vt->next(st->in_self, 1 << 20, &in_out)) {
                        auto* task = reinterpret_cast<CoroTask<void>*>(t);
                        co_await *task;
                        delete task;
                    }
                    if (!DFTU_RESULT_OK(in_out)) {
                        *out = in_out;
                        co_return;
                    }
                    dftu_dataframe* frame = DFTU_RESULT_VALUE(in_out);
                    if (!frame) break;
                    total += dftu_dataframe_num_rows(frame);
                    dftu_dataframe_free(frame);
                }
                dftu_series* c =
                    dftu_series_new_flat(DFTU_TYPE_INT64, &total, 1, nullptr);
                const char* names[1] = {"count"};
                dftu_series* cols[1] = {c};
                out->ok = 1;
                out->u.value = dftu_dataframe_new(names, cols, 1);
            }

            static dftu_task* next(void* self, std::int64_t,
                                   dftu_result_frame* out) {
                auto* st = static_cast<CountState*>(self);
                if (st->done) {
                    out->ok = 1;
                    out->u.value = nullptr;
                    return nullptr;
                }
                st->done = true;
                return dftracer::utils::task_to_abi(drain_and_count(st, out));
            }
            static void destroy(void* self) {
                auto* st = static_cast<CountState*>(self);
                if (st->in_vt->destroy) st->in_vt->destroy(st->in_self);
                delete st;
            }
        };

        static const dftu_cursor_vt COUNT_VT = {Impl::next, Impl::destroy};

        struct NodeImpl {
            static void schema(void*, const dftu_schema*, const dftu_op_arg*,
                               dftu_schema* out) {
                dftu_schema_add_field(out, "count", DFTU_TYPE_INT64, 0,
                                      DFTU_TIME_UNIT_MICRO, nullptr, 0, 0, 0);
            }
            static void* open(void*, void* in_self, const dftu_cursor_vt* in_vt,
                              const dftu_op_arg*, void** out_self,
                              const dftu_cursor_vt** out_vt) {
                auto* st = new CountState{in_self, in_vt};
                *out_self = st;
                *out_vt = &COUNT_VT;
                return st;
            }
        };

        dftu_node_vt vt{};
        vt.output_schema = NodeImpl::schema;
        vt.open = NodeImpl::open;
        REQUIRE(dftu_node_register("test.count_breaker", &vt, nullptr) == 0);

        IntSource fx{{"val"}, {1, 2, 3, 4, 5, 6, 7}};
        LazyFrame lf = LazyFrame::scan(make_int_source(fx, false))
                           .op("test.count_breaker", OpArgs());
        DataFrame out = run(lf.collect());
        REQUIRE(out.num_rows() == 1);
        CHECK(out.columns[0].data<std::int64_t>()[0] == 7);

        CHECK(dftu_node_unregister("test.count_breaker") == 0);
    }
}
