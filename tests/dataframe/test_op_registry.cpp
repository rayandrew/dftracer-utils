#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/dataframe/abi.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace {

dftu_series* i64_col(const std::int64_t* v, std::int64_t n) {
    return dftu_series_new_flat(DFTU_TYPE_INT64, v, n, nullptr);
}

const std::int64_t* i64_of(const dftu_series* c) {
    return static_cast<const std::int64_t*>(dftu_series_data(c));
}

bool bit_of(const dftu_series* c, std::int64_t i) {
    const auto* d = static_cast<const std::uint8_t*>(dftu_series_data(c));
    return ((d[i >> 3] >> (i & 7)) & 1u) != 0;
}

dftu_scalar i64_scalar(std::int64_t v) {
    dftu_scalar s;
    s.kind = DFTU_SCALAR_TAG_I64;
    s.value.i = v;
    return s;
}

}  // namespace

TEST_SUITE("op_registry") {
    TEST_CASE("a built-in binary op is found and its signature decodes") {
        const dftu_op_desc* add = dftu_op_find("add");
        REQUIRE(add != nullptr);
        CHECK(dftu_op_kind_of(add->sig) == DFTU_OP_KIND_SERIES);
        CHECK(dftu_op_arity(add->sig) == 2);
        CHECK(std::string(dftu_op_signature(add->sig)) ==
              "(series, series) -> series");

        std::int64_t a[3] = {1, 2, 3}, b[3] = {10, 20, 30};
        dftu_series* ca = i64_col(a, 3);
        dftu_series* cb = i64_col(b, 3);
        const dftu_series* in[2] = {ca, cb};
        dftu_series* out = dftu_op_run(add, in, 2, nullptr);
        REQUIRE(out != nullptr);
        const std::int64_t* d = i64_of(out);
        CHECK(d[0] == 11);
        CHECK(d[2] == 33);
        dftu_series_free(out);
        dftu_series_free(ca);
        dftu_series_free(cb);
    }

    TEST_CASE(
        "scalar / enum / string / f64 operands ride args[] positionally") {
        std::int64_t v[3] = {1, 5, 2};
        dftu_series* c = i64_col(v, 3);
        const dftu_series* in[1] = {c};

        // add_scalar: SCALAR at operand slot 1.
        dftu_op_arg a1{};
        a1.args[1].scalar = i64_scalar(100);
        dftu_series* o1 = dftu_op_run(dftu_op_find("add_scalar"), in, 1, &a1);
        REQUIRE(o1 != nullptr);
        CHECK(i64_of(o1)[0] == 101);
        dftu_series_free(o1);

        // compare: enum at slot 1, scalar at slot 2 -> v > 3.
        dftu_op_arg a2{};
        a2.args[1].i32 = DFTU_CMP_GT;
        a2.args[2].scalar = i64_scalar(3);
        dftu_series* o2 = dftu_op_run(dftu_op_find("compare"), in, 1, &a2);
        REQUIRE(o2 != nullptr);
        CHECK(bit_of(o2, 1) == true);
        CHECK(bit_of(o2, 0) == false);
        dftu_series_free(o2);
        dftu_series_free(c);

        // str_contains: str at slot 1.
        std::int32_t offs[3] = {0, 6, 11};
        dftu_series* sc = dftu_series_new_string(DFTU_TYPE_STRING, offs,
                                                 "foobarhello", 2, nullptr);
        const dftu_series* sin[1] = {sc};
        dftu_op_arg a3{};
        a3.args[1].str.ptr = "oo";
        a3.args[1].str.len = 2;
        dftu_series* o3 =
            dftu_op_run(dftu_op_find("str_contains"), sin, 1, &a3);
        REQUIRE(o3 != nullptr);
        CHECK(bit_of(o3, 0) == true);
        CHECK(bit_of(o3, 1) == false);
        dftu_series_free(o3);
        dftu_series_free(sc);
    }

    TEST_CASE("reducers run through dftu_op_run_aggregate") {
        std::int64_t v[3] = {1, 5, 2};
        dftu_series* c = i64_col(v, 3);
        int ok = 0;
        CHECK(dftu_op_run_aggregate(dftu_op_find("count"), c, nullptr, &ok)
                  .value.i == 3);
        CHECK(ok == 1);
        CHECK(dftu_op_run_aggregate(dftu_op_find("arg_max"), c, nullptr, &ok)
                  .value.i == 1);

        std::int64_t p[3] = {2, 3, 4};
        dftu_series* cp = i64_col(p, 3);
        CHECK(dftu_op_run_aggregate(dftu_op_find("product"), cp, nullptr, &ok)
                  .value.i == 24);
        dftu_op_arg arg{};
        arg.args[1].i32 = DFTU_REDUCE_SUM;
        CHECK(dftu_op_run_aggregate(dftu_op_find("reduce"), cp, &arg, &ok)
                  .value.i == 9);
        dftu_series_free(c);
        dftu_series_free(cp);
    }

    TEST_CASE("the whole column-op surface is registered and listable") {
        for (const char* name :
             {"add", "compare", "cast", "prim", "logical", "str_contains",
              "str_replace", "str_slice", "count", "reduce", "mode"})
            CHECK(dftu_op_find(name) != nullptr);
        CHECK(dftu_op_find("no_such_op") == nullptr);

        uint32_t n = dftu_op_count();
        REQUIRE(n >= 120);  // ~100 column ops + ~23 frame ops
        bool saw_add = false, saw_frame = false;
        for (uint32_t i = 0; i < n; ++i) {
            const dftu_op_desc* op = dftu_op_at(i);
            REQUIRE(op != nullptr);
            if (std::strcmp(op->name, "add") == 0) saw_add = true;
            if (std::strcmp(op->name, "frame.head") == 0) saw_frame = true;
        }
        CHECK(saw_add);
        CHECK(saw_frame);
        CHECK(dftu_op_at(n) == nullptr);
    }

    TEST_CASE("newly-added column ops and sig shapes run") {
        for (const char* name :
             {"abs", "sqrt", "cumsum", "fillna", "shift", "is_in", "nunique",
              "variance", "stddev", "quantile", "skewness", "clip",
              "is_between", "sort", "rank", "rolling", "ewm_mean"})
            CHECK(dftu_op_find(name) != nullptr);

        std::int64_t v[4] = {1, 2, 3, 4};
        dftu_series* c = i64_col(v, 4);
        const dftu_series* in1[1] = {c};
        int ok = 0;

        dftu_op_arg varg{};
        varg.args[1].i32 = 1;  // sample
        dftu_scalar var =
            dftu_op_run_aggregate(dftu_op_find("variance"), c, &varg, &ok);
        CHECK(ok == 1);
        CHECK(var.kind == DFTU_SCALAR_TAG_F64);
        CHECK(var.value.d == doctest::Approx(5.0 / 3.0));

        dftu_op_arg qarg{};
        qarg.args[1].f64 = 0.5;
        dftu_scalar q =
            dftu_op_run_aggregate(dftu_op_find("quantile"), c, &qarg, &ok);
        CHECK(q.kind == DFTU_SCALAR_TAG_F64);

        dftu_op_arg carg{};
        carg.args[1].scalar = i64_scalar(2);
        carg.args[2].scalar = i64_scalar(3);
        dftu_series* clipped = dftu_op_run(dftu_op_find("clip"), in1, 1, &carg);
        REQUIRE(clipped != nullptr);
        CHECK(i64_of(clipped)[0] == 2);
        CHECK(i64_of(clipped)[3] == 3);
        dftu_series_free(clipped);

        dftu_series* cs = dftu_op_run(dftu_op_find("cumsum"), in1, 1, nullptr);
        REQUIRE(cs != nullptr);
        CHECK(i64_of(cs)[3] == 10);
        dftu_series_free(cs);
        dftu_series_free(c);
    }

    TEST_CASE("frame ops run via dftu_op_run_frame") {
        std::int64_t a[4] = {3, 1, 2, 1}, b[4] = {10, 20, 30, 40};
        const char* names[2] = {"a", "b"};
        dftu_series* cols[2] = {i64_col(a, 4), i64_col(b, 4)};  // moved into df
        dftu_dataframe* df = dftu_dataframe_new(names, cols, 2);
        REQUIRE(df != nullptr);
        const dftu_dataframe* fin[1] = {df};

        const dftu_op_desc* head = dftu_op_find("frame.head");
        REQUIRE(head != nullptr);
        CHECK(dftu_op_kind_of(head->sig) == DFTU_OP_KIND_FRAME);
        dftu_op_arg harg{};
        harg.args[1].i64 = 2;
        dftu_dataframe* h = dftu_op_run_frame(head, fin, 1, &harg);
        REQUIRE(h != nullptr);
        CHECK(dftu_dataframe_num_rows(h) == 2);
        dftu_dataframe_free(h);

        const char* sel[1] = {"a"};
        dftu_op_arg sarg{};
        sarg.args[1].list.items = sel;
        sarg.args[1].list.n = 1;
        dftu_dataframe* s =
            dftu_op_run_frame(dftu_op_find("frame.select"), fin, 1, &sarg);
        REQUIRE(s != nullptr);
        CHECK(dftu_dataframe_num_columns(s) == 1);
        dftu_dataframe_free(s);

        dftu_dataframe_free(df);

        // value_counts: series -> frame (no frame operand; series in args[0]).
        std::int64_t vc[4] = {1, 1, 2, 3};
        dftu_series* cvc = i64_col(vc, 4);
        dftu_op_arg vcarg{};
        vcarg.args[0].series = cvc;
        dftu_dataframe* vcf = dftu_op_run_frame(
            dftu_op_find("frame.value_counts"), nullptr, 0, &vcarg);
        REQUIRE(vcf != nullptr);
        CHECK(dftu_dataframe_num_rows(vcf) == 3);  // 3 distinct values
        dftu_dataframe_free(vcf);
        dftu_series_free(cvc);
    }

    TEST_CASE("a user op registers under a module prefix and runs") {
        dftu_op_desc copy{"mymod.copy", DFTU_OP_SIG(SERIES, SERIES, NONE, NONE),
                          reinterpret_cast<const void*>(&dftu_series_share)};
        REQUIRE(dftu_op_register(&copy) == 0);
        std::int64_t v[2] = {7, 9};
        dftu_series* src = i64_col(v, 2);
        const dftu_series* in[1] = {src};
        dftu_series* out =
            dftu_op_run(dftu_op_find("mymod.copy"), in, 1, nullptr);
        REQUIRE(out != nullptr);
        CHECK(i64_of(out)[1] == 9);
        dftu_series_free(out);
        dftu_series_free(src);
    }

    TEST_CASE("registration rejects a clash and run rejects a mismatch") {
        CHECK(dftu_op_register(nullptr) != 0);
        dftu_op_desc dup{"add", DFTU_OP_SIG(SERIES, SERIES, SERIES, NONE),
                         reinterpret_cast<const void*>(&dftu_series_add)};
        CHECK(dftu_op_register(&dup) != 0);
        std::int64_t v[1] = {1};
        dftu_series* c = i64_col(v, 1);
        const dftu_series* in[1] = {c};
        CHECK(dftu_op_run(dftu_op_find("add"), in, 1, nullptr) == nullptr);
        dftu_series_free(c);
    }
}
