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
        CHECK(d[1] == 22);
        CHECK(d[2] == 33);
        dftu_series_free(out);
        dftu_series_free(ca);
        dftu_series_free(cb);
    }

    TEST_CASE("a scalar-operand op reads dftu_op_arg.scalar") {
        const dftu_op_desc* op = dftu_op_find("add_scalar");
        REQUIRE(op != nullptr);
        std::int64_t v[3] = {1, 2, 3};
        dftu_series* c = i64_col(v, 3);
        const dftu_series* in[1] = {c};
        dftu_op_arg arg{};
        arg.scalar = i64_scalar(100);
        dftu_series* out = dftu_op_run(op, in, 1, &arg);
        REQUIRE(out != nullptr);
        CHECK(i64_of(out)[0] == 101);
        CHECK(i64_of(out)[2] == 103);
        dftu_series_free(out);
        dftu_series_free(c);
    }

    TEST_CASE("an enum+scalar op (compare) reads op_code and scalar") {
        const dftu_op_desc* op = dftu_op_find("compare");
        REQUIRE(op != nullptr);
        std::int64_t v[3] = {1, 5, 2};
        dftu_series* c = i64_col(v, 3);
        const dftu_series* in[1] = {c};
        dftu_op_arg arg{};
        arg.op_code = DFTU_CMP_GT;
        arg.scalar = i64_scalar(3);
        dftu_series* out = dftu_op_run(op, in, 1, &arg);  // v > 3
        REQUIRE(out != nullptr);
        CHECK(bit_of(out, 0) == false);
        CHECK(bit_of(out, 1) == true);
        CHECK(bit_of(out, 2) == false);
        dftu_series_free(out);
        dftu_series_free(c);
    }

    TEST_CASE("a string-operand op (str_contains) reads dftu_op_arg.s0") {
        const dftu_op_desc* op = dftu_op_find("str_contains");
        REQUIRE(op != nullptr);
        std::int32_t offs[3] = {0, 6, 11};  // "foobar", "hello"
        dftu_series* c = dftu_series_new_string(DFTU_TYPE_STRING, offs,
                                                "foobarhello", 2, nullptr);
        REQUIRE(c != nullptr);
        const dftu_series* in[1] = {c};
        dftu_op_arg arg{};
        arg.s0 = "oo";
        arg.s0_len = 2;
        dftu_series* out = dftu_op_run(op, in, 1, &arg);
        REQUIRE(out != nullptr);
        CHECK(bit_of(out, 0) == true);
        CHECK(bit_of(out, 1) == false);
        dftu_series_free(out);
        dftu_series_free(c);
    }

    TEST_CASE("reducers run through dftu_op_run_aggregate") {
        std::int64_t v[3] = {1, 5, 2};
        dftu_series* c = i64_col(v, 3);
        int ok = 0;

        dftu_scalar cnt =
            dftu_op_run_aggregate(dftu_op_find("count"), c, nullptr, &ok);
        CHECK(ok == 1);
        CHECK(cnt.value.i == 3);

        dftu_scalar amax =
            dftu_op_run_aggregate(dftu_op_find("arg_max"), c, nullptr, &ok);
        CHECK(amax.value.i == 1);

        std::int64_t p[3] = {2, 3, 4};
        dftu_series* cp = i64_col(p, 3);
        dftu_scalar prod =
            dftu_op_run_aggregate(dftu_op_find("product"), cp, nullptr, &ok);
        CHECK(prod.value.i == 24);

        dftu_op_arg arg{};
        arg.op_code = DFTU_REDUCE_SUM;
        dftu_scalar sum =
            dftu_op_run_aggregate(dftu_op_find("reduce"), cp, &arg, &ok);
        CHECK(sum.value.i == 9);

        dftu_series_free(c);
        dftu_series_free(cp);
    }

    TEST_CASE("the whole column-op surface is registered and listable") {
        for (const char* name :
             {"add", "sub", "mul", "div", "add_scalar", "compare", "cast",
              "prim", "logical", "str_contains", "str_replace", "str_slice",
              "str_split", "count", "reduce", "mode"})
            CHECK(dftu_op_find(name) != nullptr);
        CHECK(dftu_op_find("no_such_op") == nullptr);

        uint32_t n = dftu_op_count();
        REQUIRE(n >= 40);
        bool saw_add = false;
        for (uint32_t i = 0; i < n; ++i) {
            const dftu_op_desc* op = dftu_op_at(i);
            REQUIRE(op != nullptr);
            if (std::strcmp(op->name, "add") == 0) saw_add = true;
        }
        CHECK(saw_add);
        CHECK(dftu_op_at(n) == nullptr);
    }

    TEST_CASE("a user op registers under a module prefix and runs") {
        uint32_t before = dftu_op_count();
        dftu_op_desc copy{"mymod.copy", DFTU_OP_SIG(SERIES, SERIES, NONE, NONE),
                          reinterpret_cast<const void*>(&dftu_series_share)};
        REQUIRE(dftu_op_register(&copy) == 0);
        CHECK(dftu_op_count() == before + 1);

        const dftu_op_desc* found = dftu_op_find("mymod.copy");
        REQUIRE(found != nullptr);
        std::int64_t v[2] = {7, 9};
        dftu_series* src = i64_col(v, 2);
        const dftu_series* in[1] = {src};
        dftu_series* out = dftu_op_run(found, in, 1, nullptr);
        REQUIRE(out != nullptr);
        CHECK(i64_of(out)[0] == 7);
        CHECK(i64_of(out)[1] == 9);
        dftu_series_free(out);
        dftu_series_free(src);
    }

    TEST_CASE("registration rejects a name clash and a NULL record") {
        CHECK(dftu_op_register(nullptr) != 0);
        dftu_op_desc dup{"add", DFTU_OP_SIG(SERIES, SERIES, SERIES, NONE),
                         reinterpret_cast<const void*>(&dftu_series_add)};
        CHECK(dftu_op_register(&dup) != 0);
    }

    TEST_CASE("run rejects a kind/arity mismatch") {
        const dftu_op_desc* add = dftu_op_find("add");
        std::int64_t v[1] = {1};
        dftu_series* c = i64_col(v, 1);
        const dftu_series* in[1] = {c};
        CHECK(dftu_op_run(add, in, 1, nullptr) == nullptr);  // add needs 2
        CHECK(dftu_op_run(nullptr, in, 1, nullptr) == nullptr);
        int ok = 1;
        (void)dftu_op_run_aggregate(add, c, nullptr, &ok);
        CHECK(ok == 0);
        dftu_series_free(c);
    }
}
