// Coverage for the capability identity and semver machinery in abi.h: version
// parsing, ordering, the constraint operators, and capability/requirement
// parsing and matching.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/plugins/abi.h>
#include <doctest/doctest.h>

#include <cstring>
#include <string>

namespace {

dftu_version ver(uint16_t a, uint16_t b, uint16_t c) {
    dftu_version v;
    v.major = a;
    v.minor = b;
    v.patch = c;
    return v;
}

dftu_version parsed(const char* s) {
    dftu_version v = {9, 9, 9};
    REQUIRE(dftu_version_parse(s, (uint32_t)std::strlen(s), &v) == 0);
    return v;
}

bool sat(dftu_version have, dftu_ver_op op, dftu_version want) {
    return dftu_version_satisfies(have, op, want) != 0;
}

}  // namespace

TEST_CASE("version parse: components default to zero") {
    CHECK(dftu_version_cmp(parsed("1"), ver(1, 0, 0)) == 0);
    CHECK(dftu_version_cmp(parsed("1.2"), ver(1, 2, 0)) == 0);
    CHECK(dftu_version_cmp(parsed("1.2.3"), ver(1, 2, 3)) == 0);
    CHECK(dftu_version_cmp(parsed("0.1"), ver(0, 1, 0)) == 0);
    CHECK(dftu_version_cmp(parsed("65535.65535.65535"),
                           ver(65535, 65535, 65535)) == 0);
}

TEST_CASE("version parse: malformed input is rejected") {
    dftu_version v;
    CHECK(dftu_version_parse("", 0, &v) == -1);
    CHECK(dftu_version_parse("1.", 2, &v) == -1);
    CHECK(dftu_version_parse(".1", 2, &v) == -1);
    CHECK(dftu_version_parse("1.2.3.4", 7, &v) == -1);
    CHECK(dftu_version_parse("1.x", 3, &v) == -1);
    CHECK(dftu_version_parse("abc", 3, &v) == -1);
    CHECK(dftu_version_parse("70000", 5, &v) == -1);  // overflows uint16
}

TEST_CASE("version ordering") {
    CHECK(dftu_version_cmp(ver(1, 2, 3), ver(1, 2, 3)) == 0);
    CHECK(dftu_version_cmp(ver(1, 2, 3), ver(1, 2, 4)) < 0);
    CHECK(dftu_version_cmp(ver(1, 3, 0), ver(1, 2, 9)) > 0);
    CHECK(dftu_version_cmp(ver(2, 0, 0), ver(1, 9, 9)) > 0);
}

TEST_CASE("comparison operators") {
    CHECK(sat(ver(1, 2, 3), DFTU_VER_GE, ver(1, 2, 0)));
    CHECK_FALSE(sat(ver(1, 1, 0), DFTU_VER_GE, ver(1, 2, 0)));
    CHECK(sat(ver(1, 2, 1), DFTU_VER_GT, ver(1, 2, 0)));
    CHECK_FALSE(sat(ver(1, 2, 0), DFTU_VER_GT, ver(1, 2, 0)));
    CHECK(sat(ver(1, 2, 0), DFTU_VER_LE, ver(1, 2, 0)));
    CHECK(sat(ver(1, 1, 9), DFTU_VER_LT, ver(1, 2, 0)));
    CHECK_FALSE(sat(ver(1, 2, 0), DFTU_VER_LT, ver(1, 2, 0)));
    CHECK(sat(ver(1, 2, 3), DFTU_VER_EQ, ver(1, 2, 3)));
    CHECK_FALSE(sat(ver(1, 2, 3), DFTU_VER_EQ, ver(1, 2, 4)));
}

TEST_CASE("caret: compatible within major") {
    CHECK(sat(ver(1, 2, 0), DFTU_VER_CARET, ver(1, 2, 0)));
    CHECK(sat(ver(1, 9, 9), DFTU_VER_CARET, ver(1, 2, 0)));
    CHECK_FALSE(sat(ver(2, 0, 0), DFTU_VER_CARET, ver(1, 2, 0)));
    CHECK_FALSE(
        sat(ver(1, 1, 0), DFTU_VER_CARET, ver(1, 2, 0)));  // below floor
}

TEST_CASE("caret: 0.x treats minor as the breaking axis") {
    CHECK(sat(ver(0, 2, 3), DFTU_VER_CARET, ver(0, 2, 3)));
    CHECK(sat(ver(0, 2, 9), DFTU_VER_CARET, ver(0, 2, 3)));
    CHECK_FALSE(sat(ver(0, 3, 0), DFTU_VER_CARET, ver(0, 2, 3)));
    CHECK_FALSE(sat(ver(0, 1, 0), DFTU_VER_CARET, ver(0, 2, 3)));
}

TEST_CASE("caret: 0.0.z is exact") {
    CHECK(sat(ver(0, 0, 3), DFTU_VER_CARET, ver(0, 0, 3)));
    CHECK_FALSE(sat(ver(0, 0, 4), DFTU_VER_CARET, ver(0, 0, 3)));
}

TEST_CASE("tilde: compatible within minor") {
    CHECK(sat(ver(1, 2, 0), DFTU_VER_TILDE, ver(1, 2, 0)));
    CHECK(sat(ver(1, 2, 9), DFTU_VER_TILDE, ver(1, 2, 0)));
    CHECK_FALSE(sat(ver(1, 3, 0), DFTU_VER_TILDE, ver(1, 2, 0)));
    CHECK_FALSE(sat(ver(2, 0, 0), DFTU_VER_TILDE, ver(1, 2, 0)));
    CHECK_FALSE(sat(ver(1, 1, 9), DFTU_VER_TILDE, ver(1, 2, 0)));
}

TEST_CASE("requirement parse: operators and bare id") {
    char buf[64];
    dftu_requirement r;
    r.required = 1;

    REQUIRE(dftu_requirement_parse("com.acme.foo>=1.2", buf, sizeof buf, &r) ==
            0);
    CHECK(std::string(r.id) == "com.acme.foo");
    CHECK(r.op == DFTU_VER_GE);
    CHECK(dftu_version_cmp(r.ver, ver(1, 2, 0)) == 0);
    CHECK(r.required == 1);  // parse does not touch required

    REQUIRE(dftu_requirement_parse("foo^1.2", buf, sizeof buf, &r) == 0);
    CHECK(r.op == DFTU_VER_CARET);
    CHECK(dftu_version_cmp(r.ver, ver(1, 2, 0)) == 0);

    REQUIRE(dftu_requirement_parse("foo~1.0", buf, sizeof buf, &r) == 0);
    CHECK(r.op == DFTU_VER_TILDE);

    REQUIRE(dftu_requirement_parse("foo=1.2.3", buf, sizeof buf, &r) == 0);
    CHECK(r.op == DFTU_VER_EQ);
    CHECK(dftu_version_cmp(r.ver, ver(1, 2, 3)) == 0);

    REQUIRE(dftu_requirement_parse("foo<2", buf, sizeof buf, &r) == 0);
    CHECK(r.op == DFTU_VER_LT);

    REQUIRE(dftu_requirement_parse("foo<=1.4", buf, sizeof buf, &r) == 0);
    CHECK(r.op == DFTU_VER_LE);

    REQUIRE(dftu_requirement_parse("foo>1.0", buf, sizeof buf, &r) == 0);
    CHECK(r.op == DFTU_VER_GT);

    // Bare id: any version.
    REQUIRE(dftu_requirement_parse("com.acme.bar", buf, sizeof buf, &r) == 0);
    CHECK(std::string(r.id) == "com.acme.bar");
    CHECK(r.op == DFTU_VER_GE);
    CHECK(dftu_version_cmp(r.ver, ver(0, 0, 0)) == 0);
}

TEST_CASE("requirement parse: rejects empty id and small buffer") {
    char buf[64];
    char tiny[4];
    dftu_requirement r;
    CHECK(dftu_requirement_parse(">=1.0", buf, sizeof buf, &r) == -1);
    CHECK(dftu_requirement_parse("com.acme.foo>=1.0", tiny, sizeof tiny, &r) ==
          -1);
    CHECK(dftu_requirement_parse("foo>=bad", buf, sizeof buf, &r) == -1);
}

TEST_CASE("capability parse: id@version and bare id") {
    char buf[64];
    dftu_capability c;
    REQUIRE(dftu_capability_parse("com.acme.foo@1.2.0", buf, sizeof buf, &c) ==
            0);
    CHECK(std::string(c.id) == "com.acme.foo");
    CHECK(dftu_version_cmp(c.ver, ver(1, 2, 0)) == 0);

    REQUIRE(dftu_capability_parse("com.acme.bar", buf, sizeof buf, &c) == 0);
    CHECK(dftu_version_cmp(c.ver, ver(0, 0, 0)) == 0);
}

TEST_CASE("capability satisfies a requirement: id and version both matter") {
    char cbuf[64], rbuf[64];
    dftu_capability c;
    dftu_requirement r;
    REQUIRE(dftu_capability_parse("com.acme.events@1.4.0", cbuf, sizeof cbuf,
                                  &c) == 0);

    REQUIRE(dftu_requirement_parse("com.acme.events^1.0", rbuf, sizeof rbuf,
                                   &r) == 0);
    CHECK(dftu_capability_satisfies(&c, &r));  // 1.4.0 in ^1.0

    REQUIRE(dftu_requirement_parse("com.acme.events^2.0", rbuf, sizeof rbuf,
                                   &r) == 0);
    CHECK_FALSE(dftu_capability_satisfies(&c, &r));  // major mismatch

    REQUIRE(dftu_requirement_parse("com.acme.other^1.0", rbuf, sizeof rbuf,
                                   &r) == 0);
    CHECK_FALSE(dftu_capability_satisfies(&c, &r));  // id mismatch
}
