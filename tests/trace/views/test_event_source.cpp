#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/trace/views/event_source.h>
#include <dftracer/utils/trace/views/fold_event.h>
#include <dftracer/utils/trace/views/view_aggregate.h>
#include <doctest/doctest.h>
#include <simdjson.h>

#include <map>
#include <string>

using dftracer::utils::StringIntern;

// DomSource must reproduce the existing extractors (top_or_args_str, agg_field)
// byte-for-byte, so the coming hot-path rewire is identical by construction.
using namespace dftracer::utils::trace::views::detail;

namespace {

std::string padded(std::string s) {
    s.resize(s.size() + simdjson::SIMDJSON_PADDING, '\0');
    return s;
}

simdjson::dom::element parse(simdjson::dom::parser& p, std::string& buf,
                             const char* json) {
    buf = padded(json);
    return p.parse(buf.data(), buf.size() - simdjson::SIMDJSON_PADDING, false)
        .value_unsafe();
}

}  // namespace

TEST_SUITE("EventSource") {
    TEST_CASE("DomSource.value matches top_or_args_str across field shapes") {
        simdjson::dom::parser p;
        std::string buf;
        auto root = parse(
            p, buf,
            R"({"ph":"X","name":"read","cat":"POSIX","pid":42,"tid":7,"ts":100,"dur":9,"args":{"fhash":"fh1","ret":4096,"level":"warn"}})");
        DomSource src(root);

        // top-level string, top-level int, args string, args int, missing.
        for (const char* field : {"name", "cat", "pid", "tid", "ts", "dur",
                                  "fhash", "ret", "level", "absent"}) {
            CAPTURE(field);
            CHECK(src.value(field) == top_or_args_str(root, field));
        }
    }

    TEST_CASE("DomSource.number matches a raw top-then-args agg_field") {
        simdjson::dom::parser p;
        std::string buf;
        auto root = parse(
            p, buf,
            R"({"ph":"X","name":"read","pid":1,"ts":10,"dur":3,"args":{"ret":8192,"offset":512}})");
        DomSource src(root);

        // Non-derived fields: agg_field is a plain top-then-args number.
        for (const char* field : {"pid", "ts", "dur", "ret", "offset"}) {
            CAPTURE(field);
            auto got = src.number(field);
            auto want = agg_field(root, field);
            REQUIRE(got.has_value() == want.has_value());
            if (got) CHECK(*got == doctest::Approx(*want));
        }
        CHECK_FALSE(src.number("absent").has_value());
    }

    TEST_CASE("append_value writes the same bytes value returns") {
        simdjson::dom::parser p;
        std::string buf;
        auto root =
            parse(p, buf,
                  R"({"ph":"X","name":"n","pid":123,"args":{"k":"v","m":9}})");
        DomSource src(root);
        for (const char* field : {"name", "pid", "k", "m", "absent"}) {
            std::string out = "pre|";
            src.append_value(out, field);
            CHECK(out == "pre|" + src.value(field));
        }
    }

    // The payoff: an event read through PodSource (owned, interned) must yield
    // the same bytes as through DomSource (live element), so AggFold over the
    // POD equals scan_fold over the element.
    TEST_CASE("PodSource reproduces DomSource on the same event") {
        simdjson::dom::parser p;
        std::string buf;
        auto root = parse(
            p, buf,
            R"({"ph":"X","name":"read","cat":"POSIX","pid":42,"tid":7,"ts":100,"dur":9,"args":{"fhash":"fh1","hhash":"hh2","ret":4096,"rate":1.5,"path":"/x"}})");

        StringIntern intern;
        FoldEvent ev = extract_fold_event(root, intern, /*needs_args=*/true);
        DomSource dom(root);
        PodSource pod(ev, intern);

        for (const char* field :
             {"name", "cat", "pid", "tid", "ts", "dur", "fhash", "hhash", "ret",
              "rate", "path", "absent"}) {
            CAPTURE(field);
            CHECK(pod.value(field) == dom.value(field));
            auto pn = pod.number(field);
            auto dn = dom.number(field);
            CHECK(pn.has_value() == dn.has_value());
            if (pn) CHECK(*pn == doctest::Approx(*dn));
        }

        // Group dims read args-only (fhash/hhash/arg), which must also match.
        for (const char* key : {"fhash", "hhash", "ret", "path", "absent"}) {
            CAPTURE(key);
            std::string pd, dd;
            pod.append_arg(pd, key);
            dom.append_arg(dd, key);
            CHECK(pd == dd);
        }

        std::map<std::string, double> dom_args, pod_args;
        dom.for_each_numeric_arg([&](std::string_view k, double v) {
            dom_args[std::string(k)] = v;
        });
        pod.for_each_numeric_arg([&](std::string_view k, double v) {
            pod_args[std::string(k)] = v;
        });
        CHECK(dom_args == pod_args);
        CHECK(pod.phase() == dom.phase());
    }

    TEST_CASE("for_each_numeric_arg visits every numeric arg once") {
        simdjson::dom::parser p;
        std::string buf;
        auto root = parse(
            p, buf,
            R"({"ph":"X","name":"n","args":{"ret":4096,"offset":512,"path":"/x","dur_d":1.5}})");
        DomSource src(root);

        std::map<std::string, double> seen;
        src.for_each_numeric_arg(
            [&](std::string_view k, double v) { seen[std::string(k)] = v; });

        CHECK(seen.size() == 3);  // ret, offset, dur_d -- not the string path
        CHECK(seen["ret"] == doctest::Approx(4096));
        CHECK(seen["offset"] == doctest::Approx(512));
        CHECK(seen["dur_d"] == doctest::Approx(1.5));
    }
}
