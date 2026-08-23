#include <dftracer/utils/json/json_escape.h>
#include <dftracer/utils/trace/schema.h>
#include <dftracer/utils/trace/views/view_counter_format.h>
#include <dftracer/utils/trace/views/view_executor.h>

#include <cstdint>
#include <vector>

namespace dftracer::utils::trace::views::detail {

using json::append_json_escaped;

namespace {

void append_number(double v, std::string& out) {
    if (v == static_cast<double>(static_cast<std::int64_t>(v)) && v < 9.0e15 &&
        v > -9.0e15) {
        out += std::to_string(static_cast<std::int64_t>(v));
    } else {
        out += std::to_string(v);
    }
}

int col_index(const std::vector<std::string>& cols, const std::string& name) {
    for (std::size_t i = 0; i < cols.size(); ++i)
        if (cols[i] == name) return static_cast<int>(i);
    return -1;
}

}  // namespace

std::string counter_line(const std::vector<std::string>& group_cols,
                         const std::vector<std::string>& keys,
                         const std::vector<std::string>& value_cols,
                         const std::vector<double>& values) {
    const int i_bucket = col_index(group_cols, "time_bucket");
    const int i_name = col_index(group_cols, "name");
    const int i_cat = col_index(group_cols, "cat");
    const int i_pid = col_index(group_cols, "pid");
    const int i_tid = col_index(group_cols, "tid");

    const std::string& name =
        i_name >= 0 ? keys[i_name] : (i_cat >= 0 ? keys[i_cat] : "");
    const std::string cat = i_cat >= 0 ? keys[i_cat] : "";

    std::string s = "{\"name\":\"";
    append_json_escaped(s, name);
    s += "\",\"cat\":\"";
    append_json_escaped(s, cat);
    s += "\",\"pid\":";
    s += i_pid >= 0 && !keys[i_pid].empty() ? keys[i_pid] : "0";
    s += ",\"tid\":";
    s += i_tid >= 0 && !keys[i_tid].empty() ? keys[i_tid] : "0";
    s += ",\"ts\":";
    s += i_bucket >= 0 && !keys[i_bucket].empty() ? keys[i_bucket] : "0";
    s += ",\"ph\":";
    s += std::to_string(phase_to_int(RecordPhase::COUNTER));
    s += ",\"type\":";
    s += std::to_string(event_type_to_int(EventType::PSUTIL));
    s += ",\"args\":{";

    bool first = true;
    for (std::size_t j = 0; j < value_cols.size(); ++j) {
        if (!first) s += ",";
        first = false;
        s += "\"";
        append_json_escaped(s, value_cols[j]);
        s += "\":";
        append_number(values[j], s);
    }
    for (std::size_t k = 0; k < group_cols.size(); ++k) {
        const auto& c = group_cols[k];
        if (c == "time_bucket" || c == "name" || c == "cat" || c == "pid" ||
            c == "tid")
            continue;
        if (!first) s += ",";
        first = false;
        s += "\"";
        append_json_escaped(s, c);
        s += "\":\"";
        append_json_escaped(s, keys[k]);
        s += "\"";
    }
    s += "}}";
    return s;
}

void emit_group_counter(const std::string& /*key*/, const AggAccum& a,
                        const ViewPlan& plan, ExportSink& sink) {
    std::vector<std::string> group_cols;
    if (plan.time_bucket_us > 0) group_cols.push_back("time_bucket");
    for (const auto& gk : plan.group_by)
        group_cols.push_back(group_col_name(gk));
    std::vector<std::string> value_cols;
    std::vector<double> values;
    if (plan.agg.empty()) {
        value_cols.push_back("count");
        values.push_back(static_cast<double>(a.count));
    } else {
        for (std::size_t i = 0; i < plan.agg.size(); ++i) {
            const auto& spec = plan.agg[i];
            if (spec.op == AggOp::ArgMax) continue;  // counter args are numeric
            value_cols.push_back(agg_col_name(spec));
            values.push_back(finalize_value(a, plan, i));
        }
    }
    for (const auto& [name, m] : a.dyn) {
        value_cols.push_back(name);
        values.push_back(m.n ? m.sum / static_cast<double>(m.n) : 0.0);
    }
    sink.write(counter_line(group_cols, a.keys, value_cols, values));
    sink.write("\n");
}

}  // namespace dftracer::utils::trace::views::detail
