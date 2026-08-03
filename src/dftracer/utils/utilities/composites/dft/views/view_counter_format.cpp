#include <dftracer/utils/utilities/common/json/json_escape.h>
#include <dftracer/utils/utilities/composites/dft/schema.h>
#include <dftracer/utils/utilities/composites/dft/views/view_counter_format.h>
#include <dftracer/utils/utilities/composites/dft/views/view_executor.h>

#include <cstdint>
#include <vector>

namespace dftracer::utils::utilities::composites::dft::views::detail {

using common::json::append_json_escaped;

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

std::string counter_line(const ResultTable& t, const ResultRow& row) {
    const int i_bucket = col_index(t.group_columns, "time_bucket");
    const int i_name = col_index(t.group_columns, "name");
    const int i_cat = col_index(t.group_columns, "cat");
    const int i_pid = col_index(t.group_columns, "pid");
    const int i_tid = col_index(t.group_columns, "tid");

    const std::string& name =
        i_name >= 0 ? row.keys[i_name] : (i_cat >= 0 ? row.keys[i_cat] : "");
    const std::string cat = i_cat >= 0 ? row.keys[i_cat] : "";

    std::string s = "{\"name\":\"";
    append_json_escaped(s, name);
    s += "\",\"cat\":\"";
    append_json_escaped(s, cat);
    s += "\",\"pid\":";
    s += i_pid >= 0 && !row.keys[i_pid].empty() ? row.keys[i_pid] : "0";
    s += ",\"tid\":";
    s += i_tid >= 0 && !row.keys[i_tid].empty() ? row.keys[i_tid] : "0";
    s += ",\"ts\":";
    s +=
        i_bucket >= 0 && !row.keys[i_bucket].empty() ? row.keys[i_bucket] : "0";
    s += ",\"ph\":";
    s += std::to_string(phase_to_int(RecordPhase::COUNTER));
    s += ",\"type\":";
    s += std::to_string(event_type_to_int(EventType::PSUTIL));
    s += ",\"args\":{";

    bool first = true;
    for (std::size_t j = 0; j < t.value_columns.size(); ++j) {
        if (!first) s += ",";
        first = false;
        s += "\"";
        append_json_escaped(s, t.value_columns[j]);
        s += "\":";
        append_number(row.values[j], s);
    }
    for (std::size_t k = 0; k < t.group_columns.size(); ++k) {
        const auto& c = t.group_columns[k];
        if (c == "time_bucket" || c == "name" || c == "cat" || c == "pid" ||
            c == "tid")
            continue;
        if (!first) s += ",";
        first = false;
        s += "\"";
        append_json_escaped(s, c);
        s += "\":\"";
        append_json_escaped(s, row.keys[k]);
        s += "\"";
    }
    s += "}}";
    return s;
}

void emit_group_counter(const std::string& /*key*/, const AggAccum& a,
                        const ViewPlan& plan, ExportSink& sink) {
    ResultTable h;
    if (plan.time_bucket_us > 0) h.group_columns.push_back("time_bucket");
    for (const auto& gk : plan.group_by)
        h.group_columns.push_back(group_col_name(gk));
    ResultRow row;
    row.keys = a.keys;
    if (plan.agg.empty()) {
        h.value_columns.push_back("count");
        row.values.push_back(static_cast<double>(a.count));
    } else {
        for (std::size_t i = 0; i < plan.agg.size(); ++i) {
            const auto& spec = plan.agg[i];
            if (spec.op == AggOp::ArgMax) continue;  // counter args are numeric
            h.value_columns.push_back(agg_col_name(spec));
            row.values.push_back(finalize_value(a, plan, i));
        }
    }
    for (const auto& [name, m] : a.dyn) {
        h.value_columns.push_back(name);
        row.values.push_back(m.n ? m.sum / static_cast<double>(m.n) : 0.0);
    }
    sink.write(counter_line(h, row));
    sink.write("\n");
}

}  // namespace dftracer::utils::utilities::composites::dft::views::detail
