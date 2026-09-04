#include <dftracer/utils/core/common/hash/fnv1a.h>
#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/agg_expr.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/fold_adapter_ext.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dftracer::utils::plugins {
namespace {

::dftu_agg* host_agg_new(void* h, const char* name,
                         const char* const* key_names, std::uint32_t key_n,
                         const ::dftu_agg_col* specs, std::uint32_t spec_n) {
    return static_cast<PluginFold*>(h)->agg_new(name, key_names, key_n, specs,
                                                spec_n);
}

void host_agg_accumulate(void* h, ::dftu_agg* a, const ::dftu_dataframe* df) {
    static_cast<PluginFold*>(h)->agg_accumulate(a, df);
}

const dftu_ext_agg g_agg = {host_agg_new, host_agg_accumulate};

}  // namespace

const void* detail::agg_ext_vtable() { return &g_agg; }

::dftu_agg* PluginFold::agg_new(const char* name, const char* const* key_names,
                                std::uint32_t key_n,
                                const ::dftu_agg_col* specs,
                                std::uint32_t spec_n) {
    if (!name || (key_n && !key_names) || spec_n == 0 || !specs) return nullptr;
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(name);
    if (std::unique_ptr<AggAccum>* slot = aggs_.find(key))
        return reinterpret_cast<::dftu_agg*>(slot->get());

    std::vector<std::string> value_names;
    auto column_index = [&](const char* col) -> std::int32_t {
        if (!col) return -1;
        for (std::size_t i = 0; i < value_names.size(); ++i)
            if (value_names[i] == col) return static_cast<std::int32_t>(i);
        value_names.emplace_back(col);
        return static_cast<std::int32_t>(value_names.size() - 1);
    };

    std::vector<dataframe::AggSpec> aspecs;
    aspecs.reserve(spec_n);
    for (std::uint32_t i = 0; i < spec_n; ++i) {
        if (specs[i].op < DFTU_AGG_COUNT ||
            specs[i].op > DFTU_AGG_COUNT_VALID) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin agg '%s' aggregate %u has an out-of-range op code %d",
                name, i, specs[i].op);
            return nullptr;
        }
        if (!specs[i].out || !specs[i].out[0]) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin agg '%s' aggregate %u is missing an output name", name,
                i);
            return nullptr;
        }
        dataframe::AggSpec s;
        s.op = static_cast<dataframe::AggOp>(specs[i].op);
        s.value_col = column_index(specs[i].value);
        s.out = specs[i].out;
        s.param = specs[i].param;
        s.by_col = column_index(specs[i].by);
        aspecs.push_back(std::move(s));
    }

    std::unique_ptr<AggAccum>* slot = aggs_.get_or_create(key, [&] {
        auto acc = std::make_unique<AggAccum>();
        acc->name = name;
        acc->key_names.reserve(key_n);
        for (std::uint32_t i = 0; i < key_n; ++i)
            acc->key_names.emplace_back(key_names[i] ? key_names[i] : "");
        acc->value_names = std::move(value_names);
        acc->state = dataframe::agg_new(std::move(aspecs));
        return acc;
    });
    return slot ? reinterpret_cast<::dftu_agg*>(slot->get()) : nullptr;
}

void PluginFold::agg_accumulate(::dftu_agg* a, const ::dftu_dataframe* df) {
    if (!a || !df) return;
    AggAccum& acc = *reinterpret_cast<AggAccum*>(a);
    if (!acc.state) return;

    // Series wraps and frees each shared handle; a missing column yields a null
    // handle, so skip the whole batch rather than accumulate a partial key.
    std::vector<dataframe::Series> owned;
    owned.reserve(acc.key_names.size() + acc.value_names.size());
    auto column = [&](const std::string& n) -> const dataframe::Series* {
        dftu_series* h = dftu_dataframe_column(df, n.c_str());
        if (!h) return nullptr;
        owned.emplace_back(h);
        return &owned.back();
    };

    std::vector<const dataframe::Series*> keys;
    keys.reserve(acc.key_names.size());
    for (const std::string& n : acc.key_names) {
        const dataframe::Series* c = column(n);
        if (!c) return;
        keys.push_back(c);
    }
    std::vector<const dataframe::Series*> values;
    values.reserve(acc.value_names.size());
    for (const std::string& n : acc.value_names) {
        const dataframe::Series* c = column(n);
        if (!c) return;
        values.push_back(c);
    }

    try {
        dataframe::agg_accumulate(*acc.state, keys, values);
    } catch (...) {
        DFTRACER_UTILS_LOG_ERROR("Plugin agg '%s' accumulate failed",
                                 acc.name.c_str());
    }
}

void PluginFold::materialize_aggs() {
    if (!named_results_) return;
    for (std::unique_ptr<AggAccum>& acc : aggs_) {
        if (!acc || !acc->state) continue;
        try {
            dataframe::DataFrame out =
                dataframe::agg_finalize(*acc->state, acc->key_names);
            // Move the columns into a dftu_dataframe handle (the engine's own
            // ABI boundary type) so the result crosses as our DataFrame, no
            // Arrow round-trip.
            std::vector<dftu_series*> handles;
            handles.reserve(out.columns.size());
            std::vector<const char*> names;
            names.reserve(out.names.size());
            for (dataframe::Series& c : out.columns)
                handles.push_back(c.release());
            for (const std::string& n : out.names) names.push_back(n.c_str());
            dftu_dataframe* h =
                dftu_dataframe_new(names.data(), handles.data(),
                                   static_cast<std::int32_t>(handles.size()));
            if (h) named_results_->emit_frame(acc->name.c_str(), h);
        } catch (...) {
        }
    }
}

}  // namespace dftracer::utils::plugins
