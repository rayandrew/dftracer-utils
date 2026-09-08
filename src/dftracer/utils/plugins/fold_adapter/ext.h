#ifndef DFTRACER_UTILS_PLUGINS_FOLD_ADAPTER_EXT_H
#define DFTRACER_UTILS_PLUGINS_FOLD_ADAPTER_EXT_H

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/dataframe/agg.h>
#include <dftracer/utils/plugins/abi.h>

#include <string>
#include <string_view>
#include <vector>

// Private seam shared by the fold_adapter extension translation units
// (fold_adapter.cpp core plus fold_adapter/{agg,result,ops}.cpp). Each ext TU
// defines its own file-static g_* vtable and exposes it through the detail::
// accessor below; host_get_extension in the core TU calls the accessors. The
// remaining declarations are the free helpers referenced across more than one
// of those TUs.
namespace dftracer::utils::plugins {

// Per-slice engine-backed aggregation accumulator (dft.ext.agg). value_names
// are the distinct value/by columns the specs reference, ordered so each spec's
// value_col/by_col indexes into it.
struct AggAccum {
    std::string name;
    std::vector<std::string> key_names;
    std::vector<std::string> value_names;
    dataframe::AggStatePtr state;
};

inline std::string_view resolve_id(dftracer::utils::StringIntern& intern,
                                   dftu_str id) {
    if (id == DFTU_STR_NONE ||
        id >= dftracer::utils::StringIntern::FAST_CAPACITY)
        return {};
    return intern.resolve(id);
}

namespace detail {

// Each returns its extension's stateless host vtable, owned file-static by the
// TU that implements that extension's trampolines.
const void* agg_ext_vtable();
const void* result_ext_vtable();
const void* ops_ext_vtable();

}  // namespace detail

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_PLUGINS_FOLD_ADAPTER_EXT_H
