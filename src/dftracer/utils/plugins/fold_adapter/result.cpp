#include <dftracer/utils/core/common/hash/fnv1a.h>
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/fold_adapter/ext.h>

#include <cstdint>

namespace dftracer::utils::plugins {
namespace {

::dftu_handle* host_handle_get(void* h, const char* cap_id,
                               dftu_monoid_kind kind) {
    return reinterpret_cast<::dftu_handle*>(
        static_cast<PluginFold*>(h)->handle_get(cap_id, kind));
}

void host_handle_add_u64(void*, ::dftu_handle* hd, std::uint64_t v) {
    if (hd) reinterpret_cast<MonoidAccumulator*>(hd)->add_u64(v);
}

void host_handle_add_f64(void*, ::dftu_handle* hd, double v, double w) {
    if (hd) reinterpret_cast<MonoidAccumulator*>(hd)->add_f64(v, w);
}

int host_handle_result(void* h, const char* cap_id, dftu_monoid_value* out) {
    return static_cast<PluginFold*>(h)->handle_result(cap_id, out);
}

const dftu_ext_handles g_handles = {host_handle_get, host_handle_add_u64,
                                    host_handle_add_f64, host_handle_result};

void host_result_emit(void* h, const char* name, const void* data,
                      std::uint64_t len) {
    static_cast<PluginFold*>(h)->result_emit(name, data, len);
}

int host_result_emit_arrow(void* h, const char* name, ::ArrowArray* a,
                           ::ArrowSchema* s) {
    return static_cast<PluginFold*>(h)->result_emit_arrow(name, a, s);
}

int host_result_emit_frame(void* h, const char* name, ::dftu_dataframe* df) {
    return static_cast<PluginFold*>(h)->result_emit_frame(name, df);
}

int host_result_emit_lazyframe(void* h, const char* name,
                               ::dftu_lazyframe* lf) {
    return static_cast<PluginFold*>(h)->result_emit_lazyframe(name, lf);
}

const dftu_ext_result g_result = {host_result_emit, host_result_emit_arrow,
                                  host_result_emit_frame,
                                  host_result_emit_lazyframe};

}  // namespace

const void* detail::handles_ext_vtable() { return &g_handles; }
const void* detail::result_ext_vtable() { return &g_result; }

MonoidAccumulator* PluginFold::handle_get(const char* cap_id,
                                          dftu_monoid_kind kind) {
    if (!cap_id) return nullptr;
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(cap_id);
    return handles_.get_or_create(key, [&] { return MonoidAccumulator(kind); });
}

int PluginFold::handle_result(const char* cap_id,
                              dftu_monoid_value* out) const {
    if (!cap_id || !results_) return -1;
    auto it = results_->values.find(dftracer::utils::hash::fnv1a_hash(cap_id));
    if (it == results_->values.end()) return -1;
    if (out) *out = it->second;
    return 0;
}

void PluginFold::result_emit(const char* name, const void* data,
                             std::uint64_t len) {
    if (named_results_) named_results_->emit_blob(name, data, len);
}

int PluginFold::result_emit_arrow(const char* name, ::ArrowArray* a,
                                  ::ArrowSchema* s) {
    return named_results_ ? named_results_->emit_arrow(name, a, s) : -1;
}

int PluginFold::result_emit_frame(const char* name, ::dftu_dataframe* df) {
    if (!named_results_) return -1;
    named_results_->emit_frame(name, df);
    return 0;
}

int PluginFold::result_emit_lazyframe(const char* name, ::dftu_lazyframe* lf) {
    if (!named_results_) return -1;
    named_results_->emit_lazyframe(name, lf);
    return 0;
}

}  // namespace dftracer::utils::plugins
