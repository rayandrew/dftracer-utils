#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/fold_adapter/ext.h>

#include <cstdint>

namespace dftracer::utils::plugins {
namespace {

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

const void* detail::result_ext_vtable() { return &g_result; }

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
