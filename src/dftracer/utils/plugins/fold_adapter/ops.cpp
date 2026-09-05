#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/plugins/fold_adapter/ext.h>

#include <cstdint>

namespace dftracer::utils::plugins {
namespace {

::dftu_series* host_ops_run(void*, const char* name,
                            const ::dftu_series* const* in, std::uint32_t n_in,
                            const ::dftu_op_arg* args) {
    const ::dftu_op_desc* op = ::dftu_op_find(name);
    if (!op) return nullptr;
    return ::dftu_op_run(op, in, n_in, args);
}

void host_ops_run_aggregate(void*, const char* name,
                            const ::dftu_series* const* in, std::uint32_t n_in,
                            const ::dftu_op_arg* args, ::dftu_scalar* out,
                            int* ok) {
    if (out) *out = ::dftu_scalar{};
    const ::dftu_op_desc* op = name ? ::dftu_op_find(name) : nullptr;
    if (!op || n_in != 1) {
        if (ok) *ok = 0;
        return;
    }
    int run_ok = 0;
    ::dftu_scalar result = ::dftu_op_run_aggregate(op, in[0], args, &run_ok);
    if (out) *out = result;
    if (ok) *ok = run_ok;
}

::dftu_dataframe* host_ops_run_frame(void*, const char* name,
                                     const ::dftu_dataframe* const* in,
                                     std::uint32_t n_in,
                                     const ::dftu_op_arg* args) {
    const ::dftu_op_desc* op = ::dftu_op_find(name);
    if (!op) return nullptr;
    return ::dftu_op_run_frame(op, in, n_in, args);
}

const ::dftu_op_desc* host_ops_find(void*, const char* name) {
    return ::dftu_op_find(name);
}

int host_ops_register(void*, const ::dftu_op_desc* desc) {
    return ::dftu_op_register(desc);
}

const ::dftu_ext_ops g_ops = {host_ops_run, host_ops_run_aggregate,
                              host_ops_run_frame, host_ops_find,
                              host_ops_register};

}  // namespace

const void* detail::ops_ext_vtable() { return &g_ops; }

}  // namespace dftracer::utils::plugins
