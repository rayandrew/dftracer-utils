#ifndef DFTRACER_UTILS_SRC_PLUGINS_UTILITY_REGISTRY_H
#define DFTRACER_UTILS_SRC_PLUGINS_UTILITY_REGISTRY_H

// Host side of the plugin utility registry: one unified lookup over every
// exported utility (scalar functors and reflected utilities alike), keyed by
// dftu_util_id. Defined in plugin_exports.cpp.

#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/plugins/abi.h>

#include <cstdint>

namespace dftracer::utils::plugins {

namespace coro = dftracer::utils::coro;

// Single-value descriptor for a utility id (< DFTU_UTIL__COUNT), or nullptr
// when the id is unknown or the utility is stream-only.
const dftu_utility* registry_find(std::uint32_t id);

// Byte sizes of a single-value utility's C in/out structs, for building a
// compose leaf that threads its value through the dftu_op engine; false for a
// stream-only or unknown id.
bool registry_run_sizes(std::uint32_t id, std::uint32_t* in_size,
                        std::uint32_t* out_size);

// Drive a utility id as a stream, delivering each output item to on_item; 0 on
// success, -1 on failure or when the id has no stream form.
int registry_run_stream(std::uint32_t id, const void* in,
                        dftu_stream_item_fn on_item, void* ud);

// Run a single-output utility id on the current executor and marshal in/out,
// resolving to the utility status; -1 for a stream-only or unknown id.
coro::CoroTask<int> registry_run_async(std::uint32_t id, const void* in,
                                       void* out);

// Drive a utility id as a stream on the current executor, delivering each
// marshalled item to on_item as it is produced; resolves to the utility status,
// -1 for an unknown id or an id with no stream form.
coro::CoroTask<int> registry_run_stream_async(std::uint32_t id, const void* in,
                                              dftu_stream_item_fn on_item,
                                              void* ud);

// Pull-model stream handle: owns the native input, the suspended generator, and
// the per-item arena; the concrete driver embeds it as a base.
struct StreamDriver {
    // Reset the arena, pull one item, and on a value marshal it into the
    // driver-owned C slot, setting *out_item (BORROWED until the next call or
    // destroy) or nullptr at end-of-stream. Resolves to the utility status.
    coro::CoroTask<int> (*next)(StreamDriver* self, const void** out_item);
    void (*destroy)(StreamDriver* self);
};

// Build a pull-model driver for a stream utility id (or a single-value utility
// with process()); NULL for a non-stream or unknown id. The caller owns the
// driver and must call its destroy exactly once.
StreamDriver* registry_stream_open(std::uint32_t id, const void* in);

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_SRC_PLUGINS_UTILITY_REGISTRY_H
