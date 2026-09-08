#ifndef DFTRACER_UTILS_UTILITIES_HOST_OPS_H
#define DFTRACER_UTILS_UTILITIES_HOST_OPS_H

namespace dftracer::utils::utilities {

/**
 * @brief Register the utilities-layer host ops into the dataframe op registry.
 *
 * Adds the file, filesystem and text utilities a plugin reaches by name
 * through dftu.ext.ops: dftu.fs.scan_dir, dftu.fs.scan_dir_pattern,
 * dftu.file.compress, dftu.file.decompress and dftu.text.line_filter. Each is
 * synchronous and materializes its result, unlike the async utility it wraps.
 *
 * Idempotent and thread-safe. A shared build runs it from a library
 * initializer, so only a consumer that links the static library (where the
 * linker may drop that initializer's object file) needs to call it.
 */
void register_host_ops();

}  // namespace dftracer::utils::utilities

#endif  // DFTRACER_UTILS_UTILITIES_HOST_OPS_H
