#ifndef DFTRACER_UTILS_PLUGINS_BUILD_HOST_H
#define DFTRACER_UTILS_PLUGINS_BUILD_HOST_H

#include <dftracer/utils/plugins/abi.h>

#include <string>

namespace dftracer::utils::plugins {

/// The host a plugin factory receives. It answers only the registration
/// groups - DFTU_EXT_OPS, DFTU_EXT_AGG, DFTU_EXT_PORTS - and, inside those,
/// only the registration slots. Anything that needs a scan (a fold, a batch,
/// the intern table, the runtime) does not exist yet at load, so every other
/// service returns NULL and records the denial: the loader then fails the load
/// naming it, rather than letting the factory dereference a host that is not
/// there. What a factory may do is enforced here, not documented.
class BuildHost {
   public:
    explicit BuildHost(std::string plugin_name);

    /// Handed to the factory; valid only for that call.
    dftu_host* host() { return &host_; }

    /// The first service the factory was denied, empty when it stayed inside
    /// the registration surface.
    const std::string& denied() const { return denied_; }

    const std::string& plugin_name() const { return plugin_name_; }

    /// Record `what` as denied; only the first is kept, so the loader reports
    /// the call that first stepped outside the build phase.
    void deny(const char* what);

   private:
    std::string plugin_name_;
    std::string denied_;
    dftu_host host_{};
};

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_PLUGINS_BUILD_HOST_H
