#ifndef DFTRACER_UTILS_SERVER_VIZ_UI_H
#define DFTRACER_UTILS_SERVER_VIZ_UI_H

#include <string_view>

namespace dftracer::utils::server {

class Router;

/// Return the embedded single-page trace viewer. The body is generated at build
/// time from web/dist/index.html by cmake/scripts/embed_asset.cmake.
std::string_view viz_index_html();

/// Register the visualization UI route: serves the embedded SPA at GET / and
/// GET /index.html.
void register_viz_ui(Router& router);

}  // namespace dftracer::utils::server

#endif  // DFTRACER_UTILS_SERVER_VIZ_UI_H
