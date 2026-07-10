#include <dftracer/utils/server/http_request.h>
#include <dftracer/utils/server/http_response.h>
#include <dftracer/utils/server/router.h>
#include <dftracer/utils/server/viz_ui.h>

#include <string>

namespace dftracer::utils::server {

static coro::CoroTask<HttpResponse> handle_index() {
    co_return HttpResponse::ok(std::string(viz_index_html()),
                               "text/html; charset=utf-8");
}

void register_viz_ui(Router& router) {
    RouteHandler serve_index = [](const HttpRequest&, const QueryParams&)
        -> coro::CoroTask<HttpResponse> { co_return co_await handle_index(); };
    router.get("/", serve_index);
    router.get("/index.html", serve_index);
}

}  // namespace dftracer::utils::server
