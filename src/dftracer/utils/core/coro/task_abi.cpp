#include <dftracer/utils/core/coro/abi.h>
#include <dftracer/utils/core/coro/task_abi.h>
#include <dftracer/utils/core/coro/when_all.h>
#include <dftracer/utils/core/coro/when_any.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/core/tasks/coro_scope.h>

#include <utility>
#include <vector>

namespace dftracer::utils {

dftu_task* task_to_abi(coro::CoroTask<void>&& task) {
    return reinterpret_cast<dftu_task*>(
        new coro::CoroTask<void>(std::move(task)));
}

namespace {

std::vector<coro::CoroTask<void>> take(dftu_task* const* ts, std::uint32_t n) {
    std::vector<coro::CoroTask<void>> out;
    out.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        auto* p = reinterpret_cast<coro::CoroTask<void>*>(ts[i]);
        out.push_back(std::move(*p));
        delete p;
    }
    return out;
}

// when_any cannot hold a void result, so race valued tasks and discard it.
coro::CoroTask<int> as_valued(coro::CoroTask<void> t) {
    co_await std::move(t);
    co_return 0;
}

}  // namespace

}  // namespace dftracer::utils

using dftracer::utils::CoroScope;
using dftracer::utils::default_runtime;
using dftracer::utils::Runtime;
using dftracer::utils::task_to_abi;
using dftracer::utils::coro::CoroTask;
using dftracer::utils::coro::when_all;
using dftracer::utils::coro::when_any;

extern "C" {

dftu_runtime* dftu_default_runtime(void) {
    return reinterpret_cast<dftu_runtime*>(&default_runtime());
}

int dftu_task_run(dftu_runtime* rt, dftu_task* t) {
    if (!rt || !t) return -1;
    auto* runtime = reinterpret_cast<Runtime*>(rt);
    auto* task = reinterpret_cast<CoroTask<void>*>(t);
    int rc = 0;
    try {
        runtime->run_blocking(
            "dftu_task_run",
            [&](CoroScope&) -> CoroTask<void> { co_await std::move(*task); });
    } catch (...) {
        rc = -1;
    }
    delete task;
    return rc;
}

dftu_task* dftu_task_when_all(dftu_task* const* ts, uint32_t n) {
    if (n == 0 || !ts) return nullptr;
    return task_to_abi([](std::vector<CoroTask<void>> inner) -> CoroTask<void> {
        co_await when_all(std::move(inner));
    }(dftracer::utils::take(ts, n)));
}

dftu_task* dftu_task_when_any(dftu_task* const* ts, uint32_t n) {
    if (n == 0 || !ts) return nullptr;
    return task_to_abi([](std::vector<CoroTask<void>> inner) -> CoroTask<void> {
        std::vector<CoroTask<int>> valued;
        valued.reserve(inner.size());
        for (auto& t : inner)
            valued.push_back(dftracer::utils::as_valued(std::move(t)));
        co_await when_any(std::move(valued));
    }(dftracer::utils::take(ts, n)));
}

}  // extern "C"
