#ifndef DFTRACER_UTILS_CORE_CORO_ABI_H
#define DFTRACER_UTILS_CORE_CORO_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dftu_task dftu_task;
typedef struct dftu_runtime dftu_runtime;

dftu_runtime* dftu_default_runtime(void);

/* Drive t to completion on rt, blocking the caller (pool-safe when reentrant).
   Consumes t. 0 on success, -1 if it threw. */
int dftu_task_run(dftu_runtime* rt, dftu_task* t);

/* Combine n tasks into one; consumes the inputs. NULL if n == 0. */
dftu_task* dftu_task_when_all(dftu_task* const* ts, uint32_t n);
dftu_task* dftu_task_when_any(dftu_task* const* ts, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* DFTRACER_UTILS_CORE_CORO_ABI_H */
