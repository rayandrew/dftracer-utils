#ifndef DFTRACER_UTILS_CORE_PIPELINE_EXECUTOR_H
#define DFTRACER_UTILS_CORE_PIPELINE_EXECUTOR_H

#include <concurrentqueue.h>
#include <dftracer/utils/core/common/platform_compat.h>
#include <dftracer/utils/core/common/timer_service.h>
#include <dftracer/utils/core/common/typedefs.h>
#include <dftracer/utils/core/coro/coro.h>
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/core/io/io_backend.h>

#include <any>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dftracer::utils {

class Task;
class CoroScope;
class Scheduler;

struct ExecutorConfig {
    std::size_t num_threads = 0;   // 0 = hardware_concurrency
    std::chrono::seconds idle_timeout{5};
    std::chrono::seconds deadlock_timeout{10};
    std::size_t io_pool_size = 0;  // 0 = hardware_concurrency
    io::IoBackendType io_backend_type = io::IoBackendType::AUTO;
    unsigned io_batch_threshold = 16;
};

/**
 * Task information for progress tracking
 */
struct TaskInfo {
    TaskIndex task_id;
    TaskIndex parent_task_id;  // -1 for root tasks
    std::string name;
    std::size_t worker_id;     // Which worker is executing

    enum State {
        QUEUED,                // In queue (shared or local)
        RUNNING,               // Currently executing
        WAITING,               // Waiting for child tasks
        COMPLETED,             // Successfully finished
        FAILED                 // Failed with error
    } state;

    std::chrono::steady_clock::time_point queued_at;
    std::chrono::steady_clock::time_point started_at;
    std::chrono::steady_clock::time_point completed_at;

    // Child tracking
    std::vector<TaskIndex> child_task_ids;
    std::atomic<std::size_t> completed_children{0};

    // Error info
    std::string error_message;

    // Queue location
    enum Location { SHARED_QUEUE, LOCAL_QUEUE, EXECUTING, DONE } location;
};

/**
 * Task progress information
 */
struct TaskProgress {
    TaskIndex task_id;
    std::string name;
    std::string state;  // "queued", "running", "waiting", "completed", "failed"

    // Timing
    double queued_duration_ms;
    double execution_duration_ms;

    // Progress
    std::size_t total_subtasks;
    std::size_t completed_subtasks;
    double progress_percentage;  // 0-100

    // Location
    // "shared_queue", "worker_2_local", "executing_on_worker_3"
    std::string location;

    // Children
    std::vector<TaskProgress> children;  // Recursive structure!
};

/**
 * Executor progress report
 */
struct ExecutorProgress {
    // Overall stats
    std::size_t total_tasks_submitted;
    std::size_t tasks_queued;
    std::size_t tasks_running;
    std::size_t tasks_completed;
    std::size_t tasks_failed;

    // Queue depths
    std::vector<std::size_t> worker_queue_depths;

    // Task tree (root tasks with their children)
    std::vector<TaskProgress> root_tasks;

    // Worker states
    struct WorkerStatus {
        std::size_t worker_id;
        bool is_idle;
        std::optional<TaskIndex> current_task_id;
        std::string current_task_name;
        std::size_t local_queue_depth;
    };
    std::vector<WorkerStatus> workers;

    // Errors
    // task_id, error_msg
    std::vector<std::pair<TaskIndex, std::string>> recent_errors;
};

/**
 * Executor - where coroutines get resumed.
 */
class Executor {
   public:
    virtual ~Executor() = default;

    virtual void enqueue(std::coroutine_handle<> handle,
                         TaskIndex task_id = -1) = 0;
    virtual void schedule_coroutine_resumption(
        std::coroutine_handle<> handle) = 0;
    virtual void schedule_destroy(std::coroutine_handle<> handle) = 0;
    virtual void mark_coro_completed(TaskIndex id) = 0;

    virtual bool is_running() const = 0;
    virtual std::size_t get_num_threads() const = 0;
    virtual std::size_t get_io_pool_size() const = 0;
    virtual TimerService& get_timer_service() = 0;

    /// Run pending work on the calling thread until `done()` holds.
    ///
    /// The caller is lending its thread rather than parking it, so an
    /// executor with a shared queue keeps serving it and loses no capacity.
    virtual void drive_until(const std::function<bool()>& done) = 0;

    /// Without a backend the io layer falls back to blocking syscalls.
    virtual bool has_io_backend() const noexcept = 0;
    virtual io::IoBackend& io_backend() = 0;

    /// Executor driving the calling thread, nullptr if none.
    static Executor* current() noexcept;

    /// Sets current(), returning the previous value.
    static Executor* set_current(Executor* e) noexcept;
};

/**
 * TaskExecutor - an Executor that also runs DAG tasks and owns its lifecycle.
 *
 * What Pipeline, Scheduler and Watchdog require. Implement this to run the
 * pipeline on something other than the bundled thread pool.
 */
class TaskExecutor : public Executor {
   public:
    using CompletionCallback = std::function<void(std::shared_ptr<Task>)>;

    virtual void start() = 0;
    virtual void shutdown() = 0;
    virtual void request_shutdown() = 0;

    virtual void set_completion_callback(CompletionCallback callback) = 0;
    virtual void set_scheduler(Scheduler* scheduler) = 0;

    virtual void submit_task(std::shared_ptr<Task> task,
                             std::shared_ptr<std::any> input,
                             TaskIndex parent_task_id = -1) = 0;
    virtual TaskIndex enqueue_tracked(
        coro::Coro coro, std::string name,
        std::shared_ptr<std::atomic<TaskIndex>> tid_out = nullptr) = 0;

    /// Task the calling worker is running, -1 when not on a worker thread.
    virtual TaskIndex current_worker_task_id() const = 0;

    virtual bool is_responsive() const = 0;
    virtual ExecutorProgress get_progress() const = 0;
};

/// Build the executor described by `config`. The implementation is chosen
/// here so callers never name a concrete executor type.
std::unique_ptr<TaskExecutor> make_task_executor(const ExecutorConfig& config);

void set_current_worker_context(void* context);

/// Push a coroutine handle onto the current worker's thread-local
/// destroy list.  The worker drains this list after each resume(),
/// guaranteeing the frame is fully suspended before destruction.
void schedule_thread_local_destroy(std::coroutine_handle<> h);

/// Drain the current thread's pending-destroy list.
void drain_thread_local_destroys();

}  // namespace dftracer::utils

#endif  // DFTRACER_UTILS_CORE_PIPELINE_EXECUTOR_H
