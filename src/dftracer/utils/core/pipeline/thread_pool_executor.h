#ifndef DFTRACER_UTILS_CORE_PIPELINE_THREAD_POOL_EXECUTOR_H
#define DFTRACER_UTILS_CORE_PIPELINE_THREAD_POOL_EXECUTOR_H

#include <concurrentqueue.h>
#include <dftracer/utils/core/common/platform_compat.h>
#include <dftracer/utils/core/common/timer_service.h>
#include <dftracer/utils/core/common/typedefs.h>
#include <dftracer/utils/core/coro/coro.h>
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/core/io/io_backend.h>
#include <dftracer/utils/core/pipeline/executor.h>

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
class Scheduler;

/**
 * ThreadPoolExecutor - Executes tasks from queue using worker thread pool
 *
 * Features:
 * - Large thread pool for CPU/IO-bound work
 * - Pulls tasks from queue
 * - Executes task functions
 * - Notifies scheduler on completion via callback
 *
 * Thread pool size: N threads (default: hardware_concurrency)
 */
class ThreadPoolExecutor : public TaskExecutor {
   public:
    using TaskExecutor::CompletionCallback;

   private:
    // Worker context for per-thread state.
    // Aligned to avoid false sharing between adjacent workers.
    struct alignas(DFTRACER_OPTIMAL_ALIGNMENT) WorkerContext {
        std::size_t worker_id;

        // Health monitoring for watchdog
        std::atomic<bool> is_idle{false};
        std::atomic<uint64_t> tasks_executed{0};
        std::chrono::steady_clock::time_point last_activity;

        // Current task info (for debugging/watchdog)
        std::atomic<TaskIndex> current_task_id{-1};
        std::string current_task_name;
        std::mutex task_name_mutex;

        // Worker thread
        std::thread thread;

        explicit WorkerContext(std::size_t id) : worker_id(id) {}
    };

    // Per-worker contexts
    std::vector<std::unique_ptr<WorkerContext>> workers_;
    std::atomic<std::size_t> next_worker_{0};  // For round-robin submission

    std::atomic<bool> running_{false};
    std::size_t num_threads_;

    CompletionCallback completion_callback_;

    // Reference to scheduler for dynamic task context
    Scheduler* scheduler_{nullptr};

    // Global tracking, padded to avoid false sharing between counters
    // and with work_signal_ below.
    alignas(DFTRACER_OPTIMAL_ALIGNMENT)
        std::atomic<std::size_t> tasks_completed_{0};
    alignas(DFTRACER_OPTIMAL_ALIGNMENT) std::atomic<std::size_t> tasks_started_{
        0};
    alignas(DFTRACER_OPTIMAL_ALIGNMENT)
        std::atomic<std::size_t> total_tasks_submitted_{0};

    std::atomic<std::int64_t> last_activity_ns_;

    // Shutdown coordination
    std::atomic<bool> shutdown_requested_{false};

    // Responsiveness timeout thresholds
    std::chrono::seconds idle_timeout_;
    std::chrono::seconds deadlock_timeout_;
    // Timer service for timeout operations
    TimerService timer_service_;

    // Task registry for progress tracking
    std::unordered_map<TaskIndex, TaskInfo> task_registry_;
    mutable std::shared_mutex registry_mutex_;

    // Counter for tracked-coro IDs (negative to avoid collision with DAG IDs).
    std::atomic<TaskIndex> next_coro_task_id_{-1000000};

    // Run queue entry. -1 means untracked.
    struct RunQueueEntry {
        std::coroutine_handle<> handle{};
        TaskIndex task_id{-1};
        long long monitor_id{-1};  // coroutine monitor id, -1 if untracked
    };

    moodycamel::ConcurrentQueue<RunQueueEntry> run_queue_;
    alignas(DFTRACER_OPTIMAL_ALIGNMENT) std::atomic<std::uint64_t> work_signal_{
        0};

    // Deferred destruction queue for released Coro handles.
    // FinalAwaiter pushes here; worker loop drains periodically.
    moodycamel::ConcurrentQueue<std::coroutine_handle<>> destroy_queue_;

    // I/O backend (owned by executor, created by factory)
    std::unique_ptr<io::IoBackend> io_backend_;

    // Configuration (stored from ExecutorConfig)
    std::size_t io_pool_size_ = 0;
    io::IoBackendType io_backend_type_ = io::IoBackendType::AUTO;
    unsigned io_batch_threshold_ = 16;

   public:
    /**
     * Constructor
     */
    explicit ThreadPoolExecutor(const ExecutorConfig& config = {});

    ~ThreadPoolExecutor() override;

    // Prevent copying
    ThreadPoolExecutor(const ThreadPoolExecutor&) = delete;
    ThreadPoolExecutor& operator=(const ThreadPoolExecutor&) = delete;

    // Prevent moving (threads are not movable once started)
    ThreadPoolExecutor(ThreadPoolExecutor&&) = delete;
    ThreadPoolExecutor& operator=(ThreadPoolExecutor&&) = delete;

    /**
     * Start the executor (spawn worker threads)
     */
    void start() override;

    /**
     * Shutdown the executor gracefully
     */
    void shutdown() override;

    /**
     * Reset the executor (prepare for new execution)
     */
    void reset();

    /**
     * Set completion callback (called when task finishes)
     */
    void set_completion_callback(CompletionCallback callback) override;

    /**
     * Set scheduler reference (for CoroScope)
     */
    void set_scheduler(Scheduler* scheduler) override {
        scheduler_ = scheduler;
    }

    /**
     * Get timer service for timeout operations
     */
    TimerService& get_timer_service() override { return timer_service_; }

    /**
     * Check if executor is running
     */
    bool is_running() const override { return running_.load(); }

    /**
     * Get number of worker threads
     */
    std::size_t get_num_threads() const override { return num_threads_; }

    std::size_t get_io_pool_size() const override { return io_pool_size_; }

    /**
     * Check if an I/O backend is available
     */
    bool has_io_backend() const noexcept override {
        return io_backend_ != nullptr;
    }

    /**
     * Get the I/O backend (must check has_io_backend() first)
     */
    io::IoBackend& io_backend() override { return *io_backend_; }
    const io::IoBackend& io_backend() const { return *io_backend_; }

    /**
     * Request graceful shutdown
     * Stops accepting new tasks and waits for current tasks to complete
     */
    void request_shutdown() override;

    /**
     * Check if shutdown was requested
     */
    bool is_shutdown_requested() const { return shutdown_requested_.load(); }

    /**
     * Check if executor is responsive (making progress)
     *
     * Used by watchdog to detect if executor is hung.
     * Returns false if executor appears to be stuck or unresponsive.
     */
    bool is_responsive() const override;

    /**
     * Get full progress report
     */
    ExecutorProgress get_progress() const override;

    /**
     * Schedule a coroutine handle to be resumed on the executor's thread pool
     * This is a lightweight operation that submits the resumption as work
     * @param handle The coroutine handle to resume
     *
     * This is useful for when_all and other coroutine combinators that need
     * to resume coroutines from completion callbacks without directly calling
     * .resume()
     */
    void schedule_coroutine_resumption(std::coroutine_handle<> handle) override;

    /**
     * Enqueue a coroutine handle for execution on the thread pool.
     * This is the primary submission method for goroutines -- all
     * lightweight work funnels through here.
     * Cost: ~20ns (lock-free queue push + atomic signal).
     * @param handle The coroutine handle to resume
     * @param task_id Tracked-task id for registry progress, or -1 if untracked.
     */
    void enqueue(std::coroutine_handle<> handle,
                 TaskIndex task_id = -1) override;

    /**
     * Enqueue a Coro with progress tracking in task_registry_.
     */
    TaskIndex enqueue_tracked(
        coro::Coro coro, std::string name,
        std::shared_ptr<std::atomic<TaskIndex>> tid_out = nullptr) override;

    void mark_coro_completed(TaskIndex id) override;

    /**
     * Submit a Task for execution via a Coro (Phase 3 path).
     * Creates a run_task() Coro, registers in task_registry_,
     * and enqueues the released handle to run_queue_.
     * @param task The DAG task to execute
     * @param input Task input
     * @param parent_task_id Parent task ID for tracking (-1 for root)
     */
    void submit_task(std::shared_ptr<Task> task,
                     std::shared_ptr<std::any> input,
                     TaskIndex parent_task_id = -1) override;

    TaskIndex current_worker_task_id() const override;

    void drive_until(const std::function<bool()>& done) override;

    /**
     * Schedule a completed Coro handle for deferred destruction.
     * Called from CoroPromise::FinalAwaiter for released handles.
     * @param handle The coroutine handle at final_suspend
     */
    void schedule_destroy(std::coroutine_handle<> handle) override;

   private:
    /**
     * Worker thread main loop
     */
    void worker_thread(WorkerContext* context);

    /**
     * Update task location in registry
     */
    void update_task_location(TaskIndex task_id, TaskInfo::Location location,
                              std::size_t worker_id);

    /**
     * Build task progress tree recursively
     */
    TaskProgress build_task_progress_tree(
        TaskIndex task_id, std::unordered_set<TaskIndex>& processed) const;

    /**
     * Notify completion callback
     */
    void notify_completion(std::shared_ptr<Task> task);

    /**
     * Mark activity (task start or completion) for responsiveness tracking
     */
    void mark_activity();

    /**
     * Signal workers that new global work is available.
     */
    void signal_global_work();

    /**
     * Wake one worker thread.
     */
    void wake_one_worker();

    /**
     * Wake all worker threads.
     */
    void wake_all_workers();

    /**
     * Create a Coro that executes a Task with full bookkeeping.
     * The returned Coro is at initial_suspend -- caller must
     * set executor on promise and enqueue via release().
     */
    coro::Coro run_task(std::shared_ptr<Task> task,
                        std::shared_ptr<std::any> input);

    /**
     * Drain the destroy queue (called from worker loop).
     */
    void drain_destroy_queue();

    friend class Scheduler;
    friend struct coro::CoroPromise;
};

void* get_current_worker_context();

}  // namespace dftracer::utils

#endif  // DFTRACER_UTILS_CORE_PIPELINE_THREAD_POOL_EXECUTOR_H
