#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace sevobench {

namespace parallel_task_detail {

// Type-erased callable wrapper, similar to std::move_only_function<void()>.
class task_function_wrapper {
    struct impl_base {
        virtual void call() = 0;
        virtual ~impl_base() = default;
    };

    template <typename F>
    struct impl_type : impl_base {
        F f;
        void call() override { f(); }
        impl_type(F&& _f) : f(std::move(_f)) {}
    };

    std::unique_ptr<impl_base> impl;

public:
    // Never noexcept – the underlying task may throw, and packaged_task can
    // capture the exception normally.
    void operator()() { impl->call(); }

    template <typename F>
    task_function_wrapper(F&& f)
        : impl(new impl_type<F>(std::move(f))) {}

    task_function_wrapper() = default;
    task_function_wrapper(const task_function_wrapper&) = delete;
    task_function_wrapper& operator=(const task_function_wrapper&) = delete;
    task_function_wrapper(task_function_wrapper&&) noexcept = default;
    task_function_wrapper& operator=(task_function_wrapper&&) noexcept = default;
};

// Thread-safe queue that supports blocking pop with a stop flag.
template <typename T>
class threadsafe_queue {
    mutable std::mutex mut;
    std::deque<T> data_queue;
    std::condition_variable data_cond;

public:
    void push(T new_value) {
        {
            std::lock_guard<std::mutex> lk(mut);
            data_queue.push_back(std::move(new_value));
        }
        data_cond.notify_one();
    }

    // Non-blocking pop.
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lk(mut);
        if (data_queue.empty())
            return false;
        value = std::move(data_queue.front());
        data_queue.pop_front();
        return true;
    }

    // Block until a task is available OR the stop flag becomes true.
    // Returns false if the stop flag is set and the queue is empty.
    bool wait_and_pop(T& value, std::atomic<bool>& stop_flag) {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this, &stop_flag] {
            return !data_queue.empty() || stop_flag.load();
        });
        if (stop_flag.load() && data_queue.empty())
            return false;
        value = std::move(data_queue.front());
        data_queue.pop_front();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lk(mut);
        return data_queue.empty();
    }

    void notify_all() noexcept { data_cond.notify_all(); }
    void notify_one() noexcept { data_cond.notify_one(); }

    void request_stop(std::atomic<bool>& stop_flag) noexcept {
        {
            // Synchronize the predicate change with wait_and_pop's mutex so a
            // worker cannot miss the final notification between checking the
            // predicate and entering the condition-variable wait.
            std::lock_guard<std::mutex> lk(mut);
            stop_flag.store(true);
        }
        data_cond.notify_all();
    }
};

} // namespace parallel_task_detail

class parallel_task {
    std::vector<std::thread> workers;
    parallel_task_detail::threadsafe_queue<
        parallel_task_detail::task_function_wrapper> tasks;
    std::atomic<bool> stop_flag{false};

    // Main loop executed by each worker thread.
    // Blocks on the queue and exits only when stop_flag is true and the queue is empty.
    void worker_loop() noexcept {
        while (true) {
            parallel_task_detail::task_function_wrapper task;
            // First, try a non‑blocking pop.
            if (tasks.try_pop(task)) {
                task();
                continue;
            }
            // Queue is empty – block until a new task arrives or stop is requested.
            if (!tasks.wait_and_pop(task, stop_flag))
                break;  // stop flag set and queue empty → exit
            task();
        }
        // After the stop flag is seen, drain any remaining tasks that might have been
        // added concurrently and notify the queue (to avoid lost wake‑ups).
        parallel_task_detail::task_function_wrapper remaining;
        while (tasks.try_pop(remaining)) {
            remaining();
        }
    }

public:
    // Thread pools are not copyable or movable (they manage running threads).
    parallel_task(const parallel_task&) = delete;
    parallel_task& operator=(const parallel_task&) = delete;
    parallel_task(parallel_task&&) = delete;
    parallel_task& operator=(parallel_task&&) = delete;

    // Create a thread pool with exactly `sz` worker threads.
    explicit parallel_task(unsigned int sz) {
        workers.reserve(sz);
        for (unsigned int i = 0; i < sz; ++i)
            workers.emplace_back(&parallel_task::worker_loop, this);
    }

    // Default constructor: use hardware concurrency, or 2 if detection fails.
    parallel_task()
        : parallel_task(std::max(2u, std::thread::hardware_concurrency())) {}

    // Submit a callable with arguments and obtain a future to the result.
    // Throws std::logic_error if the pool has already been stopped.
    template <class Function, class... Args>
    auto submit(Function&& f, Args&&... args)
        -> std::future<std::invoke_result_t<Function, Args...>> {
        using result_type = std::invoke_result_t<Function, Args...>;

        if (stop_flag.load())
            throw std::logic_error("Cannot submit tasks to a stopped pool.");

        // Perfectly forward the callable and its arguments into a mutable lambda.
        auto task = [f = std::forward<Function>(f),
                     ... args = std::forward<Args>(args)]() mutable
                     -> result_type {
            return std::invoke(std::forward<Function>(f),
                               std::forward<Args>(args)...);
        };

        std::packaged_task<result_type()> pt(std::move(task));
        auto future = pt.get_future();
        tasks.push(std::move(pt));
        return future;
    }

    // Request stop and wait for all workers to finish.
    // All previously submitted tasks are guaranteed to be executed.
    void stop_and_wait() noexcept {
        tasks.request_stop(stop_flag);
        for (auto& t : workers) {
            if (t.joinable())
                t.join();
        }
    }

    ~parallel_task() noexcept {
        stop_and_wait();
    }
};

} // namespace sevobench
