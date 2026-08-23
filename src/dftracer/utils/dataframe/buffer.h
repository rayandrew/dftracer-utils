#ifndef DFTRACER_UTILS_DATAFRAME_BUFFER_H
#define DFTRACER_UTILS_DATAFRAME_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <new>

namespace dftracer::utils::dataframe {

/// Aligned, reference-counted byte buffer with a polymorphic owner. Owns memory
/// it allocates (freed on destruction) or wraps foreign memory (a release
/// callback runs instead), so the same type carries engine-allocated buffers
/// and zero-copy-imported Arrow buffers. Default alignment (64) suits SIMD and
/// Arrow.
class Buffer {
   public:
    static constexpr std::size_t DEFAULT_ALIGN = 64;

    static std::shared_ptr<Buffer> allocate(std::size_t bytes,
                                            std::size_t align = DEFAULT_ALIGN) {
        void* p = nullptr;
        if (bytes != 0) {
            std::size_t rounded = (bytes + align - 1) & ~(align - 1);
            p = std::aligned_alloc(align, rounded);
            if (p == nullptr) throw std::bad_alloc();
        }
        return std::shared_ptr<Buffer>(
            new Buffer(static_cast<std::uint8_t*>(p), bytes,
                       [](void* q) { std::free(q); }));
    }

    /// Wrap foreign memory; `release` runs on destruction instead of free.
    static std::shared_ptr<Buffer> wrap(std::uint8_t* data, std::size_t bytes,
                                        std::function<void(void*)> release) {
        return std::shared_ptr<Buffer>(
            new Buffer(data, bytes, std::move(release)));
    }

    ~Buffer() {
        if (release_ && data_ != nullptr) release_(data_);
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    std::uint8_t* data() noexcept { return data_; }
    const std::uint8_t* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

   private:
    Buffer(std::uint8_t* data, std::size_t size,
           std::function<void(void*)> release)
        : data_(data), size_(size), release_(std::move(release)) {}

    std::uint8_t* data_;
    std::size_t size_;
    std::function<void(void*)> release_;
};

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_BUFFER_H
