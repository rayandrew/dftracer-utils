#ifndef DFTRACER_UTILS_PLUGINS_PLUGIN_REGISTER_H
#define DFTRACER_UTILS_PLUGINS_PLUGIN_REGISTER_H

#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/plugin/async.h>
#include <dftracer/utils/plugins/plugin/map.h>
#include <dftracer/utils/plugins/plugin/types.h>

#include <concepts>
#include <coroutine>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace dftracer::utils::plugins {

/// Typed, non-owning view over one dftu_value node; a Slice may keep it since
/// the host-owned config tree outlives it. Default-constructed (or an
/// out-of-range/wrong-kind lookup) yields a null ConfigValue: kind() is
/// ValueKind::Null and every predicate but is_null() is false.
class ConfigValue {
   public:
    ConfigValue() = default;
    explicit ConfigValue(const dftu_value* v) noexcept : v_(v) {}

    ValueKind kind() const noexcept {
        return v_ ? static_cast<ValueKind>(v_->kind) : ValueKind::Null;
    }
    bool is_null() const noexcept { return !v_ || v_->kind == DFTU_VAL_NULL; }
    bool is_bool() const noexcept { return v_ && v_->kind == DFTU_VAL_BOOL; }
    bool is_i64() const noexcept { return v_ && v_->kind == DFTU_VAL_I64; }
    bool is_f64() const noexcept { return v_ && v_->kind == DFTU_VAL_F64; }
    bool is_str() const noexcept { return v_ && v_->kind == DFTU_VAL_STR; }
    bool is_array() const noexcept { return v_ && v_->kind == DFTU_VAL_ARRAY; }
    bool is_object() const noexcept {
        return v_ && v_->kind == DFTU_VAL_OBJECT;
    }

    bool as_bool(bool dflt = false) const noexcept {
        return dftu_as_bool(v_, dflt ? 1 : 0) != 0;
    }
    std::int64_t as_i64(std::int64_t dflt = 0) const noexcept {
        return dftu_as_i64(v_, dflt);
    }
    double as_f64(double dflt = 0.0) const noexcept {
        return dftu_as_f64(v_, dflt);
    }
    std::string_view as_str(std::string_view dflt = {}) const noexcept {
        std::uint32_t n = 0;
        const char* s = dftu_as_str(v_, &n);
        return s ? std::string_view{s, n} : dflt;
    }

    /// ARRAY child count or OBJECT member count; 0 for any other kind.
    std::uint32_t size() const noexcept {
        return (is_array() || is_object()) ? v_->count : 0;
    }
    /// Element `i` of an ARRAY; out of range or not an array yields null.
    ConfigValue operator[](std::uint32_t i) const noexcept {
        return (is_array() && i < v_->count) ? ConfigValue{&v_->as.items[i]}
                                             : ConfigValue{};
    }
    /// Member `key` of an OBJECT; absent or not an object yields null.
    ConfigValue operator[](std::string_view key) const noexcept {
        if (!is_object()) return ConfigValue{};
        for (std::uint32_t i = 0; i < v_->count; ++i) {
            const dftu_member& m = v_->as.members[i];
            if (std::string_view{m.key, m.key_len} == key)
                return ConfigValue{m.value};
        }
        return ConfigValue{};
    }

    const dftu_value* raw() const noexcept { return v_; }

   private:
    const dftu_value* v_ = nullptr;
};

/** View over the config tree; a Slice may keep returned string_views since the
   host-owned tree outlives it. */
class Config {
   public:
    Config() = default;
    explicit Config(const dftu_value* root) : root_(root) {}

    const dftu_value* find(std::string_view key) const {
        if (!root_ || root_->kind != DFTU_VAL_OBJECT) return nullptr;
        for (std::uint32_t i = 0; i < root_->count; ++i) {
            const dftu_member& m = root_->as.members[i];
            if (std::string_view{m.key, m.key_len} == key) return m.value;
        }
        return nullptr;
    }

    std::string_view get(std::string_view key,
                         std::string_view dflt = {}) const {
        const dftu_value* v = find(key);
        return (v && v->kind == DFTU_VAL_STR)
                   ? std::string_view{v->as.str, v->count}
                   : dflt;
    }
    std::int64_t get_int(std::string_view key, std::int64_t dflt = 0) const {
        return dftu_as_i64(find(key), dflt);
    }
    double get_double(std::string_view key, double dflt = 0.0) const {
        return dftu_as_f64(find(key), dflt);
    }
    bool get_bool(std::string_view key, bool dflt = false) const {
        return dftu_as_bool(find(key), dflt ? 1 : 0) != 0;
    }
    Config child(std::string_view key) const {
        const dftu_value* v = find(key);
        return Config{(v && v->kind == DFTU_VAL_OBJECT) ? v : nullptr};
    }

    /// Typed view of the value at `key`, of any kind (including ARRAY/OBJECT);
    /// null if absent. Prefer this over raw() for walking nested config.
    ConfigValue value(std::string_view key) const {
        return ConfigValue{find(key)};
    }
    /// Typed view of the ARRAY at `key`; null if absent or not an array.
    /// Iterate elements with ConfigValue::size()/operator[].
    ConfigValue array(std::string_view key) const {
        return ConfigValue{raw_array(key)};
    }
    /// The ARRAY at `key` coerced to int64 per element (dftu_as_i64 rules);
    /// empty if the key is absent or not an array.
    std::vector<std::int64_t> get_int_array(std::string_view key) const {
        std::vector<std::int64_t> out;
        if (const dftu_value* a = raw_array(key)) {
            out.reserve(a->count);
            for (std::uint32_t i = 0; i < a->count; ++i)
                out.push_back(dftu_as_i64(&a->as.items[i], 0));
        }
        return out;
    }
    /// The ARRAY at `key` coerced to double per element (dftu_as_f64 rules);
    /// empty if the key is absent or not an array.
    std::vector<double> get_double_array(std::string_view key) const {
        std::vector<double> out;
        if (const dftu_value* a = raw_array(key)) {
            out.reserve(a->count);
            for (std::uint32_t i = 0; i < a->count; ++i)
                out.push_back(dftu_as_f64(&a->as.items[i], 0.0));
        }
        return out;
    }
    /// The ARRAY at `key` as string_views (non-STR elements yield an empty
    /// view); the views borrow the host-owned config tree. Empty if the key is
    /// absent or not an array.
    std::vector<std::string_view> get_string_array(std::string_view key) const {
        std::vector<std::string_view> out;
        if (const dftu_value* a = raw_array(key)) {
            out.reserve(a->count);
            for (std::uint32_t i = 0; i < a->count; ++i) {
                const dftu_value& e = a->as.items[i];
                out.push_back(e.kind == DFTU_VAL_STR
                                  ? std::string_view{e.as.str, e.count}
                                  : std::string_view{});
            }
        }
        return out;
    }

    explicit operator bool() const { return root_ != nullptr; }
    const dftu_value* raw() const { return root_; }

   private:
    const dftu_value* raw_array(std::string_view key) const {
        const dftu_value* v = find(key);
        return (v && v->kind == DFTU_VAL_ARRAY) ? v : nullptr;
    }

    const dftu_value* root_ = nullptr;
};

namespace detail {

template <class Slice>
constexpr std::uint32_t slice_needs() {
    if constexpr (requires { Slice::needs; })
        return Slice::needs;
    else
        return 0;
}

template <class Slice>
struct Holder {
    dftu_plugin vt{};
    Config config;
    std::string plan; /**< backs plan_query's const char* */
};

template <class Slice>
Holder<Slice>* holder_of(void* self) {
    return static_cast<Holder<Slice>*>(self);
}

/// A Slice whose async on_batch takes the ergonomic Batch view.
template <class Slice>
concept AsyncBatchView = requires(Slice& s, const Batch& b, Host h) {
    { s.on_batch(b, h) } -> std::same_as<Task>;
};

/// A Slice whose async on_batch takes the raw C dftu_batch (legacy signature).
template <class Slice>
concept AsyncBatch = requires(Slice& s, const dftu_batch& b, Host h) {
    { s.on_batch(b, h) } -> std::same_as<Task>;
};

/// A Slice whose synchronous step takes the ergonomic Batch view.
template <class Slice>
concept BatchViewStep =
    requires(Slice& s, const Batch& b, Host h) { s.step(b, h); };

/// A Slice whose synchronous step takes a column batch. Such a fold is wired to
/// the vectorized seam (dftu_plugin::on_batch_columns) and never sees rows; it
/// is the shape that feeds Host::agg, whose accumulator takes column batches.
template <class Slice>
concept ColumnStep =
    requires(Slice& s, const dftu_dataframe* df, Host h) { s.step(df, h); };

template <class Slice>
concept AsyncFinalize = requires(Slice& s, Host h) {
    { s.on_finalize(h) } -> std::same_as<Task>;
};

/// Resume the driven coroutine one step; NULL once it has run to completion.
inline dftu_task* step_thunk(void* coro) {
    auto h = std::coroutine_handle<Task::promise_type>::from_address(coro);
    h.resume();
    if (!h.done()) return h.promise().pending;
    if (h.promise().exc) {
        if (const dftu_host* host = h.promise().host) {
            try {
                std::rethrow_exception(h.promise().exc);
            } catch (const std::exception& e) {
                Host{host}.log(DFTU_LOG_ERROR, e.what());
            } catch (...) {
                Host{host}.log(DFTU_LOG_ERROR, "plugin coroutine threw");
            }
        }
    }
    h.destroy();
    return nullptr;
}

template <class Slice, class Coro>
dftu_task* drive_coro(const dftu_host* host, Coro&& coro) {
    Task t = std::forward<Coro>(coro);
    auto h = t.release();
    h.promise().host = host;
    const dftu_ext_coro* c =
        host->get_extension ? static_cast<const dftu_ext_coro*>(
                                  host->get_extension(host->h, DFTU_EXT_CORO))
                            : nullptr;
    return c && c->drive ? c->drive(host->h, &step_thunk, h.address())
                         : nullptr;
}

/// A Slice naming the ports and accumulators it produces via a static
/// `provides()` returning a range of const char*.
template <class Slice>
concept DeclaresProvides = requires {
    { std::begin(Slice::provides()) };
    { std::end(Slice::provides()) };
};

/// A Slice naming the ports and accumulators it reads via a static
/// `consumes()` returning a range of const char*.
template <class Slice>
concept DeclaresConsumes = requires {
    { std::begin(Slice::consumes()) };
    { std::end(Slice::consumes()) };
};

// The declared range copied into a NULL-terminated array with static storage,
// which is the lifetime dftu_plugin::provides/consumes require.
template <class Slice, bool Produced>
const char* const* name_list_thunk(void*) {
    static const std::vector<const char*> names = [] {
        std::vector<const char*> v;
        if constexpr (Produced) {
            for (const char* n : Slice::provides()) v.push_back(n);
        } else {
            for (const char* n : Slice::consumes()) v.push_back(n);
        }
        v.push_back(nullptr);
        return v;
    }();
    return names.data();
}

}  // namespace detail

/** Build a dftu_plugin from a Slice providing Slice(const Config&), merge, and
   either sync step/finalize or a Task-returning on_batch/on_finalize coroutine,
   plus optionally `static constexpr uint32_t needs`. To take part in the fold
   ordering a Slice may also declare `static ... provides()` and `static ...
   consumes()`, each a range of const char* port/accumulator names outliving the
   plugin; make_plugin wires them to dftu_plugin::provides / ::consumes.
   Exceptions must not escape the ABI boundary, so every callback catches. */
template <class Slice>
dftu_plugin* make_plugin(const dftu_value* config) {
    auto* hd = new detail::Holder<Slice>();
    hd->config = Config(config);
    hd->plan = std::string(hd->config.get("query"));

    dftu_plugin& vt = hd->vt;
    vt.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    vt.self = hd;

    vt.needs = [](void*) -> std::uint32_t {
        return detail::slice_needs<Slice>();
    };

    vt.plan_query = [](void* self) -> const char* {
        auto* h = detail::holder_of<Slice>(self);
        return h->plan.empty() ? nullptr : h->plan.c_str();
    };

    vt.make_slice = [](void* self) -> void* {
        try {
            return new Slice(detail::holder_of<Slice>(self)->config);
        } catch (...) {
            return nullptr;
        }
    };

    // A plugin fills exactly one of the two batch seams; leave the other null.
    if constexpr (detail::ColumnStep<Slice>) {
        vt.on_batch_columns = [](void* slice, const dftu_dataframe* df,
                                 const dftu_host* host) -> dftu_task* {
            try {
                static_cast<Slice*>(slice)->step(df, Host{host});
            } catch (const std::exception& e) {
                Host{host}.log(DFTU_LOG_ERROR, e.what());
            } catch (...) {
                Host{host}.log(DFTU_LOG_ERROR, "plugin step threw");
            }
            return nullptr;
        };
    } else {
        vt.on_batch = [](void* slice, const dftu_batch* b,
                         const dftu_host* host) -> dftu_task* {
            try {
                Slice* sl = static_cast<Slice*>(slice);
                Host h{host};
                if constexpr (detail::AsyncBatchView<Slice>) {
                    return detail::drive_coro<Slice>(
                        host, sl->on_batch(Batch{*b}, h));
                } else if constexpr (detail::AsyncBatch<Slice>) {
                    return detail::drive_coro<Slice>(host, sl->on_batch(*b, h));
                } else if constexpr (detail::BatchViewStep<Slice>) {
                    sl->step(Batch{*b}, h);
                } else {
                    sl->step(*b, h);
                }
            } catch (const std::exception& e) {
                Host{host}.log(DFTU_LOG_ERROR, e.what());
            } catch (...) {
                Host{host}.log(DFTU_LOG_ERROR, "plugin step threw");
            }
            return nullptr;
        };
    }

    vt.merge = [](void* into, void* other) {
        try {
            static_cast<Slice*>(into)->merge(*static_cast<Slice*>(other));
        } catch (...) {
        }
    };

    vt.on_finalize = [](void* slice, const dftu_host* host) -> dftu_task* {
        try {
            if constexpr (detail::AsyncFinalize<Slice>) {
                return detail::drive_coro<Slice>(
                    host, static_cast<Slice*>(slice)->on_finalize(Host{host}));
            } else {
                static_cast<Slice*>(slice)->finalize(Host{host});
            }
        } catch (const std::exception& e) {
            Host{host}.log(DFTU_LOG_ERROR, e.what());
        } catch (...) {
            Host{host}.log(DFTU_LOG_ERROR, "plugin finalize threw");
        }
        return nullptr;
    };

    vt.destroy_slice = [](void* slice) { delete static_cast<Slice*>(slice); };
    vt.destroy = [](void* self) { delete detail::holder_of<Slice>(self); };

    if constexpr (detail::DeclaresProvides<Slice>)
        vt.provides = &detail::name_list_thunk<Slice, true>;
    if constexpr (detail::DeclaresConsumes<Slice>)
        vt.consumes = &detail::name_list_thunk<Slice, false>;

    return &vt;
}

}  // namespace dftracer::utils::plugins

#endif /* DFTRACER_UTILS_PLUGINS_PLUGIN_REGISTER_H */
