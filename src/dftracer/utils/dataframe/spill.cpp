#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/spill.h>
#include <dftracer/utils/dataframe/types.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace dftracer::utils::dataframe::spill {

namespace {

namespace fs = std::filesystem;

template <class T>
void put_pod(std::string& out, T v) {
    out.append(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
T get_pod(const std::uint8_t*& p, const std::uint8_t* end) {
    if (p + sizeof(T) > end) throw std::runtime_error("spill: truncated read");
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}

void put_bytes(std::string& out, const void* data, std::size_t n) {
    out.append(static_cast<const char*>(data), n);
}

const std::uint8_t* take_bytes(const std::uint8_t*& p, const std::uint8_t* end,
                               std::size_t n) {
    if (p + n > end) throw std::runtime_error("spill: truncated read");
    const std::uint8_t* start = p;
    p += n;
    return start;
}

// Validity bitmap (Arrow layout, 1 = valid) built from per-row null flags.
std::vector<std::uint8_t> validity_bitmap(const Series& s) {
    const std::int64_t n = s.length();
    std::vector<std::uint8_t> bits(static_cast<std::size_t>((n + 7) / 8), 0);
    for (std::int64_t i = 0; i < n; ++i)
        if (!s.is_null(i))
            bits[static_cast<std::size_t>(i >> 3)] |=
                static_cast<std::uint8_t>(1u << (i & 7));
    return bits;
}

// Atomic sequence so many spilling sinks in one process pick distinct dirs.
int next_seq() {
    static std::atomic<int> seq{0};
    return seq.fetch_add(1);
}

}  // namespace

void put_series(std::string& out, const Series& s_in) {
    Series s =
        s_in.encoding() == Encoding::Flat ? s_in.share() : s_in.materialize();
    const TypeId t = s.type();
    const std::int64_t n = s.length();
    const std::int64_t nulls = s.null_count();
    put_pod<std::int32_t>(out, static_cast<std::int32_t>(t));
    put_pod<std::int64_t>(out, n);
    put_pod<std::int64_t>(out, nulls);
    if (nulls > 0) {
        std::vector<std::uint8_t> v = validity_bitmap(s);
        put_bytes(out, v.data(), v.size());
    }
    if (t == TypeId::List || t == TypeId::Struct)
        throw std::invalid_argument("spill: nested columns unsupported");
    if (byte_width(t) == 0) {  // String / Binary
        const std::int32_t* offs = dftu_series_offsets(s.handle());
        const std::int64_t nbytes = n > 0 ? offs[n] : 0;
        put_pod<std::int64_t>(out, nbytes);
        put_bytes(out, offs,
                  static_cast<std::size_t>(n + 1) * sizeof(std::int32_t));
        put_bytes(out, dftu_series_data(s.handle()),
                  static_cast<std::size_t>(nbytes));
    } else {
        const std::size_t bytes = buffer_bytes(t, n);
        put_bytes(out, dftu_series_data(s.handle()), bytes);
    }
}

Series get_series(const std::uint8_t*& p, const std::uint8_t* end) {
    const auto type = static_cast<TypeId>(get_pod<std::int32_t>(p, end));
    const std::int64_t n = get_pod<std::int64_t>(p, end);
    const std::int64_t nulls = get_pod<std::int64_t>(p, end);
    const std::uint8_t* validity = nullptr;
    if (nulls > 0)
        validity = take_bytes(p, end, static_cast<std::size_t>((n + 7) / 8));
    const auto dt = static_cast<dftu_dtype>(static_cast<std::int32_t>(type));
    if (byte_width(type) == 0) {  // String / Binary
        const std::int64_t nbytes = get_pod<std::int64_t>(p, end);
        const auto* offs = reinterpret_cast<const std::int32_t*>(take_bytes(
            p, end, static_cast<std::size_t>(n + 1) * sizeof(std::int32_t)));
        const std::uint8_t* data =
            take_bytes(p, end, static_cast<std::size_t>(nbytes));
        return Series{dftu_series_new_string(dt, offs, data, n, validity)};
    }
    const std::size_t bytes = buffer_bytes(type, n);
    const std::uint8_t* data = take_bytes(p, end, bytes);
    return Series{dftu_series_new_flat(dt, data, n, validity)};
}

Dir::Dir() {
    fs::path base =
        fs::temp_directory_path() / ("dftu_lazy_" + std::to_string(::getpid()) +
                                     "_" + std::to_string(next_seq()));
    std::error_code ec;
    fs::create_directories(base, ec);
    dir_ = base.string();
}

Dir::~Dir() {
    std::error_code ec;
    fs::remove_all(dir_, ec);
}

std::string Dir::run_path(int id) const {
    return (fs::path(dir_) / ("run_" + std::to_string(id) + ".bin")).string();
}

Writer::Writer(const std::string& path)
    : os_(path, std::ios::binary | std::ios::trunc) {
    if (!os_) throw std::runtime_error("spill: cannot open run file " + path);
}

void Writer::write(const std::vector<Series>& cols, std::int64_t rows) {
    std::string blob;
    put_pod<std::int64_t>(blob, rows);
    put_pod<std::int32_t>(blob, static_cast<std::int32_t>(cols.size()));
    for (const Series& c : cols) put_series(blob, c);
    const std::int64_t len = static_cast<std::int64_t>(blob.size());
    os_.write(reinterpret_cast<const char*>(&len), sizeof(len));
    os_.write(blob.data(), static_cast<std::streamsize>(blob.size()));
}

void Writer::close() { os_.close(); }

Reader::Reader(const std::string& path) : is_(path, std::ios::binary) {
    if (!is_) throw std::runtime_error("spill: cannot open run file " + path);
}

std::optional<Morsel> Reader::next(std::int64_t /*max_rows*/) {
    std::int64_t len = 0;
    is_.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!is_ || is_.gcount() == 0) return std::nullopt;
    std::string blob(static_cast<std::size_t>(len), '\0');
    is_.read(blob.data(), static_cast<std::streamsize>(len));
    if (is_.gcount() != len) throw std::runtime_error("spill: short run read");
    const auto* p = reinterpret_cast<const std::uint8_t*>(blob.data());
    const std::uint8_t* pend = p + blob.size();
    Morsel m;
    m.rows = get_pod<std::int64_t>(p, pend);
    const std::int32_t ncols = get_pod<std::int32_t>(p, pend);
    m.columns.reserve(static_cast<std::size_t>(ncols));
    for (std::int32_t c = 0; c < ncols; ++c)
        m.columns.push_back(get_series(p, pend));
    return m;
}

}  // namespace dftracer::utils::dataframe::spill
