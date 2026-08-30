#ifndef DFTRACER_UTILS_DATAFRAME_INTERNAL_SPILL_H
#define DFTRACER_UTILS_DATAFRAME_INTERNAL_SPILL_H

#include <dftracer/utils/dataframe/lazyframe.h>  // Cursor, Morsel
#include <dftracer/utils/dataframe/series.h>

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

// On-disk spill for bounded-memory pipeline breakers. Serializes FLAT columns
// to a temp file and reads them back through the same Cursor pull interface, so
// a spilled run and a live input are interchangeable. Same-machine, native
// endianness (temp files never move between hosts).
namespace dftracer::utils::dataframe::spill {

/// Append a Series to `out` (materialized FLAT first). Supports fixed-width,
/// String and Binary columns; throws std::invalid_argument on List/Struct.
void put_series(std::string& out, const Series& s);

/// Read one Series written by put_series, advancing `p` toward `end`.
Series get_series(const std::uint8_t*& p, const std::uint8_t* end);

/// A self-cleaning temp directory holding one query's spill runs.
class Dir {
   public:
    Dir();
    ~Dir();
    Dir(const Dir&) = delete;
    Dir& operator=(const Dir&) = delete;
    /// Path of run file `id` within this directory.
    std::string run_path(int id) const;

   private:
    std::string dir_;
};

/// Appends morsels (columns + row count) to one run file.
class Writer {
   public:
    explicit Writer(const std::string& path);
    void write(const std::vector<Series>& cols, std::int64_t rows);
    void close();

   private:
    std::ofstream os_;
};

/// Reads morsels back from a run file. The max_rows hint is ignored: chunks
/// come back exactly as written.
class Reader : public Cursor {
   public:
    explicit Reader(const std::string& path);
    std::optional<Morsel> next(std::int64_t max_rows) override;

   private:
    std::ifstream is_;
};

}  // namespace dftracer::utils::dataframe::spill

#endif  // DFTRACER_UTILS_DATAFRAME_INTERNAL_SPILL_H
