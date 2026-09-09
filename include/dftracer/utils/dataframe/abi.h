#ifndef DFTRACER_UTILS_DATAFRAME_ABI_H
#define DFTRACER_UTILS_DATAFRAME_ABI_H

#include <dftracer/utils/core/common/export.h>
#include <dftracer/utils/query/abi.h>
#include <stdint.h>

/*
 * Stable C ABI for the dataframe columnar engine. The handle is opaque; its
 * layout is defined only in the implementation. Type codes match
 * dataframe::TypeId ordinals, encoding codes match dataframe::Encoding
 * ordinals. Every returned column is owned by the caller and freed with
 * dftu_series_free. NULL is returned on an unsupported type or a failed
 * operation.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dftu_series dftu_series;

/** The domain a dftu_scalar carries (its `kind`). */
typedef enum {
    DFTU_SCALAR_TAG_I64 = 0,
    DFTU_SCALAR_TAG_U64 = 1,
    DFTU_SCALAR_TAG_F64 = 2,
    DFTU_SCALAR_TAG_STR = 3
} dftu_scalar_tag;

/** A scalar operand, tagged by its domain so a kernel converts it to the
 * column's element type. Pass a signed integer as I64, an unsigned integer as
 * U64, a float as F64, a string as STR (see dftu_scalar_tag).
 *
 * A STR scalar BORROWS `value.s` for the duration of the call: nothing stores a
 * dftu_scalar past the call it was passed to, and the owner of the text (an
 * ExprNode, a caller's std::string) outlives it. `len` sits in the padding
 * after `kind`, so the struct is 16 bytes and 8-aligned exactly as before and
 * no by-value call site changes. */
typedef struct dftu_scalar {
    int32_t kind;
    uint32_t len; /**< STR only: byte length of `value.s`; 0 otherwise */
    union {
        int64_t i;
        uint64_t u;
        double d;
        const char* s;
    } value;
} dftu_scalar;

/** Element type of a dftu_series (mirrors dataframe::TypeId ordinals). */
typedef enum {
    DFTU_TYPE_BOOL = 0,
    DFTU_TYPE_INT8 = 1,
    DFTU_TYPE_INT16 = 2,
    DFTU_TYPE_INT32 = 3,
    DFTU_TYPE_INT64 = 4,
    DFTU_TYPE_UINT8 = 5,
    DFTU_TYPE_UINT16 = 6,
    DFTU_TYPE_UINT32 = 7,
    DFTU_TYPE_UINT64 = 8,
    DFTU_TYPE_FLOAT32 = 9,
    DFTU_TYPE_FLOAT64 = 10,
    DFTU_TYPE_STRING = 11,
    DFTU_TYPE_BINARY = 12
} dftu_dtype;

/** Create a FLAT column copying `n` values of `type` from `data`. `validity` is
 * an Arrow-layout bitmap (1 = valid) or NULL for no nulls. */
DFTU_EXPORT dftu_series* dftu_series_new_flat(dftu_dtype type, const void* data,
                                              int64_t n,
                                              const uint8_t* validity);

/** Create a FLAT column BORROWING `n` values of `type` from `data` (no copy):
 * the column's data buffer points straight at `data`. `data` must stay valid
 * until `release(release_ctx)` runs, which happens when the column's last
 * reference is dropped; pass NULL for `release` to borrow non-owned static
 * memory. `validity` is copied if non-NULL (usually NULL). Returns NULL on a
 * variable-width type (String/Binary), where borrowing a single flat buffer is
 * not meaningful. */
DFTU_EXPORT dftu_series* dftu_series_new_flat_borrowed(
    dftu_dtype type, const void* data, int64_t n, const uint8_t* validity,
    void (*release)(void* ctx), void* release_ctx);

/** Create a FLAT variable-width column (String or Binary) copying `n` values.
 * `offsets` has n+1 int32 entries into `data`; `data` holds offsets[n] bytes.
 * `validity` is an Arrow-layout bitmap (1 = valid) or NULL for no nulls. */
DFTU_EXPORT dftu_series* dftu_series_new_string(dftu_dtype type,
                                                const int32_t* offsets,
                                                const void* data, int64_t n,
                                                const uint8_t* validity);

/** Create a STRUCT column from `n_fields` child columns (each the same length),
 * named by `names`. Takes ownership of the child columns. */
DFTU_EXPORT dftu_series* dftu_series_new_struct(dftu_series** fields,
                                                const char** names,
                                                int32_t n_fields);

/** Create a LIST column: `offsets` (n+1 int32 entries) index into `values` (the
 * flattened child column). Takes ownership of `values`. */
DFTU_EXPORT dftu_series* dftu_series_new_list(const int32_t* offsets, int64_t n,
                                              dftu_series* values);

DFTU_EXPORT void dftu_series_free(dftu_series* col);

DFTU_EXPORT int32_t dftu_series_type(const dftu_series* col);
DFTU_EXPORT int32_t dftu_series_encoding(const dftu_series* col);
DFTU_EXPORT int64_t dftu_series_length(const dftu_series* col);
DFTU_EXPORT int64_t dftu_series_null_count(const dftu_series* col);

/** Whether row `i` is null (0/1). A column with no validity bitmap is
 * all-valid.
 */
DFTU_EXPORT int32_t dftu_series_is_null(const dftu_series* col, int64_t i);

/** Raw FLAT value buffer (or the byte data of a String/Binary column), or NULL
 * when the column is not FLAT. */
DFTU_EXPORT const void* dftu_series_data(const dftu_series* col);

/** int32 offset buffer (length+1 entries) of a variable-width String/Binary or
 * List column, or NULL for fixed-width types. */
DFTU_EXPORT const int32_t* dftu_series_offsets(const dftu_series* col);

/** Number of child columns: 1 for a List (its flattened values), the field
 * count for a Struct, 0 otherwise. */
DFTU_EXPORT int32_t dftu_series_num_children(const dftu_series* col);

/** Child column `i` (List: only 0, the values; Struct: field i) as a new owned
 * column sharing the child's buffers. NULL if out of range. */
DFTU_EXPORT dftu_series* dftu_series_child(const dftu_series* col, int32_t i);

/** A new owned column sharing `col`'s buffers zero-copy (a reference-count
 * bump, no data copy). NULL if `col` is NULL. Underpins projection/rename frame
 * ops.
 */
DFTU_EXPORT dftu_series* dftu_series_share(const dftu_series* col);

/** A zero-copy view of the FLAT rows [offset, offset+len) of `col`: the
 * result's data pointer is shifted into `col`'s buffer and keeps it alive, so
 * SIMD kernels run on the sub-range with no gather. FLAT, no-null, fixed-width
 * only (the chunked columnar evaluator's inputs); NULL otherwise. */
DFTU_EXPORT dftu_series* dftu_series_slice(const dftu_series* col,
                                           int64_t offset, int64_t len);

/** Elementwise binary arithmetic (same numeric type, same length). */
DFTU_EXPORT dftu_series* dftu_series_add(const dftu_series* a,
                                         const dftu_series* b);
DFTU_EXPORT dftu_series* dftu_series_sub(const dftu_series* a,
                                         const dftu_series* b);
DFTU_EXPORT dftu_series* dftu_series_mul(const dftu_series* a,
                                         const dftu_series* b);
/** Elementwise a / b (floats SIMD; integers scalar with a zero guard). */
DFTU_EXPORT dftu_series* dftu_series_div(const dftu_series* a,
                                         const dftu_series* b);

/** Broadcast a scalar across a FLAT numeric column. The scalar is converted to
 * the column's element type, so 64-bit integers stay exact. */
DFTU_EXPORT dftu_series* dftu_series_add_scalar(const dftu_series* a,
                                                dftu_scalar s);
DFTU_EXPORT dftu_series* dftu_series_sub_scalar(const dftu_series* a,
                                                dftu_scalar s);
DFTU_EXPORT dftu_series* dftu_series_mul_scalar(const dftu_series* a,
                                                dftu_scalar s);
DFTU_EXPORT dftu_series* dftu_series_div_scalar(const dftu_series* a,
                                                dftu_scalar s);

/** Comparison ops for dftu_series_compare. */
typedef enum {
    DFTU_CMP_GT = 0,
    DFTU_CMP_GE = 1,
    DFTU_CMP_LT = 2,
    DFTU_CMP_LE = 3,
    DFTU_CMP_EQ = 4,
    DFTU_CMP_NE = 5
} dftu_cmp_op;

/** Compare each numeric value of `v` against `rhs` (op is a dftu_cmp_op
 * code); returns a Bool column (bit-packed), carrying v's validity. The scalar
 * is converted to v's element type, so a 64-bit integer compares exactly. */
DFTU_EXPORT dftu_series* dftu_series_compare(const dftu_series* v,
                                             dftu_cmp_op op, dftu_scalar rhs);

/** Rows where the numeric value is greater than `threshold`, as a SELECTION
 * column over `v` (shares v's buffers). */
DFTU_EXPORT dftu_series* dftu_series_filter_gt(const dftu_series* v,
                                               dftu_scalar threshold);

/** Rows of `v` where the bit-packed Bool `mask` (same length) is true, as a
 * SELECTION column over `v`. */
DFTU_EXPORT dftu_series* dftu_series_filter(const dftu_series* v,
                                            const dftu_series* mask);

/** Convert a FLAT numeric column to the numeric type `target`. */
DFTU_EXPORT dftu_series* dftu_series_cast(const dftu_series* v,
                                          dftu_dtype target);

/** Unary numeric primitives for dftu_series_prim (mirrors dataframe::PrimOp).
 */
typedef enum {
    DFTU_PRIM_ILOG2 = 0,
    DFTU_PRIM_BIT_WIDTH = 1,
    DFTU_PRIM_POPCOUNT = 2,
    DFTU_PRIM_CLZ = 3,
    DFTU_PRIM_CTZ = 4,
    DFTU_PRIM_MIX64 = 5
} dftu_prim_op;

/** Unary numeric primitive over a FLAT Int64/Uint64 column, producing an Int64
 * column (`op` is a dftu_prim_op). Null if the input is not a 64-bit integer
 * column. */
DFTU_EXPORT dftu_series* dftu_series_prim(const dftu_series* v,
                                          dftu_prim_op op);

/** Boolean ops on bit-packed Bool columns for dftu_series_logical. */
typedef enum { DFTU_LOGICAL_AND = 0, DFTU_LOGICAL_OR = 1 } dftu_logical_op;
DFTU_EXPORT dftu_series* dftu_series_logical(const dftu_series* a,
                                             const dftu_series* b,
                                             dftu_logical_op op);
DFTU_EXPORT dftu_series* dftu_series_logical_not(const dftu_series* a);

/** Dictionary-encode a FLAT String/Binary column: dedup values into a
 * dictionary child, returning a DICTIONARY column of int32 codes. */
DFTU_EXPORT dftu_series* dftu_series_dictionary_encode(const dftu_series* v);

/** String predicates over a String/Binary column (FLAT or DICTIONARY), each
 * returning a Bool column carrying v's validity. A DICTIONARY input evaluates
 * the predicate once per dictionary entry, then maps codes (predicate
 * pushdown), so cost is O(dict + n) rather than O(n * len). */
DFTU_EXPORT dftu_series* dftu_series_str_eq(const dftu_series* v,
                                            const char* rhs, int32_t rhs_len);
DFTU_EXPORT dftu_series* dftu_series_str_contains(const dftu_series* v,
                                                  const char* needle,
                                                  int32_t needle_len);
DFTU_EXPORT dftu_series* dftu_series_str_starts_with(const dftu_series* v,
                                                     const char* prefix,
                                                     int32_t prefix_len);
/** True where the string ends with the literal `suffix`. */
DFTU_EXPORT dftu_series* dftu_series_str_ends_with(const dftu_series* v,
                                                   const char* suffix,
                                                   int32_t suffix_len);
/** True where the WHOLE string matches the ECMAScript regex `pattern` (the
 * regex entry point; std::regex). NULL if the pattern fails to compile. */
DFTU_EXPORT dftu_series* dftu_series_str_matches(const dftu_series* v,
                                                 const char* pattern,
                                                 int32_t pattern_len);
/** True where the string matches the SQL LIKE / glob `pattern`: `%` matches any
 * run (including empty), `_` matches exactly one character, `\` escapes a
 * literal `%`/`_`/`\`; other bytes are literal. A Bool column carrying v's
 * validity. Affix patterns (`text%`, `%text`, `%text%`, exact) take a fast
 * equality / affix / SIMD-substring route; interior wildcards use a per-row
 * two-pointer matcher. */
DFTU_EXPORT dftu_series* dftu_series_str_like(const dftu_series* v,
                                              const char* pattern,
                                              int32_t pattern_len);

/** Per-row byte length (offsets[i+1]-offsets[i]) as an Int64 column, nulls
 * preserved. FLAT input is vectorized (Highway); DICTIONARY is scalar. */
DFTU_EXPORT dftu_series* dftu_series_str_len_bytes(const dftu_series* v);
/** Per-row UTF-8 codepoint count (non-continuation bytes) as an Int64 column,
 * nulls preserved. */
DFTU_EXPORT dftu_series* dftu_series_str_len_chars(const dftu_series* v);
/** Byte index of the first occurrence of literal `needle`, or -1, per row, as
 * an Int64 column. */
DFTU_EXPORT dftu_series* dftu_series_str_find(const dftu_series* v,
                                              const char* needle,
                                              int32_t needle_len);

/** ASCII case fold ('A'-'Z' <-> 'a'-'z'); non-ASCII bytes pass through
 * unchanged. Returns a String column, nulls preserved. FLAT input folds the
 * byte buffer with Highway. */
/** FNV-1a 64-bit hash of each row's bytes as a UInt64 column; null rows stay
 * null. Matches the host's own fnv1a over the same bytes, so a plugin can
 * derive the trace's fhash/hhash columns itself. */
DFTU_EXPORT dftu_series* dftu_series_fnv1a(const dftu_series* v);

/** Parse each row as dftracer's 16-lowercase-hex-digit 64-bit hash form into a
 * UInt64 column; a null row, or one not in that exact form, is null. */
DFTU_EXPORT dftu_series* dftu_series_hex64_parse(const dftu_series* v);

/** Format each row of a UInt64/Int64 column back into dftracer's
 * 16-lowercase-hex-digit form as a String column; null rows stay null. The
 * inverse of dftu_series_hex64_parse. Returns NULL on any other input type. */
DFTU_EXPORT dftu_series* dftu_series_hex64_format(const dftu_series* v);

DFTU_EXPORT dftu_series* dftu_series_to_lowercase(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_to_uppercase(const dftu_series* v);
/** Trim ASCII whitespace from both/left/right ends. String column, nulls
 * preserved. */
DFTU_EXPORT dftu_series* dftu_series_str_strip(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_str_lstrip(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_str_rstrip(const dftu_series* v);
/** Replace the first / all literal occurrence(s) of `pat` with `repl` per row.
 * An empty `pat` leaves the row unchanged. String column, nulls preserved. */
DFTU_EXPORT dftu_series* dftu_series_str_replace(const dftu_series* v,
                                                 const char* pat,
                                                 int32_t pat_len,
                                                 const char* repl,
                                                 int32_t repl_len);
DFTU_EXPORT dftu_series* dftu_series_str_replace_all(const dftu_series* v,
                                                     const char* pat,
                                                     int32_t pat_len,
                                                     const char* repl,
                                                     int32_t repl_len);
/** Substring by BYTE offset. `start` may be negative (from the end); a negative
 * `length` means "to the end". Bounds are clamped. String column, nulls
 * preserved. */
DFTU_EXPORT dftu_series* dftu_series_str_slice(const dftu_series* v,
                                               int64_t start, int64_t length);
/** Left/right pad to `width` bytes with `fill` (rows already >= width are
 * unchanged). String column, nulls preserved. */
DFTU_EXPORT dftu_series* dftu_series_str_pad_start(const dftu_series* v,
                                                   int64_t width, char fill);
DFTU_EXPORT dftu_series* dftu_series_str_pad_end(const dftu_series* v,
                                                 int64_t width, char fill);
/** Left-pad with '0' to `width` bytes, after an optional leading sign (Python
 * zfill). String column, nulls preserved. */
DFTU_EXPORT dftu_series* dftu_series_str_zfill(const dftu_series* v,
                                               int64_t width);
/** Split each row on the literal `sep` into a List<String> column, nulls
 * preserved (a null input row yields a null list). An empty `sep` yields a
 * single element (the whole string). */
DFTU_EXPORT dftu_series* dftu_series_str_split(const dftu_series* v,
                                               const char* sep,
                                               int32_t sep_len);

/* Reduce a FLAT numeric column to a scalar (op is a dftu_reduce_op code),
 * skipping nulls. SUM accumulates in the widest type of the column's domain
 * (i64/u64/f64) so it does not overflow narrow types or lose integer precision;
 * MIN/MAX return a value in the column's domain. */
/** Aggregation ops. A bitmask (power-of-two values) so group_by can compute
 * several in one pass; dftu_series_reduce takes a single flag (SUM/MIN/MAX). */
typedef enum {
    DFTU_REDUCE_SUM = 1,
    DFTU_REDUCE_MIN = 2,
    DFTU_REDUCE_MAX = 4,
    DFTU_REDUCE_COUNT = 8,
    DFTU_REDUCE_MEAN = 16
} dftu_reduce_op;
DFTU_EXPORT dftu_scalar dftu_series_reduce(const dftu_series* v,
                                           dftu_reduce_op op);

/** Count of non-null rows. */
DFTU_EXPORT int64_t dftu_series_count(const dftu_series* v);

/** Product of the non-null values (accumulated in the column's wide domain,
 * like SUM). Read with scalar_value<T>(). */
DFTU_EXPORT dftu_scalar dftu_series_product(const dftu_series* v);

/** Boolean reductions over a bit-packed Bool column, skipping nulls: `all` is
 * 1 when every non-null bit is set (1 for an all-null/empty column), `any` is
 * 1 when at least one non-null bit is set. */
DFTU_EXPORT int32_t dftu_series_all(const dftu_series* v);
DFTU_EXPORT int32_t dftu_series_any(const dftu_series* v);

/** Row index of the minimum / maximum non-null value (first on ties), or -1
 * when the column has no non-null rows. */
DFTU_EXPORT int64_t dftu_series_arg_min(const dftu_series* v);
DFTU_EXPORT int64_t dftu_series_arg_max(const dftu_series* v);

/** Most frequent non-null value (hash-based; first-seen wins on ties). Read
 * with scalar_value<T>(). */
DFTU_EXPORT dftu_scalar dftu_series_mode(const dftu_series* v);

/** Group `values` by `keys` (equal length) in one pass. `op_mask` is a
 * bitwise-or of DFTU_REDUCE_* flags. Sets `*out_keys` to the distinct keys
 * (first-seen order) and fills `out_values[]` with one aggregate column per set
 * flag, in flag order (sum, min, max, count, mean), up to `max_values`; returns
 * the number written. The caller owns every returned column. SUM keeps the
 * value's wide domain type, COUNT is int64, MEAN is float64, MIN/MAX keep the
 * value type. */
DFTU_EXPORT int32_t dftu_dataframe_group_by(
    const dftu_series* keys, const dftu_series* values, int32_t op_mask,
    dftu_series** out_keys, dftu_series** out_values, int32_t max_values);

/** Materialize any encoding to a new FLAT column (gather). */
DFTU_EXPORT dftu_series* dftu_series_materialize(const dftu_series* v);

/** Gather `n` rows of `v` at row indices `idx` into a new FLAT column. Handles
 * every column type (nested List/Struct included) and propagates validity. */
DFTU_EXPORT dftu_series* dftu_series_take(const dftu_series* v,
                                          const int64_t* idx, int64_t n);

/** Stable argsort: an Int64 column of row indices ordering `v` ascending, or
 * descending when `descending` is nonzero. Null rows sort last in both
 * directions. The caller owns the returned column. */
DFTU_EXPORT dftu_series* dftu_series_argsort(const dftu_series* v,
                                             int32_t descending);

/** Summary statistics over the non-null values of a FLAT column. Moment stats
 * are numeric-only; quantile sorts (SIMD); nunique/unique work for any type. */
DFTU_EXPORT double dftu_series_variance(const dftu_series* v, int32_t sample);
DFTU_EXPORT double dftu_series_stddev(const dftu_series* v, int32_t sample);
DFTU_EXPORT double dftu_series_skewness(const dftu_series* v);
DFTU_EXPORT double dftu_series_kurtosis(const dftu_series* v);
DFTU_EXPORT double dftu_series_quantile(const dftu_series* v, double q);
DFTU_EXPORT int64_t dftu_series_nunique(const dftu_series* v);
/** The distinct non-null values, ascending. Caller owns the column. */
DFTU_EXPORT dftu_series* dftu_series_unique(const dftu_series* v);

/** Elementwise transforms -> a new FLAT column (caller owns). abs/clip/round
 * are SIMD; fillna clears nulls; cumsum is a running sum. */
DFTU_EXPORT dftu_series* dftu_series_abs(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_clip(const dftu_series* v, dftu_scalar lo,
                                          dftu_scalar hi);
DFTU_EXPORT dftu_series* dftu_series_round(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_fillna(const dftu_series* v,
                                            dftu_scalar fill);
DFTU_EXPORT dftu_series* dftu_series_cumsum(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_cummax(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_cummin(const dftu_series* v);
/** Running product (nulls contribute 1), same element type. */
DFTU_EXPORT dftu_series* dftu_series_cum_prod(const dftu_series* v);
/** Running count of non-null rows, as an Int64 column. */
DFTU_EXPORT dftu_series* dftu_series_cum_count(const dftu_series* v);

/** ceil/floor/trunc round floats toward +inf / -inf / zero (integers
 * unchanged); sign yields -1/0/1 in the column's type; negate is unary minus.
 * All SIMD. */
DFTU_EXPORT dftu_series* dftu_series_ceil(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_floor(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_trunc(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_sign(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_negate(const dftu_series* v);

/** First discrete difference x[i]-x[i-1] (same type) and percent change
 * (x[i]-x[i-1])/x[i-1] (Float64); row 0 is null in both. */
DFTU_EXPORT dftu_series* dftu_series_diff(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_pct_change(const dftu_series* v);

/** sqrt/exp/log over the values, as a Float64 column (sqrt is SIMD). */
DFTU_EXPORT dftu_series* dftu_series_sqrt(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_exp(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_log(const dftu_series* v);

/** Predicate masks over a FLAT column, each a bit-packed Bool column (no
 * nulls). is_nan/is_finite/is_infinite test the IEEE class of a float column
 * (Highway lanes); an integer column is all-finite (never NaN/Inf), so the mask
 * is filled accordingly. NULL for a non-numeric column. */
DFTU_EXPORT dftu_series* dftu_series_is_nan(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_is_finite(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_is_infinite(const dftu_series* v);

/** is_unique[i] is true iff v's value at row i occurs exactly once;
 * is_duplicated is its complement (true for a repeated value). Both are
 * bit-packed Bool columns (hash pass; polars semantics). NULL rows form one
 * group. */
DFTU_EXPORT dftu_series* dftu_series_is_unique(const dftu_series* v);
DFTU_EXPORT dftu_series* dftu_series_is_duplicated(const dftu_series* v);

/** Whether `v` is sorted ascending (or descending when `descending` is
 * nonzero); a single scan. An empty or single-row column is sorted. */
DFTU_EXPORT int32_t dftu_series_is_sorted(const dftu_series* v,
                                          int32_t descending);

/** Drop null rows, returning a SELECTION over `v` of the non-null rows. */
DFTU_EXPORT dftu_series* dftu_series_drop_nulls(const dftu_series* v);

/** Bit-packed Bool mask, true where `v`'s value is present in the `values`
 * column (a hash-set probe). NULL rows are never in the set. Numeric `v`
 * matches numeric `values`; String `v` matches String `values`. */
DFTU_EXPORT dftu_series* dftu_series_is_in(const dftu_series* v,
                                           const dftu_series* values);

/** Sort the rows of `v` ascending (or descending), as a new FLAT column:
 * take(argsort(v, descending)). */
DFTU_EXPORT dftu_series* dftu_series_sort(const dftu_series* v,
                                          int32_t descending);

/** First / last `n` rows (n clamped to [0, length]) as a new FLAT column. */
DFTU_EXPORT dftu_series* dftu_series_head(const dftu_series* v, int64_t n);
DFTU_EXPORT dftu_series* dftu_series_tail(const dftu_series* v, int64_t n);

/** Rows of `v` in reverse order, as a new FLAT column. */
DFTU_EXPORT dftu_series* dftu_series_reverse(const dftu_series* v);

/** Shift rows by `n` (positive = down/lag, negative = up/lead); vacated rows
 * become null. New FLAT column. */
DFTU_EXPORT dftu_series* dftu_series_shift(const dftu_series* v, int64_t n);

/** The `k` largest / smallest rows of `v` (k clamped to [0, length]),
 * best-first: take(topk_indices(v, k, largest)). */
DFTU_EXPORT dftu_series* dftu_series_top_k(const dftu_series* v, int64_t k);
DFTU_EXPORT dftu_series* dftu_series_bottom_k(const dftu_series* v, int64_t k);

/** A DETERMINISTIC sample of `n` distinct rows (n clamped to [0, length]): the
 * rows whose mix64(index, seed) hash is smallest, in ascending row order. Not
 * RNG - reproducible for a given `seed`. */
DFTU_EXPORT dftu_series* dftu_series_sample(const dftu_series* v, int64_t n,
                                            uint64_t seed);

/** Tie-breaking method for dftu_series_rank. */
typedef enum {
    DFTU_RANK_AVERAGE = 0,
    DFTU_RANK_MIN = 1,
    DFTU_RANK_DENSE = 2,
    DFTU_RANK_ORDINAL = 3
} dftu_rank_method;

/** Rank of each row (`method` is a dftu_rank_method:
 * DFTU_RANK_AVERAGE/MIN/DENSE/ORDINAL), ascending unless `descending`. Returns
 * a Float64 column; nulls rank last with a NaN rank. */
DFTU_EXPORT dftu_series* dftu_series_rank(const dftu_series* v,
                                          dftu_rank_method method,
                                          int32_t descending);

/** Reduction for dftu_series_rolling. */
typedef enum {
    DFTU_ROLLING_SUM = 0,
    DFTU_ROLLING_MEAN = 1,
    DFTU_ROLLING_MIN = 2,
    DFTU_ROLLING_MAX = 3
} dftu_rolling_op;

/** Rolling-window reduction (`op` is a dftu_rolling_op:
 * DFTU_ROLLING_SUM/MEAN/MIN/MAX) over `window` trailing rows; the first
 * window-1 rows are null. Returns Float64. */
DFTU_EXPORT dftu_series* dftu_series_rolling(const dftu_series* v,
                                             int64_t window,
                                             dftu_rolling_op op);

/** Rolling sample variance / standard deviation over `window` trailing rows;
 * the first window-1 rows are null, a window of fewer than 2 values yields 0.
 * Returns Float64. */
DFTU_EXPORT dftu_series* dftu_series_rolling_var(const dftu_series* v,
                                                 int64_t window);
DFTU_EXPORT dftu_series* dftu_series_rolling_std(const dftu_series* v,
                                                 int64_t window);
/** Rolling median / quantile (`q` in [0, 1], linear interpolation) over
 * `window` trailing rows; the first window-1 rows are null. Returns Float64. */
DFTU_EXPORT dftu_series* dftu_series_rolling_median(const dftu_series* v,
                                                    int64_t window);
DFTU_EXPORT dftu_series* dftu_series_rolling_quantile(const dftu_series* v,
                                                      int64_t window, double q);

/** Exponentially weighted mean / sample std with smoothing `alpha` in (0, 1],
 * a sequential scan returning Float64. ewm_std's row 0 (variance undefined) is
 * null. */
DFTU_EXPORT dftu_series* dftu_series_ewm_mean(const dftu_series* v,
                                              double alpha);
DFTU_EXPORT dftu_series* dftu_series_ewm_std(const dftu_series* v,
                                             double alpha);

/** Bin each value into the half-open intervals defined by the ascending numeric
 * `breaks` column: bin index = count of breaks <= x, in 0..breaks length.
 * Returns Int32; a null input row yields a null bin. */
DFTU_EXPORT dftu_series* dftu_series_cut(const dftu_series* v,
                                         const dftu_series* breaks);
/** Like dftu_series_cut, but the edges are the `q`-quantiles of the column (q
 * buckets, bins 0..q-1). Returns Int32; a null input row yields a null bin. */
DFTU_EXPORT dftu_series* dftu_series_qcut(const dftu_series* v, int32_t q);

/** For each element of `values`, the lower-bound insertion index into `v`
 * (assumed sorted ascending) that keeps it sorted. Returns an Int64 column of
 * length values length. */
DFTU_EXPORT dftu_series* dftu_series_search_sorted(const dftu_series* v,
                                                   const dftu_series* values);

/** Linearly interpolate null interior values between their nearest non-null
 * neighbors (by row index); leading/trailing nulls stay null. Returns
 * Float64. */
DFTU_EXPORT dftu_series* dftu_series_interpolate(const dftu_series* v);

/** Bit-packed Bool mask, true where lo <= x <= hi (inclusive); carries v's
 * validity. The scalars convert to v's element type (exact for i64). SIMD. */
DFTU_EXPORT dftu_series* dftu_series_is_between(const dftu_series* v,
                                                dftu_scalar lo, dftu_scalar hi);

/** Sum of x[i]*y[i] over the non-null pairs of `v` and `other` (equal length),
 * as an F64 scalar. SIMD for a null-free Float64 pair. Read with
 * scalar_value<double>(). */
DFTU_EXPORT dftu_scalar dftu_series_dot(const dftu_series* v,
                                        const dftu_series* other);

#ifndef __cplusplus
/** C convenience: build a tagged scalar from a value, e.g.
 * dftu_series_compare(col, DFTU_CMP_GT, DFTU_SCALAR_I64(100)). (C++ callers
 * pass a plain value to the typed templates; these are for C.) */
/* Designated, not positional: `len` sits between `kind` and `value`, so a
 * positional form would bind the second initializer to `len`. */
#define DFTU_SCALAR_I64(v) \
    ((dftu_scalar){        \
        .kind = DFTU_SCALAR_TAG_I64, .len = 0, .value = {.i = (int64_t)(v)}})
#define DFTU_SCALAR_U64(v) \
    ((dftu_scalar){        \
        .kind = DFTU_SCALAR_TAG_U64, .len = 0, .value = {.u = (uint64_t)(v)}})
#define DFTU_SCALAR_F64(v) \
    ((dftu_scalar){        \
        .kind = DFTU_SCALAR_TAG_F64, .len = 0, .value = {.d = (double)(v)}})
/** `s` is BORROWED for the call; it must outlive the call it is passed to. */
#define DFTU_SCALAR_STR(p, n)                   \
    ((dftu_scalar){.kind = DFTU_SCALAR_TAG_STR, \
                   .len = (uint32_t)(n),        \
                   .value = {.s = (p)}})
#endif

/** Evaluate `q` as a bit-packed Bool mask over a materialized batch of `n`
 * columns named by `names`. Returns an owned column (length == the columns'
 * length), or NULL if the predicate has no columnar lowering (pattern match,
 * ordered string compare, a referenced field absent from the batch) - the
 * caller should fall back to the scan-time evaluator - or on error. */
DFTU_EXPORT dftu_series* dftu_dataframe_mask(const dftu_query* q,
                                             const dftu_series* const* columns,
                                             const char* const* names,
                                             int32_t n);

/*
 * dftu_dataframe: an opaque handle to a named set of columns (the frame-level
 * counterpart to dftu_series). Frame ops take and return dftu_dataframe*; each
 * returned handle is owned by the caller and freed with dftu_dataframe_free.
 */
typedef struct dftu_dataframe dftu_dataframe;

/** Build a frame from `n` columns named by `names`. Takes ownership of the
 * `columns` handles (they are moved in; do not free them after). NULL on a
 * length/row-count mismatch. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_new(const char* const* names,
                                               dftu_series* const* columns,
                                               int32_t n);
DFTU_EXPORT void dftu_dataframe_free(dftu_dataframe* df);

DFTU_EXPORT int64_t dftu_dataframe_num_rows(const dftu_dataframe* df);
DFTU_EXPORT int32_t dftu_dataframe_num_columns(const dftu_dataframe* df);
/** Name of column `i` (valid for the lifetime of `df`), or NULL if out of
 * range. */
DFTU_EXPORT const char* dftu_dataframe_column_name(const dftu_dataframe* df,
                                                   int32_t i);
/** A column by name, sharing its buffers zero-copy (owned; free with
 * dftu_series_free), or NULL if absent. */
DFTU_EXPORT dftu_series* dftu_dataframe_column(const dftu_dataframe* df,
                                               const char* name);

/* Row ops (return a new frame). */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_take(const dftu_dataframe* df,
                                                const int64_t* idx, int64_t n);
DFTU_EXPORT dftu_dataframe* dftu_dataframe_filter(const dftu_dataframe* df,
                                                  const dftu_series* mask);
DFTU_EXPORT dftu_dataframe* dftu_dataframe_slice(const dftu_dataframe* df,
                                                 int64_t offset, int64_t len);
DFTU_EXPORT dftu_dataframe* dftu_dataframe_head(const dftu_dataframe* df,
                                                int64_t n);
DFTU_EXPORT dftu_dataframe* dftu_dataframe_sort_by(const dftu_dataframe* df,
                                                   const char* name,
                                                   int32_t descending);
DFTU_EXPORT dftu_dataframe* dftu_dataframe_topk(const dftu_dataframe* df,
                                                const char* name, int64_t k,
                                                int32_t largest);

/* Projection ops (share buffers zero-copy). */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_select(const dftu_dataframe* df,
                                                  const char* const* names,
                                                  int32_t n);
DFTU_EXPORT dftu_dataframe* dftu_dataframe_rename(const dftu_dataframe* df,
                                                  const char* const* new_names,
                                                  int32_t n);
DFTU_EXPORT dftu_dataframe* dftu_dataframe_with_column(const dftu_dataframe* df,
                                                       const char* name,
                                                       const dftu_series* col);

/** Drop every row that is null in ANY column. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_drop_nulls(const dftu_dataframe* df);
/** Fill nulls in every column with `value`; columns whose type rejects the
 * scalar (string/bool) are carried over unchanged. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_fill_null(const dftu_dataframe* df,
                                                     dftu_scalar value);
/** Distinct ROWS (keep first), hashing all columns as the composite key.
 * dftu_dataframe_drop_duplicates is an alias. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_unique(const dftu_dataframe* df);
DFTU_EXPORT dftu_dataframe* dftu_dataframe_drop_duplicates(
    const dftu_dataframe* df);
/** Stable lexicographic sort by `n` key columns named by `names`, ascending
 * unless `descending` (nulls last in both directions). */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_sort_by_multi(
    const dftu_dataframe* df, const char* const* names, int32_t n,
    int32_t descending);
/** Per-column direction form: `descending[i]` (0/1) applies to `names[i]`.
 * `descending_n` must be 1 (broadcast to every key) or equal `n`. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_sort_by_multi_per_col(
    const dftu_dataframe* df, const char* const* names, int32_t n,
    const int32_t* descending, int32_t descending_n);
/** The last `n` rows (clamped). */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_tail(const dftu_dataframe* df,
                                                int64_t n);
/** The rows in reverse order. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_reverse(const dftu_dataframe* df);
/** A DETERMINISTIC sample of `n` rows (mix64(index + seed) bottom-n, ascending
 * row order). */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_sample(const dftu_dataframe* df,
                                                  int64_t n, uint64_t seed);
/** Prepend an Int64 column [0, num_rows) named `name`. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_with_row_index(
    const dftu_dataframe* df, const char* name);
/** Per-column summary statistics: one row per statistic (count, null_count,
 * mean, std, min, max), a leading "statistic" String column, and one Float64
 * column per numeric input column. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_describe(const dftu_dataframe* df);
/** A one-row frame with one Int64 column per input column giving its null
 * count. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_null_count(const dftu_dataframe* df);
/** Bit-packed Bool column (length num_rows): true where the whole ROW is
 * duplicated / unique, hashing all columns. Caller owns it (dftu_series_free).
 */
DFTU_EXPORT dftu_series* dftu_dataframe_is_duplicated(const dftu_dataframe* df);
DFTU_EXPORT dftu_series* dftu_dataframe_is_unique(const dftu_dataframe* df);

/** The distinct values of `v` and their counts, as a frame with columns `value`
 * (v's type) and `count` (Int64), most-frequent first. Caller owns the frame
 * (dftu_dataframe_free). */
DFTU_EXPORT dftu_dataframe* dftu_series_value_counts(const dftu_series* v);

/** Reshape wide -> long: keep the `n_id` `id_vars` columns and stack the
 * `n_val` `value_vars` columns into two new columns `variable` (String) and
 * `value`. The result has num_rows * n_val rows; value columns that differ in
 * type must all be numeric and are cast to a common type (Float64 if any is
 * float, else Int64). NULL on an unknown column or an unmixable value set. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_unpivot(
    const dftu_dataframe* df, const char* const* id_vars, int32_t n_id,
    const char* const* value_vars, int32_t n_val);

/** Expand a List `column`: each element becomes its own row (other columns
 * repeated); an empty/null list yields one null row. NULL if `column` is absent
 * or not a List column. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_explode(const dftu_dataframe* df,
                                                   const char* column);

/** One-hot encode `column`: replace it with one Int8 column per distinct value
 * (sorted ascending), named `<column>_<value>` and 1 where the row equals that
 * value. Other columns pass through. NULL if `column` is absent. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_to_dummies(const dftu_dataframe* df,
                                                      const char* column);

/** Reshape long -> wide: rows are the distinct `index` values (sorted), one
 * value column per distinct `columns` value (sorted, named by its value), each
 * cell `values` aggregated over the matching rows. `agg` is the collision
 * reducer: first|last|sum|min|max|mean (NULL defaults to "first"; mean yields
 * Float64). NULL on an unknown column or agg. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_pivot(const dftu_dataframe* df,
                                                 const char* index,
                                                 const char* columns,
                                                 const char* values,
                                                 const char* agg);

/** One aggregate for dftu_dataframe_group_by_dynamic: `op` is
 * sum|min|max|count|mean, `column` the value column (ignored / may be NULL for
 * count), `out` the result column name. */
typedef struct dftu_group_agg {
    const char* op;
    const char* column;
    const char* out;
} dftu_group_agg;

/** Tumbling/sliding time-window aggregation over an ascending Int64 `time_col`:
 * windows start at the first time floored to a multiple of `every`, stride by
 * `every`, and each covers `[start, start + period)` (`period <= 0` means "=
 * every"; `period > every` overlaps). Emits one row per non-empty window: a
 * leading Int64 `time_col` window-start column, then one column per `aggs`
 * entry, named by its `out`. NULL if `every <= 0`, `time_col` is absent or not
 * Int64, or an agg names an unknown column/op. */
DFTU_EXPORT dftu_dataframe* dftu_dataframe_group_by_dynamic(
    const dftu_dataframe* df, const char* time_col, int64_t every,
    int64_t period, const dftu_group_agg* aggs, int32_t n_aggs);

/*
 * dftu_lazyframe: an opaque handle to a deferred query over a dftu_dataframe
 * (the lazy counterpart to the eager frame ops above). Builder ops record a
 * step and return a NEW dftu_lazyframe sharing the source frame's buffers (no
 * data copy); nothing runs until dftu_lazyframe_collect. Every returned handle
 * is owned by the caller and freed with dftu_lazyframe_free. Ops reuse the
 * existing dftu_expr / dftu_group_agg types.
 */
typedef struct dftu_lazyframe dftu_lazyframe;

/* The expression handle (defined in dataframe/expr.h, which includes this
 * header); forward-declared here for the lazyframe filter/with_column ops. */
typedef struct dftu_expr dftu_expr;

/** Wrap a materialized frame as a deferred query over a self-contained
 * in-memory source. The frame's columns are shared (no data copy) into a source
 * that owns them, so the result outlives `df` and never references external
 * scan/executor state; this is the supported way to produce a lazy result that
 * crosses the plugin ABI. `df` is borrowed, not consumed. NULL if `df` is
 * NULL. */
DFTU_EXPORT dftu_lazyframe* dftu_dataframe_lazy(const dftu_dataframe* df);

/** Run the deferred pipeline and materialize the surviving rows into a new
 * owned frame (free with dftu_dataframe_free). Does NOT consume `lf`: it stays
 * valid and re-runnable. `morsel_rows` is the scan chunk size; <= 0 means auto.
 * NULL on error. */
DFTU_EXPORT dftu_dataframe* dftu_lazyframe_collect(dftu_lazyframe* lf,
                                                   int64_t morsel_rows);

/** Free a lazyframe handle. */
DFTU_EXPORT void dftu_lazyframe_free(dftu_lazyframe* lf);

/** Output column names without running the query, joined by '\n' into a new
 * malloc'd C string (empty for a plan ending in a data-dependent op). The
 * caller frees it with dftu_query_string_free. NULL on error. */
DFTU_EXPORT char* dftu_lazyframe_schema(const dftu_lazyframe* lf);

/** The optimized plan as text (one op per line), for introspection. A new
 * malloc'd C string the caller frees with dftu_query_string_free. NULL on
 * error. */
DFTU_EXPORT char* dftu_lazyframe_explain(const dftu_lazyframe* lf);

/** Keep rows where `pred` (a dftu_expr over the frame's columns by position) is
 * true. `pred` is borrowed. NULL on error. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_filter(const dftu_lazyframe* lf,
                                                  const dftu_expr* pred);

/** Project to the `n` columns named by `names`, in that order. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_select(const dftu_lazyframe* lf,
                                                  const char* const* names,
                                                  int32_t n);

/** Add or replace column `name` with the value of `expr` (a dftu_expr over the
 * frame's columns by position). `expr` is borrowed. NULL on error. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_with_column(const dftu_lazyframe* lf,
                                                       const char* name,
                                                       const dftu_expr* expr);

/** Group by the `n_keys` key columns named by `keys` (a composite key) and
 * compute each aggregate in `aggs` (op|column|out, as
 * dftu_dataframe_group_by_dynamic). NULL on an unknown column/op. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_group_by(const dftu_lazyframe* lf,
                                                    const char* const* keys,
                                                    int32_t n_keys,
                                                    const dftu_group_agg* aggs,
                                                    int32_t n_aggs);

/** Sort by the column `name`, ascending unless `descending`. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_sort_by(const dftu_lazyframe* lf,
                                                   const char* name,
                                                   int32_t descending);

/** First / last `n` rows (head is the row-limit form). */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_head(const dftu_lazyframe* lf,
                                                int64_t n);
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_tail(const dftu_lazyframe* lf,
                                                int64_t n);

/** Drop every row that is null in ANY column. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_drop_nulls(const dftu_lazyframe* lf);

/** Distinct rows (keep first), in original order. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_unique(const dftu_lazyframe* lf);

/** Alias of dftu_lazyframe_unique. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_drop_duplicates(
    const dftu_lazyframe* lf);

/** Replace column names positionally with the `n` `names`, keeping column
 * order. `n` must match the frame's column count; a mismatch surfaces as a
 * collect error, not here. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_rename(const dftu_lazyframe* lf,
                                                  const char* const* names,
                                                  int32_t n);

/** Keep `len` rows starting at `offset` (negative `offset` counts from the
 * end, as DataFrame::slice). */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_slice(const dftu_lazyframe* lf,
                                                 int64_t offset, int64_t len);

/** Replace every null with `value`, cast to each column's type. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_fill_null(const dftu_lazyframe* lf,
                                                     dftu_scalar value);

/** Append an Int64 row-index column named `name`, starting at 0. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_with_row_index(
    const dftu_lazyframe* lf, const char* name);

/** A one-row frame of each column's null count. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_null_count(const dftu_lazyframe* lf);

/** Expand a List `column`: each element becomes its own row. An unknown
 * column or a non-List column surfaces as a collect error, not here. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_explode(const dftu_lazyframe* lf,
                                                   const char* column);

/** Reshape wide -> long: keep the `n_id` `id_vars` columns, stack the `n_val`
 * `value_vars` columns into `variable`/`value` columns, as
 * dftu_dataframe_unpivot. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_unpivot(
    const dftu_lazyframe* lf, const char* const* id_vars, int32_t n_id,
    const char* const* value_vars, int32_t n_val);

/** Alias of dftu_lazyframe_unpivot. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_melt(const dftu_lazyframe* lf,
                                                const char* const* id_vars,
                                                int32_t n_id,
                                                const char* const* value_vars,
                                                int32_t n_val);

/** The `k` rows with the largest (or smallest, when `largest` is 0) `name`
 * values. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_topk(const dftu_lazyframe* lf,
                                                const char* name, int64_t k,
                                                int32_t largest);

/** A deterministic n-row sample (mix64 min-hash), bounded to `n` rows. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_sample(const dftu_lazyframe* lf,
                                                  int64_t n, uint64_t seed);

/** One Bool column: true where the whole row is duplicated. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_is_duplicated(
    const dftu_lazyframe* lf);

/** One Bool column: true where the whole row is unique. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_is_unique(const dftu_lazyframe* lf);

/** Tumbling/sliding time-window aggregation over an ascending Int64
 * `time_col`, as dftu_dataframe_group_by_dynamic, plus `origin` (the window
 * grid's zero point) and `origin_min` (nonzero: start the grid at the data's
 * first timestamp instead of `origin`). An invalid `every`, an unknown
 * `time_col`, or an agg naming an unknown column/op surfaces as a collect
 * error, not here. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_group_by_dynamic(
    const dftu_lazyframe* lf, const char* time_col, int64_t every,
    int64_t period, const dftu_group_agg* aggs, int32_t n_aggs, int64_t origin,
    int32_t origin_min);

/** Reshape long -> wide: rows are the distinct `index` values, one value
 * column per distinct `on` value, each cell `values` aggregated over the
 * matching rows. `agg` is the collision reducer: first|last|sum|min|max|mean
 * (NULL defaults to "first"). Output columns are data-dependent, so
 * dftu_lazyframe_schema is empty until collect. An unknown column or agg
 * surfaces as a collect error, not here. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_pivot(const dftu_lazyframe* lf,
                                                 const char* index,
                                                 const char* on,
                                                 const char* values,
                                                 const char* agg);

/** One-hot encode `column`: replace it with one Int8 column per distinct
 * value, named `<column>_<value>`. Output columns are data-dependent, so
 * dftu_lazyframe_schema is empty until collect. An unknown column surfaces as
 * a collect error, not here. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_to_dummies(const dftu_lazyframe* lf,
                                                      const char* column);

/** Per-column summary statistics (count/null_count/mean/std/min/max). Output
 * columns are data-dependent, so dftu_lazyframe_schema is empty until
 * collect. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_describe(const dftu_lazyframe* lf);

/** Out-of-core budget for the pipeline breakers (sort/unique/group_by): when a
 * sink's in-memory state grows past `bytes` it spills to a sorted temp run,
 * k-way merged at collect. 0 means "auto" (~1/3 of available memory). */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_memory_budget(
    const dftu_lazyframe* lf, uint64_t bytes);

/** Sugar for dftu_lazyframe_memory_budget(lf, 0): explicitly request the
 * default auto-spill budget. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_auto_spill(const dftu_lazyframe* lf);

/** Keep rows where the positionally-aligned `mask` is true, as
 * DataFrame::filter. `mask` is borrowed. NULL if `lf` or `mask` is NULL. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_filter_mask(const dftu_lazyframe* lf,
                                                       const dftu_series* mask);

/** Rows in reverse order, as DataFrame::reverse. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_reverse(const dftu_lazyframe* lf);

/** Rows at `idx` (0-based, may repeat or reorder), as DataFrame::take. `idx`
 * is borrowed for the call. NULL if `lf` is NULL. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_take(const dftu_lazyframe* lf,
                                                const int64_t* idx, int32_t n);

/** Stable lexicographic sort by the `n` key columns named by `by`, ascending
 * unless `descending` (nulls last in both directions), as
 * DataFrame::sort_by_multi. */
DFTU_EXPORT dftu_lazyframe* dftu_lazyframe_sort_by_multi(
    const dftu_lazyframe* lf, const char* const* by, int32_t n,
    int32_t descending);

/* ---- Op registry -------------------------------------------------------- */
/* One name-keyed registry over the engine's ops so a built-in op and a user op
 * are looked up and run the same way (the plugin-ABI foundation). Built-in ops
 * carry a plain name (`add`); a user op should namespace with a module prefix
 * (`mymod.zscore`) so it never shadows a built-in. */

/** Coarse category, for filtering/listing (== dftu_op_kind_of(sig)). */
typedef enum {
    DFTU_OP_KIND_SERIES = 0, /**< column(s) -> column */
    DFTU_OP_KIND_AGGREGATE,  /**< column -> scalar (a reducer) */
    DFTU_OP_KIND_FRAME,      /**< table(s) -> table */
    DFTU_OP_KIND_LAZY        /**< lazy plan -> lazy plan (a plan node) */
} dftu_op_kind;

/** Operand/return tokens, one 5-bit field each, so 32 values exist in total. A
 * signature packs a return token plus up to DFTU_OP_MAX_ARGS operand tokens
 * into one integer (DFTU_OP_SIG6), the way a Linux ioctl number packs
 * direction/type/nr/size. APPEND ONLY - a token's value is ABI-stable, and a
 * value past 31 would silently corrupt every signature it appears in. */
typedef enum {
    DFTU_TOK_NONE = 0, /**< empty slot */
    DFTU_TOK_SERIES,   /**< a column operand, or a column return */
    DFTU_TOK_SCALAR,   /**< dftu_scalar operand, or a dftu_scalar return */
    DFTU_TOK_I64,      /**< int64 operand, or an int64 return */
    DFTU_TOK_BOOL,     /**< int32 0/1 return */
    DFTU_TOK_CMP,      /**< dftu_cmp_op operand */
    DFTU_TOK_PRIM,     /**< dftu_prim_op operand */
    DFTU_TOK_LOGICAL,  /**< dftu_logical_op operand */
    DFTU_TOK_DTYPE,    /**< dftu_dtype operand */
    DFTU_TOK_REDUCE,   /**< dftu_reduce_op operand */
    DFTU_TOK_STR,      /**< (const char*, int32 length) operand */
    DFTU_TOK_CHAR,     /**< char operand */
    DFTU_TOK_F64,      /**< double operand, or a double return */
    DFTU_TOK_I32,      /**< int32 flag operand (e.g. a `descending` flag) */
    DFTU_TOK_RANK,     /**< dftu_rank_method operand */
    DFTU_TOK_ROLLING,  /**< dftu_rolling_op operand */
    DFTU_TOK_FRAME, /**< a dftu_dataframe operand, or a dftu_dataframe return */
    DFTU_TOK_STRLIST, /**< a (const char* const*, int32 count) string-list
                         operand */
    DFTU_TOK_I32LIST, /**< a (const int32_t*, int32 count) int32-list operand */
    DFTU_TOK_LAZY,    /**< a dftu_lazyframe operand, or a dftu_lazyframe return
                       */
    DFTU_TOK_EXPR,    /**< a const dftu_expr* operand (a predicate or a
                         projection), borrowed for the call */
    DFTU_TOK_AGGLIST, /**< a (const dftu_group_agg*, int32 count) aggregate-spec
                         list operand */
    DFTU_TOK_U64,     /**< uint64 operand (a byte count, a seed), distinct from
                         I64 so a signature describes the real parameter type
                         and cannot collide with a signed-shaped op */
    DFTU_TOK_I64LIST  /**< a (const int64_t*, int32 count) int64-list operand
                         (e.g. LazyFrame::take's row indices) */
} dftu_op_tok;

/** One past the last token; a macro, so it does not become a case every switch
 * over dftu_op_tok has to answer for. Update alongside the last token. */
#define DFTU_TOK_COUNT (DFTU_TOK_I64LIST + 1)

/** Max operand tokens a signature carries (5-bit fields: a return token plus up
 * to this many operands pack into one int64). */
#define DFTU_OP_MAX_ARGS 7

/** Pack a signature from a return token and up to three operand tokens. Pass
 * each token's suffix (e.g. SERIES for DFTU_TOK_SERIES; pad unused operand
 * slots with NONE) - the suffix is pasted onto DFTU_TOK_. Compose new
 * signatures from tokens rather than allocating opaque ordinals; decode with
 * DFTU_OP_SIG_RET / DFTU_OP_SIG_ARG. Use DFTU_OP_SIG6 (5 operands) or
 * DFTU_OP_SIG8 (7 operands) for a wider op. */
#define DFTU_OP_SIG(ret, o0, o1, o2) DFTU_OP_SIG6(ret, o0, o1, o2, NONE, NONE)
/** Pack a signature with up to five operand tokens (5-bit fields). */
#define DFTU_OP_SIG6(ret, o0, o1, o2, o3, o4) \
    DFTU_OP_SIG8(ret, o0, o1, o2, o3, o4, NONE, NONE)
/** Pack a signature with up to seven operand tokens (5-bit fields): a return
 * token plus 7 operands is 8 * 5 = 40 bits, past int32 range, so every packed
 * field is cast to int64. */
#define DFTU_OP_SIG8(ret, o0, o1, o2, o3, o4, o5, o6)                  \
    ((int64_t)DFTU_TOK_##ret | ((int64_t)DFTU_TOK_##o0 << 5) |         \
     ((int64_t)DFTU_TOK_##o1 << 10) | ((int64_t)DFTU_TOK_##o2 << 15) | \
     ((int64_t)DFTU_TOK_##o3 << 20) | ((int64_t)DFTU_TOK_##o4 << 25) | \
     ((int64_t)DFTU_TOK_##o5 << 30) | ((int64_t)DFTU_TOK_##o6 << 35))
/** The return token of a signature. */
#define DFTU_OP_SIG_RET(sig) ((dftu_op_tok)((int64_t)(sig) & 0x1F))
/** Operand token `i` in [0, DFTU_OP_MAX_ARGS); DFTU_TOK_NONE past the last. */
#define DFTU_OP_SIG_ARG(sig, i) \
    ((dftu_op_tok)(((int64_t)(sig) >> (5 + 5 * (i))) & 0x1F))

/** A packed op signature: build it with DFTU_OP_SIG(ret, o0, o1, o2) at the
 * registration site, and use the same expression as a runner switch-case label
 * (a constant), the way a driver composes and switches on ioctl numbers. There
 * is deliberately no enum of named signatures: an op of an existing shape adds
 * only a registry line, and only a brand-new shape adds a runner case. */
typedef int64_t dftu_op_sig;

#ifdef __cplusplus
static_assert(sizeof(dftu_op_sig) * 8 >= 5 * (DFTU_OP_MAX_ARGS + 1),
              "dftu_op_sig is too narrow for DFTU_OP_MAX_ARGS 5-bit fields "
              "(a return token plus DFTU_OP_MAX_ARGS operands)");
static_assert(DFTU_TOK_COUNT <= 32, "dftu_op_tok exceeds its 5-bit field");
static_assert(sizeof(DFTU_OP_SIG6(SERIES, SERIES, NONE, NONE, NONE, NONE)) ==
                  sizeof(dftu_op_sig),
              "DFTU_OP_SIG6 computes in a narrower type than dftu_op_sig");
static_assert(sizeof(DFTU_OP_SIG8(SERIES, SERIES, NONE, NONE, NONE, NONE, NONE,
                                  NONE)) == sizeof(dftu_op_sig),
              "DFTU_OP_SIG8 computes in a narrower type than dftu_op_sig");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(dftu_op_sig) * 8 >= 5 * (DFTU_OP_MAX_ARGS + 1),
               "dftu_op_sig is too narrow for DFTU_OP_MAX_ARGS 5-bit fields "
               "(a return token plus DFTU_OP_MAX_ARGS operands)");
_Static_assert(DFTU_TOK_COUNT <= 32, "dftu_op_tok exceeds its 5-bit field");
_Static_assert(sizeof(DFTU_OP_SIG6(SERIES, SERIES, NONE, NONE, NONE, NONE)) ==
                   sizeof(dftu_op_sig),
               "DFTU_OP_SIG6 computes in a narrower type than dftu_op_sig");
_Static_assert(sizeof(DFTU_OP_SIG8(SERIES, SERIES, NONE, NONE, NONE, NONE, NONE,
                                   NONE)) == sizeof(dftu_op_sig),
               "DFTU_OP_SIG8 computes in a narrower type than dftu_op_sig");
#endif

/** A registry record. `sig` is authoritative: it packs the whole signature, so
 * it selects the fn cast, the operands read, and (via dftu_op_kind_of) the
 * category. For a built-in, `name` is a literal and must outlive the process;
 * dftu_op_register copies both the record and the name it points to, so a
 * caller's own `name` storage need not outlive the call. `fn` is always
 * borrowed: it is a pointer into the registrant's code, so it stops being
 * callable once that code unloads (see dftu_op_unregister). */
typedef struct dftu_op_desc {
    const char* name; /**< registry key, e.g. "add" or "mymod.zscore" */
    dftu_op_sig sig;  /**< the packed signature the runner dispatches on */
    const void* fn;   /**< the engine function pointer */
} dftu_op_desc;

/** The coarse category of a signature (decoded from its return token). */
DFTU_EXPORT dftu_op_kind dftu_op_kind_of(dftu_op_sig sig);

/** Number of leading `series` column operands a signature takes (its column
 * arity). */
DFTU_EXPORT uint32_t dftu_op_arity(dftu_op_sig sig);

/** A human-readable spelling of `sig`, e.g. "(series, scalar) -> series",
 * decoded from its tokens, for listing/discovery. Static storage; never NULL.
 */
DFTU_EXPORT const char* dftu_op_signature(dftu_op_sig sig);

/** The registered op named `name` (built-in or user), or NULL if none. */
DFTU_EXPORT const dftu_op_desc* dftu_op_find(const char* name);

/** Number of registered ops (built-ins + user), for discovery/listing. */
DFTU_EXPORT uint32_t dftu_op_count(void);

/** The op at index `i` in [0, dftu_op_count()), or NULL if out of range. The
 * ordering is unspecified and may change as user ops are registered. */
DFTU_EXPORT const dftu_op_desc* dftu_op_at(uint32_t i);

/** Register a user op. The record and its name are copied; `fn` is still
 * borrowed. Returns 0 on success, non-zero if `desc`/its name is NULL or the
 * name is already registered (no silent shadowing). A registrant whose code
 * can unload (a plugin's .so) must call dftu_op_unregister before unloading,
 * or `fn` dangles in a registry entry no one can remove it from. */
DFTU_EXPORT int dftu_op_register(const dftu_op_desc* desc);

/** Remove a user op previously added with dftu_op_register. A no-op (returns
 * non-zero) if `name` is NULL, unregistered, or a built-in - built-ins can
 * never be removed this way. Safe to call for a name that was never
 * registered. */
DFTU_EXPORT int dftu_op_unregister(const char* name);

/** One operand slot. Only the union member the operand's token names is read:
 * SCALAR->scalar, I64->i64, F64->f64, CHAR->ch, an enum token (CMP/PRIM/
 * LOGICAL/DTYPE/REDUCE/I32/RANK/ROLLING)->i32, STR->str, STRLIST->list,
 * I32LIST->i32list, I64LIST->i64list, and a frame op's SERIES operand (a
 * mask/column)->series. A SERIES/FRAME operand of a column/frame op is passed
 * in the runner's in[]/frames[] array, not here. */
typedef union dftu_op_val {
    dftu_scalar scalar;
    int64_t i64;
    uint64_t u64;
    int32_t i32;
    double f64;
    char ch;
    const dftu_series* series;
    struct {
        const char* ptr;
        int32_t len;
    } str;
    struct {
        const char* const* items;
        int32_t n;
    } list;
    struct {
        const int32_t* items;
        int32_t n;
    } i32list;
    struct {
        const int64_t* items;
        int32_t n;
    } i64list;
    const dftu_lazyframe* lazy;
    const dftu_expr* expr;
    struct {
        const dftu_group_agg* items;
        int32_t n;
    } agglist;
} dftu_op_val;

/** The operands an op consumes, one slot per operand token in order (args[i]
 * matches the i-th operand token of the signature). Only the slots a given op
 * needs are read; pass NULL when it needs none. */
typedef struct dftu_op_arg {
    dftu_op_val args[DFTU_OP_MAX_ARGS];
} dftu_op_arg;

/** Run a column op (a signature whose return token is SERIES): `in` are `n`
 * borrowed input columns (n must equal dftu_op_arity(op->sig)), `arg` supplies
 * the remaining operands the signature names (NULL when there are none).
 * Returns a new owned column (free with dftu_series_free), or NULL on a
 * NULL/kind/arity/shape mismatch. */
DFTU_EXPORT dftu_series* dftu_op_run(const dftu_op_desc* op,
                                     const dftu_series* const* in, uint32_t n,
                                     const dftu_op_arg* arg);

/** Run a reducer (a signature whose return token is SCALAR/I64/BOOL) on one
 * column, returning its reduction as a dftu_scalar (an i64/bool return is
 * widened into the I64 domain). `arg` is read only by a signature with an
 * operand token (e.g. REDUCE). On a NULL/kind/shape mismatch sets *ok to 0
 * (when ok != NULL) and returns a zero scalar; otherwise sets *ok to 1. */
DFTU_EXPORT dftu_scalar dftu_op_run_aggregate(const dftu_op_desc* op,
                                              const dftu_series* v,
                                              const dftu_op_arg* arg, int* ok);

/** Run a frame op (a signature whose return token is FRAME): `frames` are the
 * `n` borrowed input dftu_dataframe operands (n == dftu_op_arity(op->sig)), and
 * every other operand (series/scalar/string/list/int) rides `arg`. Returns a
 * new owned dftu_dataframe (free with dftu_dataframe_free), or NULL on a
 * NULL/kind/arity/shape mismatch. */
DFTU_EXPORT dftu_dataframe* dftu_op_run_frame(
    const dftu_op_desc* op, const dftu_dataframe* const* frames, uint32_t n,
    const dftu_op_arg* arg);

/** Run a lazy op (a signature whose return token is LAZY): `in` are the `n`
 * borrowed input dftu_lazyframe operands (n == dftu_op_arity(op->sig)), and
 * every other operand (string/scalar/list/expr/agglist/int) rides `arg`.
 * Returns a new owned dftu_lazyframe (free with dftu_lazyframe_free), or NULL
 * on a NULL/kind/arity/shape mismatch. */
DFTU_EXPORT dftu_lazyframe* dftu_op_run_lazy(const dftu_op_desc* op,
                                             const dftu_lazyframe* const* in,
                                             uint32_t n,
                                             const dftu_op_arg* arg);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // DFTRACER_UTILS_DATAFRAME_ABI_H
