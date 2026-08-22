#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/core/common/error.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/utilities/common/arrow/array_view.h>
#include <dftracer/utils/utilities/common/arrow/column_builder.h>
#include <dftracer/utils/utilities/common/arrow/explode.h>
#include <dftracer/utils/utilities/common/arrow/join_internal.h>
#include <nanoarrow/nanoarrow.h>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace dftracer::utils::utilities::common::arrow {

namespace {

// Null-checked scalar append, so nulls survive scalar-element and struct-field
// explosion of the target column.
void append_elem_scalar(RecordBatchBuilder& b, std::size_t out, ColumnType ct,
                        const ArrowArrayView* v, std::int64_t pos) {
    if (ArrowArrayViewIsNull(v, pos))
        b.append_null(out);
    else
        append_scalar(b, out, ct, v, pos);
}

}  // namespace

ArrowExportResult explode(ArrowSchema* schema, ArrowArray* array,
                          std::size_t list_col_idx, bool keep_empty) {
    ArrowArrayView av;
    if (init_array_view(av, schema, array) != NANOARROW_OK) {
        throw DFTUtilsException(ErrorCode::INTERNAL,
                                "unnest: failed to view input batch");
    }
    struct ViewGuard {
        ArrowArrayView* v;
        ~ViewGuard() { ArrowArrayViewReset(v); }
    } guard{&av};

    const std::int64_t ncols = av.n_children;
    if (list_col_idx >= static_cast<std::size_t>(ncols)) {
        throw DFTUtilsException(ErrorCode::INVALID_ARGUMENT,
                                "unnest: column index out of range");
    }

    const ArrowArrayView* lv = av.children[list_col_idx];
    if (lv->storage_type != NANOARROW_TYPE_LIST &&
        lv->storage_type != NANOARROW_TYPE_LARGE_LIST) {
        throw DFTUtilsException(ErrorCode::INVALID_ARGUMENT,
                                "unnest: selected column is not a list");
    }
    const ArrowArrayView* elem = lv->children[0];
    const bool elem_is_struct = elem->storage_type == NANOARROW_TYPE_STRUCT;

    // The list column expands in place to one scalar column or one column per
    // struct field; passthrough columns keep their type.
    std::vector<ColumnSpec> specs;
    std::vector<ValueCol> pass_col(static_cast<std::size_t>(ncols));
    std::vector<ColumnType> field_types;
    ColumnType elem_scalar_type = ColumnType::INT64;
    std::unordered_set<std::string_view> names;

    auto claim_name = [&](const char* nm) {
        std::string_view sv = nm ? nm : "";
        if (!names.insert(sv).second) {
            throw DFTUtilsException(
                ErrorCode::INVALID_ARGUMENT,
                std::string("unnest: output column name collision: ")
                    .append(sv));
        }
    };

    for (std::int64_t c = 0; c < ncols; ++c) {
        const char* cname = schema->children[c]->name;
        if (static_cast<std::size_t>(c) == list_col_idx) {
            if (elem_is_struct) {
                const ArrowSchema* item_schema =
                    schema->children[list_col_idx]->children[0];
                for (std::int64_t f = 0; f < elem->n_children; ++f) {
                    auto ft = scalar_type_from_storage(
                        elem->children[f]->storage_type);
                    if (!ft) {
                        throw DFTUtilsException(
                            ErrorCode::INVALID_ARGUMENT,
                            "unnest: unsupported struct field type");
                    }
                    field_types.push_back(*ft);
                    const char* fname = item_schema->children[f]->name;
                    claim_name(fname);
                    specs.push_back({fname ? fname : "", *ft});
                }
            } else {
                auto et = scalar_type_from_storage(elem->storage_type);
                if (!et) {
                    throw DFTUtilsException(
                        ErrorCode::INVALID_ARGUMENT,
                        "unnest: unsupported list element type");
                }
                elem_scalar_type = *et;
                claim_name(cname);
                specs.push_back({cname ? cname : "", *et});
            }
        } else {
            claim_name(cname);
            ColumnSpec spec;
            pass_col[static_cast<std::size_t>(c)] = plan_value_col(
                av.children[c], schema->children[c], cname, spec);
            specs.push_back(std::move(spec));
        }
    }

    RecordBatchBuilder b;
    b.declare_schema(specs);

    auto emit_row = [&](std::int64_t r, std::int64_t pos) {
        std::size_t out = 0;
        for (std::int64_t c = 0; c < ncols; ++c) {
            if (static_cast<std::size_t>(c) == list_col_idx) {
                if (elem_is_struct) {
                    for (std::size_t f = 0; f < field_types.size(); ++f) {
                        if (pos < 0)
                            b.append_null(out);
                        else
                            append_elem_scalar(b, out, field_types[f],
                                               elem->children[f], pos);
                        ++out;
                    }
                } else {
                    if (pos < 0)
                        b.append_null(out);
                    else
                        append_elem_scalar(b, out, elem_scalar_type, elem, pos);
                    ++out;
                }
            } else {
                append_value(b, out, pass_col[static_cast<std::size_t>(c)], r);
                ++out;
            }
        }
        b.end_row();
    };

    const std::int64_t nrows = av.length;
    for (std::int64_t r = 0; r < nrows; ++r) {
        const bool list_null = ArrowArrayViewIsNull(lv, r) != 0;
        const std::int64_t start =
            list_null ? 0 : ArrowArrayViewListChildOffset(lv, r);
        const std::int64_t end =
            list_null ? 0 : ArrowArrayViewListChildOffset(lv, r + 1);
        if (end <= start) {
            // Empty/null list: SQL inner unnest drops the row; outer unnest
            // (keep_empty) emits one null-exploded row.
            if (keep_empty) emit_row(r, -1);
            continue;
        }
        for (std::int64_t pos = start; pos < end; ++pos) emit_row(r, pos);
    }

    return b.finish();
}

}  // namespace dftracer::utils::utilities::common::arrow

#endif  // DFTRACER_UTILS_ENABLE_ARROW
