#include <dftracer/utils/dataframe/arrow_bridge.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>
#include <nanoarrow/nanoarrow.h>

#include <cstring>
#include <memory>
#include <vector>

namespace dftracer::utils::dataframe {
namespace {

ArrowType to_arrow_type(TypeId t) {
    switch (t) {
        case TypeId::Bool:
            return NANOARROW_TYPE_BOOL;
        case TypeId::Int8:
            return NANOARROW_TYPE_INT8;
        case TypeId::Int16:
            return NANOARROW_TYPE_INT16;
        case TypeId::Int32:
            return NANOARROW_TYPE_INT32;
        case TypeId::Int64:
            return NANOARROW_TYPE_INT64;
        case TypeId::Uint8:
            return NANOARROW_TYPE_UINT8;
        case TypeId::Uint16:
            return NANOARROW_TYPE_UINT16;
        case TypeId::Uint32:
            return NANOARROW_TYPE_UINT32;
        case TypeId::Uint64:
            return NANOARROW_TYPE_UINT64;
        case TypeId::Float32:
            return NANOARROW_TYPE_FLOAT;
        case TypeId::Float64:
            return NANOARROW_TYPE_DOUBLE;
        case TypeId::String:
            return NANOARROW_TYPE_STRING;
        case TypeId::Binary:
            return NANOARROW_TYPE_BINARY;
        case TypeId::List:
            return NANOARROW_TYPE_LIST;
        case TypeId::Struct:
            return NANOARROW_TYPE_STRUCT;
    }
    return NANOARROW_TYPE_UNINITIALIZED;
}

bool from_arrow_type(ArrowType t, TypeId& out) {
    switch (t) {
        case NANOARROW_TYPE_BOOL:
            out = TypeId::Bool;
            return true;
        case NANOARROW_TYPE_INT8:
            out = TypeId::Int8;
            return true;
        case NANOARROW_TYPE_INT16:
            out = TypeId::Int16;
            return true;
        case NANOARROW_TYPE_INT32:
            out = TypeId::Int32;
            return true;
        case NANOARROW_TYPE_INT64:
            out = TypeId::Int64;
            return true;
        case NANOARROW_TYPE_UINT8:
            out = TypeId::Uint8;
            return true;
        case NANOARROW_TYPE_UINT16:
            out = TypeId::Uint16;
            return true;
        case NANOARROW_TYPE_UINT32:
            out = TypeId::Uint32;
            return true;
        case NANOARROW_TYPE_UINT64:
            out = TypeId::Uint64;
            return true;
        case NANOARROW_TYPE_FLOAT:
            out = TypeId::Float32;
            return true;
        case NANOARROW_TYPE_DOUBLE:
            out = TypeId::Float64;
            return true;
        case NANOARROW_TYPE_STRING:
            out = TypeId::String;
            return true;
        case NANOARROW_TYPE_BINARY:
            out = TypeId::Binary;
            return true;
        default:
            return false;
    }
}

/// Keeps a column's buffers (and, for a selection, its base) alive for as long
/// as an exported ArrowArray references them, and owns the array's buffer list.
struct ExportedArray {
    std::vector<std::shared_ptr<Buffer>> keep;
    std::shared_ptr<dftu_series> child_keep;
    std::vector<std::shared_ptr<dftu_series>> children_keep;
    const void* buffers[3];
};

bool is_varwidth(TypeId t) {
    return t == TypeId::String || t == TypeId::Binary;
}

void release_exported(ArrowArray* array) {
    if (array->dictionary != nullptr) {
        if (array->dictionary->release != nullptr)
            array->dictionary->release(array->dictionary);
        delete array->dictionary;
        array->dictionary = nullptr;
    }
    for (std::int64_t i = 0; i < array->n_children; ++i) {
        if (array->children[i]->release != nullptr)
            array->children[i]->release(array->children[i]);
        delete array->children[i];
    }
    delete[] array->children;
    array->children = nullptr;
    delete static_cast<ExportedArray*>(array->private_data);
    array->private_data = nullptr;
    array->release = nullptr;
}

void export_array(const dftu_series& col, ArrowArray* array);

void export_flat(const dftu_series& col, ArrowArray* array) {
    auto* h = new ExportedArray();
    h->keep.push_back(col.validity);
    h->keep.push_back(col.data);
    h->buffers[0] = col.validity ? col.validity->data() : nullptr;
    h->buffers[1] = col.data ? col.data->data() : nullptr;
    array->length = col.length;
    array->null_count = col.null_count;
    array->offset = 0;
    array->n_buffers = 2;
    array->buffers = h->buffers;
    array->private_data = h;
    array->release = release_exported;
}

// SELECTION and DICTIONARY both export as an Arrow dictionary array: int32
// indices/codes with the base/dictionary as the child array.
void export_dict(const dftu_series& col, ArrowArray* array) {
    auto* h = new ExportedArray();
    h->keep.push_back(col.validity);
    h->keep.push_back(col.data);
    h->child_keep = col.child;
    h->buffers[0] = col.validity ? col.validity->data() : nullptr;
    h->buffers[1] = col.data ? col.data->data() : nullptr;
    array->length = col.length;
    array->null_count = col.null_count;
    array->offset = 0;
    array->n_buffers = 2;
    array->buffers = h->buffers;
    array->private_data = h;
    array->release = release_exported;
    array->dictionary = new ArrowArray();
    export_array(*col.child, array->dictionary);
}

void export_varwidth(const dftu_series& col, ArrowArray* array) {
    auto* h = new ExportedArray();
    h->keep.push_back(col.validity);
    h->keep.push_back(col.offsets);
    h->keep.push_back(col.data);
    h->buffers[0] = col.validity ? col.validity->data() : nullptr;
    h->buffers[1] = col.offsets ? col.offsets->data() : nullptr;
    h->buffers[2] = col.data ? col.data->data() : nullptr;
    array->length = col.length;
    array->null_count = col.null_count;
    array->offset = 0;
    array->n_buffers = 3;
    array->buffers = h->buffers;
    array->private_data = h;
    array->release = release_exported;
}

// STRUCT: one validity buffer, one child array per field.
void export_struct(const dftu_series& col, ArrowArray* array) {
    auto* h = new ExportedArray();
    h->keep.push_back(col.validity);
    h->buffers[0] = col.validity ? col.validity->data() : nullptr;
    array->length = col.length;
    array->null_count = col.null_count;
    array->offset = 0;
    array->n_buffers = 1;
    array->buffers = h->buffers;
    array->n_children = static_cast<std::int64_t>(col.children.size());
    array->children = new ArrowArray*[col.children.size()];
    for (std::size_t i = 0; i < col.children.size(); ++i) {
        array->children[i] = new ArrowArray();
        export_array(*col.children[i], array->children[i]);
        h->children_keep.push_back(col.children[i]);
    }
    array->private_data = h;
    array->release = release_exported;
}

// LIST: validity + int32 offsets, one child array (the flattened values).
void export_list(const dftu_series& col, ArrowArray* array) {
    auto* h = new ExportedArray();
    h->keep.push_back(col.validity);
    h->keep.push_back(col.offsets);
    h->child_keep = col.child;
    h->buffers[0] = col.validity ? col.validity->data() : nullptr;
    h->buffers[1] = col.offsets ? col.offsets->data() : nullptr;
    array->length = col.length;
    array->null_count = col.null_count;
    array->offset = 0;
    array->n_buffers = 2;
    array->buffers = h->buffers;
    array->n_children = 1;
    array->children = new ArrowArray*[1];
    array->children[0] = new ArrowArray();
    export_array(*col.child, array->children[0]);
    array->private_data = h;
    array->release = release_exported;
}

bool is_dict_encoded(const dftu_series& col) {
    return col.encoding == Encoding::Selection ||
           col.encoding == Encoding::Dictionary;
}

void export_array(const dftu_series& col, ArrowArray* array) {
    std::memset(array, 0, sizeof(*array));
    if (is_dict_encoded(col))
        export_dict(col, array);
    else if (col.type == TypeId::Struct)
        export_struct(col, array);
    else if (col.type == TypeId::List)
        export_list(col, array);
    else if (is_varwidth(col.type))
        export_varwidth(col, array);
    else
        export_flat(col, array);
}

void build_schema(const dftu_series& col, ArrowSchema* schema);

// Set the type of a pre-initialized (already-named) schema in place. Used for a
// LIST's "item" child, which ArrowSchemaInitFromType(LIST) already allocated
// and named - re-initing it would orphan that name allocation.
void set_item_schema(const dftu_series& col, ArrowSchema* item) {
    if (col.type == TypeId::Struct) {
        ArrowSchemaSetType(item, NANOARROW_TYPE_STRUCT);
        ArrowSchemaAllocateChildren(
            item, static_cast<std::int64_t>(col.children.size()));
        for (std::size_t i = 0; i < col.children.size(); ++i) {
            build_schema(*col.children[i], item->children[i]);
            ArrowSchemaSetName(item->children[i], col.field_names[i].c_str());
        }
    } else {
        ArrowSchemaSetType(item, to_arrow_type(col.type));
    }
}

// Initialize a fresh `schema` (root, or a child from AllocateChildren) for
// `col`.
void build_schema(const dftu_series& col, ArrowSchema* schema) {
    if (is_dict_encoded(col)) {
        // SELECTION indices are int64; DICTIONARY codes are int32.
        ArrowSchemaInitFromType(schema, col.encoding == Encoding::Selection
                                            ? NANOARROW_TYPE_INT64
                                            : NANOARROW_TYPE_INT32);
        ArrowSchemaAllocateDictionary(schema);
        build_schema(*col.child, schema->dictionary);
    } else if (col.type == TypeId::Struct) {
        ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT);
        ArrowSchemaAllocateChildren(
            schema, static_cast<std::int64_t>(col.children.size()));
        for (std::size_t i = 0; i < col.children.size(); ++i) {
            build_schema(*col.children[i], schema->children[i]);
            ArrowSchemaSetName(schema->children[i], col.field_names[i].c_str());
        }
    } else if (col.type == TypeId::List) {
        ArrowSchemaInitFromType(schema, NANOARROW_TYPE_LIST);
        set_item_schema(*col.child, schema->children[0]);
    } else {
        ArrowSchemaInitFromType(schema, to_arrow_type(col.type));
    }
}

}  // namespace

void to_arrow(const Series& col, ArrowSchema* schema, ArrowArray* array) {
    const dftu_series* c = col.handle();
    if (c == nullptr) {
        // A null handle is an empty result; export a valid empty struct rather
        // than dereferencing it.
        ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT);
        auto* h = new ExportedArray();
        std::memset(array, 0, sizeof(*array));
        array->length = 0;
        array->n_buffers = 1;
        array->buffers = h->buffers;
        array->private_data = h;
        array->release = release_exported;
        return;
    }
    build_schema(*c, schema);
    export_array(*c, array);
}

namespace {

// `owner` keeps the backing Arrow array alive for as long as any wrapped buffer
// survives, so the imported buffers can alias `arr` zero copy.
Series import_flat(const ArrowSchema* schema, const ArrowArray* arr,
                   std::shared_ptr<void> owner) {
    ArrowSchemaView view;
    ArrowError error;
    if (ArrowSchemaViewInit(&view, schema, &error) != NANOARROW_OK)
        return Series{};
    TypeId type;
    if (!from_arrow_type(view.type, type)) return Series{};

    auto* col = new dftu_series();
    col->type = type;
    col->encoding = Encoding::Flat;
    col->length = arr->length;
    col->null_count = arr->null_count < 0 ? 0 : arr->null_count;
    std::int64_t n = arr->length;

    auto wrap_validity = [&]() {
        if (arr->buffers[0] != nullptr) {
            auto* val =
                static_cast<std::uint8_t*>(const_cast<void*>(arr->buffers[0]));
            col->validity = Buffer::wrap(
                val, (static_cast<std::size_t>(n) + 7) / 8, [owner](void*) {});
        }
    };

    if (is_varwidth(type)) {
        // 3 buffers: validity, int32 offsets (n+1), data. We import offset 0.
        const std::int32_t* offs =
            static_cast<const std::int32_t*>(arr->buffers[1]);
        std::size_t data_len =
            offs != nullptr ? static_cast<std::size_t>(offs[n]) : 0;
        col->offsets = Buffer::wrap(
            static_cast<std::uint8_t*>(const_cast<void*>(arr->buffers[1])),
            static_cast<std::size_t>(n + 1) * sizeof(std::int32_t),
            [owner](void*) {});
        col->data = Buffer::wrap(
            static_cast<std::uint8_t*>(const_cast<void*>(arr->buffers[2])),
            data_len, [owner](void*) {});
        wrap_validity();
        return Series{col};
    }

    // Bool is bit-packed, so its data buffer is sized by buffer_bytes and a
    // (rare) nonzero Arrow offset would be bit-level; we only import offset 0.
    std::size_t width = byte_width(type);
    std::size_t ptr_off = (type == TypeId::Bool)
                              ? 0
                              : static_cast<std::size_t>(arr->offset) * width;
    auto* data = static_cast<std::uint8_t*>(const_cast<void*>(arr->buffers[1]));
    col->data =
        Buffer::wrap(data + ptr_off, buffer_bytes(type, n), [owner](void*) {});
    wrap_validity();
    return Series{col};
}

Series import_any(const ArrowSchema* schema, const ArrowArray* arr,
                  std::shared_ptr<void> owner);

// LIST: validity + int32 offsets (n+1), one child values array. Mirrors
// export_list. Only int32-offset lists (NANOARROW_TYPE_LIST) and parent offset
// 0 are imported; the child is imported recursively.
Series import_list(const ArrowSchema* schema, const ArrowArray* arr,
                   std::shared_ptr<void> owner) {
    if (schema->n_children != 1 || arr->n_children != 1) return Series{};
    auto* col = new dftu_series();
    col->type = TypeId::List;
    col->encoding = Encoding::Flat;
    col->length = arr->length;
    col->null_count = arr->null_count < 0 ? 0 : arr->null_count;
    std::int64_t n = arr->length;
    col->offsets = Buffer::wrap(
        static_cast<std::uint8_t*>(const_cast<void*>(arr->buffers[1])),
        static_cast<std::size_t>(n + 1) * sizeof(std::int32_t),
        [owner](void*) {});
    if (arr->buffers[0] != nullptr)
        col->validity = Buffer::wrap(
            static_cast<std::uint8_t*>(const_cast<void*>(arr->buffers[0])),
            (static_cast<std::size_t>(n) + 7) / 8, [owner](void*) {});
    Series child = import_any(schema->children[0], arr->children[0], owner);
    if (!child.valid()) {
        delete col;
        return Series{};
    }
    col->child = std::shared_ptr<dftu_series>(child.release());
    return Series{col};
}

// Inverse of export_struct; each field column is imported recursively and
// aliases `arr` through `owner`.
Series import_struct(const ArrowSchema* schema, const ArrowArray* arr,
                     std::shared_ptr<void> owner) {
    if (schema->n_children != arr->n_children) return Series{};
    auto* col = new dftu_series();
    col->type = TypeId::Struct;
    col->encoding = Encoding::Flat;
    col->length = arr->length;
    col->null_count = arr->null_count < 0 ? 0 : arr->null_count;
    const std::int64_t n = arr->length;
    if (arr->n_buffers > 0 && arr->buffers[0] != nullptr)
        col->validity = Buffer::wrap(
            static_cast<std::uint8_t*>(const_cast<void*>(arr->buffers[0])),
            (static_cast<std::size_t>(n) + 7) / 8, [owner](void*) {});
    for (std::int64_t i = 0; i < arr->n_children; ++i) {
        Series child = import_any(schema->children[i], arr->children[i], owner);
        if (!child.valid()) {
            delete col;
            return Series{};
        }
        col->children.push_back(std::shared_ptr<dftu_series>(child.release()));
        col->field_names.emplace_back(
            schema->children[i]->name ? schema->children[i]->name : "");
    }
    return Series{col};
}

// Dispatch on the Arrow type: List/Struct nest recursively, everything else
// (flat fixed-width and variable-width) through import_flat.
Series import_any(const ArrowSchema* schema, const ArrowArray* arr,
                  std::shared_ptr<void> owner) {
    ArrowSchemaView view;
    ArrowError error;
    if (ArrowSchemaViewInit(&view, schema, &error) != NANOARROW_OK)
        return Series{};
    if (view.type == NANOARROW_TYPE_LIST)
        return import_list(schema, arr, owner);
    if (view.type == NANOARROW_TYPE_STRUCT)
        return import_struct(schema, arr, owner);
    return import_flat(schema, arr, owner);
}

// Move `array` into a shared owner that runs its release once the last wrapped
// buffer is dropped; the caller's array is marked released.
std::shared_ptr<ArrowArray> adopt_array(ArrowArray* array) {
    auto* moved = new ArrowArray(*array);
    array->release = nullptr;
    return std::shared_ptr<ArrowArray>(moved, [](ArrowArray* a) {
        if (a->release != nullptr) a->release(a);
        delete a;
    });
}

}  // namespace

Series from_arrow(const ArrowSchema* schema, ArrowArray* array) {
    std::shared_ptr<ArrowArray> owner = adopt_array(array);
    return import_any(schema, owner.get(), owner);
}

DataFrame dataframe_from_arrow(const ArrowSchema* schema, ArrowArray* array) {
    ArrowSchemaView view;
    ArrowError error;
    if (ArrowSchemaViewInit(&view, schema, &error) != NANOARROW_OK)
        return DataFrame{};
    if (view.type != NANOARROW_TYPE_STRUCT) return DataFrame{};

    std::shared_ptr<ArrowArray> owner = adopt_array(array);
    DataFrame df;
    df.names.reserve(static_cast<std::size_t>(schema->n_children));
    df.columns.reserve(static_cast<std::size_t>(schema->n_children));
    for (std::int64_t i = 0; i < schema->n_children; ++i) {
        const ArrowSchema* cs = schema->children[i];
        df.names.emplace_back(cs->name != nullptr ? cs->name : "");
        df.columns.push_back(import_any(cs, owner->children[i], owner));
    }
    return df;
}

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_ENABLE_ARROW
