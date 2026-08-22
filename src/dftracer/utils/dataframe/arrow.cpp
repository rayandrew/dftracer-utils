#include <dftracer/utils/dataframe/arrow.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/dataframe/arrow_bridge.h>

#include <string>
#include <utility>
#include <vector>

namespace dftracer::utils::dataframe {

OwnedArrow Series::to_arrow() const {
    OwnedArrow out;
    dftracer::utils::dataframe::to_arrow(*this, out.schema(), out.array());
    return out;
}

Series Series::from_arrow(const ArrowSchema* schema, const ArrowArray* array) {
    return dftracer::utils::dataframe::from_arrow(
        schema, const_cast<ArrowArray*>(array));
}

OwnedArrow DataFrame::to_arrow() const {
    std::vector<std::string> field_names = names;
    std::vector<Series> field_columns;
    field_columns.reserve(columns.size());
    for (const Series& col : columns) field_columns.push_back(col.share());
    Series st =
        Series::structs(std::move(field_names), std::move(field_columns));
    OwnedArrow out;
    dftracer::utils::dataframe::to_arrow(st, out.schema(), out.array());
    return out;
}

DataFrame DataFrame::from_arrow(const ArrowSchema* schema,
                                const ArrowArray* array) {
    return dftracer::utils::dataframe::dataframe_from_arrow(
        schema, const_cast<ArrowArray*>(array));
}

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_ENABLE_ARROW
