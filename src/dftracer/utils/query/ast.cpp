#include <dftracer/utils/query/fields.h>

#include <type_traits>
#include <variant>

namespace dftracer::utils::query {

namespace {

void collect_fields_impl(const QueryNode& node,
                         dftracer::utils::StringViewSet& out) {
    std::visit(
        [&out](auto&& n) {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, CompareNode>) {
                out.insert(n.field.path);
            } else if constexpr (std::is_same_v<T, InNode>) {
                out.insert(n.field.path);
            } else if constexpr (std::is_same_v<T, NotInNode>) {
                out.insert(n.field.path);
            } else if constexpr (std::is_same_v<T, MatchNode>) {
                out.insert(n.field.path);
            } else if constexpr (std::is_same_v<T, AndNode>) {
                collect_fields_impl(*n.left, out);
                collect_fields_impl(*n.right, out);
            } else if constexpr (std::is_same_v<T, OrNode>) {
                collect_fields_impl(*n.left, out);
                collect_fields_impl(*n.right, out);
            } else if constexpr (std::is_same_v<T, NotNode>) {
                collect_fields_impl(*n.operand, out);
            }
        },
        node.data);
}

}  // namespace

dftracer::utils::StringViewSet collect_fields(const QueryNode& node) {
    dftracer::utils::StringViewSet fields;
    collect_fields_impl(node, fields);
    return fields;
}

}  // namespace dftracer::utils::query
