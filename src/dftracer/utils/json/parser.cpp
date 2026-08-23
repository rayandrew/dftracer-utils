#include <dftracer/utils/json/parser.h>

namespace dftracer::utils::json {

JsonParser::JsonParser(std::size_t capacity) : parser_(capacity) {}

bool JsonParser::parse(std::string_view json_line) {
    // Reuse padbuf_'s capacity instead of allocating a padded_string per line;
    // resize() zero-fills the SIMDJSON_PADDING tail On-Demand requires.
    padbuf_.assign(json_line.data(), json_line.size());
    padbuf_.resize(json_line.size() + simdjson::SIMDJSON_PADDING);
    auto result =
        parser_.iterate(padbuf_.data(), json_line.size(), padbuf_.size());
    if (result.error()) {
        valid_ = false;
        return false;
    }
    doc_ = std::move(result.value());
    active_ = simdjson::ondemand::document_reference(doc_);
    valid_ = true;
    return true;
}

void JsonParser::rewind() {
    if (valid_) {
        active_.rewind();
    }
}

std::optional<std::int64_t> JsonParser::get_int64(std::string_view key) {
    if (!valid_) return std::nullopt;
    auto result = active_[key].get_int64();
    if (result.error()) return std::nullopt;
    return result.value();
}

std::optional<std::uint64_t> JsonParser::get_uint64(std::string_view key) {
    if (!valid_) return std::nullopt;
    auto result = active_[key].get_uint64();
    if (result.error()) return std::nullopt;
    return result.value();
}

std::optional<double> JsonParser::get_double(std::string_view key) {
    if (!valid_) return std::nullopt;
    auto result = active_[key].get_double();
    if (result.error()) return std::nullopt;
    return result.value();
}

std::optional<bool> JsonParser::get_bool(std::string_view key) {
    if (!valid_) return std::nullopt;
    auto result = active_[key].get_bool();
    if (result.error()) return std::nullopt;
    return result.value();
}

std::optional<std::string_view> JsonParser::get_string(std::string_view key) {
    if (!valid_) return std::nullopt;
    auto result = active_[key].get_string();
    if (result.error()) return std::nullopt;
    return result.value();
}

std::optional<simdjson::ondemand::value> JsonParser::get_value(
    std::string_view key) {
    if (!valid_) return std::nullopt;
    auto result = active_[key];
    if (result.error()) return std::nullopt;
    return result.value();
}

}  // namespace dftracer::utils::json
