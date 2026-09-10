#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/types.h>

namespace dftracer::utils::dataframe {
namespace {

template <typename E>
constexpr std::int32_t code(E e) {
    return static_cast<std::int32_t>(e);
}

static_assert(code(CmpOp::Gt) == DFTU_CMP_GT);
static_assert(code(CmpOp::Ge) == DFTU_CMP_GE);
static_assert(code(CmpOp::Lt) == DFTU_CMP_LT);
static_assert(code(CmpOp::Le) == DFTU_CMP_LE);
static_assert(code(CmpOp::Eq) == DFTU_CMP_EQ);
static_assert(code(CmpOp::Ne) == DFTU_CMP_NE);

static_assert(code(LogicalOp::And) == DFTU_LOGICAL_AND);
static_assert(code(LogicalOp::Or) == DFTU_LOGICAL_OR);

static_assert(code(RankMethod::Average) == DFTU_RANK_AVERAGE);
static_assert(code(RankMethod::Min) == DFTU_RANK_MIN);
static_assert(code(RankMethod::Dense) == DFTU_RANK_DENSE);
static_assert(code(RankMethod::Ordinal) == DFTU_RANK_ORDINAL);

static_assert(code(RollingOp::Sum) == DFTU_ROLLING_SUM);
static_assert(code(RollingOp::Mean) == DFTU_ROLLING_MEAN);
static_assert(code(RollingOp::Min) == DFTU_ROLLING_MIN);
static_assert(code(RollingOp::Max) == DFTU_ROLLING_MAX);

static_assert(code(PrimOp::Ilog2) == DFTU_PRIM_ILOG2);
static_assert(code(PrimOp::BitWidth) == DFTU_PRIM_BIT_WIDTH);
static_assert(code(PrimOp::Popcount) == DFTU_PRIM_POPCOUNT);
static_assert(code(PrimOp::Clz) == DFTU_PRIM_CLZ);
static_assert(code(PrimOp::Ctz) == DFTU_PRIM_CTZ);
static_assert(code(PrimOp::Mix64) == DFTU_PRIM_MIX64);

static_assert(code(ScalarTag::I64) == DFTU_SCALAR_TAG_I64);
static_assert(code(ScalarTag::U64) == DFTU_SCALAR_TAG_U64);
static_assert(code(ScalarTag::F64) == DFTU_SCALAR_TAG_F64);

// Every TypeId mirrors a dftu_dtype, in lockstep, so a plugin can name any
// type a Series can hold through the C ABI.
static_assert(code(TypeId::Unknown) == DFTU_TYPE_UNKNOWN);
static_assert(code(TypeId::Bool) == DFTU_TYPE_BOOL);
static_assert(code(TypeId::Int8) == DFTU_TYPE_INT8);
static_assert(code(TypeId::Int16) == DFTU_TYPE_INT16);
static_assert(code(TypeId::Int32) == DFTU_TYPE_INT32);
static_assert(code(TypeId::Int64) == DFTU_TYPE_INT64);
static_assert(code(TypeId::Uint8) == DFTU_TYPE_UINT8);
static_assert(code(TypeId::Uint16) == DFTU_TYPE_UINT16);
static_assert(code(TypeId::Uint32) == DFTU_TYPE_UINT32);
static_assert(code(TypeId::Uint64) == DFTU_TYPE_UINT64);
static_assert(code(TypeId::Float32) == DFTU_TYPE_FLOAT32);
static_assert(code(TypeId::Float64) == DFTU_TYPE_FLOAT64);
static_assert(code(TypeId::String) == DFTU_TYPE_STRING);
static_assert(code(TypeId::Binary) == DFTU_TYPE_BINARY);
static_assert(code(TypeId::List) == DFTU_TYPE_LIST);
static_assert(code(TypeId::Struct) == DFTU_TYPE_STRUCT);
static_assert(code(TypeId::Float16) == DFTU_TYPE_FLOAT16);
static_assert(code(TypeId::Date32) == DFTU_TYPE_DATE32);
static_assert(code(TypeId::Date64) == DFTU_TYPE_DATE64);
static_assert(code(TypeId::Time32) == DFTU_TYPE_TIME32);
static_assert(code(TypeId::Time64) == DFTU_TYPE_TIME64);
static_assert(code(TypeId::Timestamp) == DFTU_TYPE_TIMESTAMP);
static_assert(code(TypeId::Duration) == DFTU_TYPE_DURATION);
static_assert(code(TypeId::Decimal128) == DFTU_TYPE_DECIMAL128);
static_assert(code(TypeId::Decimal256) == DFTU_TYPE_DECIMAL256);
static_assert(code(TypeId::FixedSizeBinary) == DFTU_TYPE_FIXED_SIZE_BINARY);
static_assert(code(TypeId::LargeString) == DFTU_TYPE_LARGE_STRING);
static_assert(code(TypeId::LargeBinary) == DFTU_TYPE_LARGE_BINARY);
static_assert(code(TypeId::LargeList) == DFTU_TYPE_LARGE_LIST);
static_assert(code(TypeId::FixedSizeList) == DFTU_TYPE_FIXED_SIZE_LIST);
static_assert(code(TypeId::Map) == DFTU_TYPE_MAP);
// TypeId::Map is declared last in types.h, so its ordinal is the count of
// TypeId values minus one; pinning DFTU_TYPE_MAP to that ordinal makes a
// TypeId appended without a matching dftu_dtype member fail this assert
// instead of silently reusing 30.
static_assert(DFTU_TYPE_MAP == 30,
              "a TypeId was added without mirroring it into dftu_dtype");

static_assert(code(TimeUnit::Second) == DFTU_TIME_UNIT_SECOND);
static_assert(code(TimeUnit::Milli) == DFTU_TIME_UNIT_MILLI);
static_assert(code(TimeUnit::Micro) == DFTU_TIME_UNIT_MICRO);
static_assert(code(TimeUnit::Nano) == DFTU_TIME_UNIT_NANO);

}  // namespace
}  // namespace dftracer::utils::dataframe
