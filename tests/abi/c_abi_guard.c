/* Compiles the public ABI headers with a C compiler. A C++ leak into any of
   them breaks this build; the ABI must stay callable from pure C. */

#include <dftracer/utils/core/abi.h>
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/query/abi.h>

int dftracer_c_abi_guard(void) { return 0; }
