#pragma once

#if defined(_MSC_VER)
#define NOMINMAX
#endif

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <sstream>

inline void optix_check_return(OptixResult result) {
        if( result != OPTIX_SUCCESS ) {
            std::stringstream ss;
            ss << optixGetErrorName(result);
            ss << ": " << optixGetErrorString(result);
            throw std::runtime_error(ss.str());
        }
}
