#pragma once

namespace sphere {
    const unsigned int NUM_ATTRIBUTE_VALUES = 2u;

    struct SpheresHitGroupData {
        float3* centers;
        float* radii;
    };
}

struct Params
{
    float3* points;
    unsigned int* visible;
    float3 ray_direction;
    float tolerance;
    OptixTraversableHandle handle;
};


struct MissData
{
    float r, g, b;
};
