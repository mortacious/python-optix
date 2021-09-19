//
// Created by mortacious on 7/29/21.
//

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
    //float ray_max;
    OptixTraversableHandle handle;
};


struct MissData
{
    float r, g, b;
};
