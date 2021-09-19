#pragma once

struct Params
{
    float3* points;
    float3* orientations;
    unsigned int* visible;
    float3 ray_direction;
    float tolerance;
    //float ray_max;
    OptixTraversableHandle handle;
};

struct DiscHitGroup {
    float3* centers;
    float3* orientations;
    float* radii;
    size_t chunk_size;
};