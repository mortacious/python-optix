#include <optix.h>

#include "vec_math.h"
#include "helpers.h"
#include "disc.h"

#define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ bool intersectPlane(const float3 &normal, const float3 &center, const float3 &origin, const float3 &direction, float &t)
{
    // assuming vectors are all normalized
    float denom = dot(normal, direction);
    if (denom > 1e-6f) {
        float3 p0l0 = center- origin;
        t = dot(p0l0, normal) / denom;
        return (t >= 0);
    }

    return false;
}

extern "C" __global__ void __intersection__disc()
{
    const DiscHitGroup* hit_group_data = reinterpret_cast<DiscHitGroup*>( optixGetSbtDataPointer() );
    const unsigned int gas_index = optixGetInstanceId();
    // get the actual primitive id for it has been split between several gas structures
    const unsigned int primitive_index = gas_index * hit_group_data->chunk_size + optixGetPrimitiveIndex();


    float t = 0;
    const float3 disc_center = hit_group_data->centers[primitive_index];
    const float3 disc_orientation = hit_group_data->orientations[primitive_index];
    const float disc_radius = hit_group_data->radii[primitive_index];
    //printf("Intersecting disc at %d center (%f, %f, %f), radius %f\n", gas_index, disc_center.x, disc_center.y, disc_center.z, disc_radius);

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    if (intersectPlane(disc_orientation, disc_center, ray_orig, ray_dir, t)) {
        if(t > ray_tmin && t < ray_tmax) {
            float3 p = ray_orig + ray_dir * t;
            float3 v = p - disc_center;
            float d2 = dot(v, v);
            float radius2 = disc_radius * disc_radius;
            if(d2 <= radius2) {

                optixReportIntersection( t, 0, float_as_int( disc_radius ));
            }
        }
    }
}

static __forceinline__ __device__ unsigned int traceOcclusion(
        OptixTraversableHandle handle,
        unsigned int idx,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
)
{
    unsigned int visible = 0u;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, // hot only once
            0,      // SBT offset
            0,          // SBT stride
            0,      // missSBTIndex
            visible,
            idx);
    return visible;
}


extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const unsigned int idx = optixGetLaunchIndex().x;
    //const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    //unsigned int idx1d = idx.x;
    float3 origin = params.points[idx];
    float3 orientation = params.orientations[idx];
    float3 direction = normalize(params.ray_direction);

    float view_angle = dot(orientation, direction);
    unsigned int visible;

    if(view_angle >= 0.0) {
        visible = traceOcclusion(params.handle, idx, origin, direction, params.tolerance, 1e16f);
    } else {
        visible = 0; // point is invisible since the point direction and ray direction do not match
    }

    params.visible[idx] = visible;
}


extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(1);
}

