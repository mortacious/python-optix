#include <optix.h>

#include "vec_math.h"
#include "helpers.h"
#include "sphere.h"

#define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )

extern "C" __global__ void __intersection__sphere()
{
    const sphere::SpheresHitGroupData* hit_group_data = reinterpret_cast<sphere::SpheresHitGroupData*>( optixGetSbtDataPointer() );

    const unsigned int primitive_index = optixGetPrimitiveIndex();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    const float3 O      = ray_orig - hit_group_data->centers[primitive_index];

    const float  l      = 1.0f / length( ray_dir );
    const float3 D      = ray_dir * l;
    const float  radius = hit_group_data->radii[primitive_index];

    float b    = dot( O, D );
    float c    = dot( O, O ) - radius * radius;
    float disc = b * b - c;
    if( disc > 0.0f )
    {
        float sdisc        = sqrtf( disc );
        float root1        = ( -b - sdisc );
        float root11       = 0.0f;
        bool  check_second = true;

        const bool do_refine = fabsf( root1 ) > ( 10.0f * radius );

        if( do_refine )
        {
            // refine root1
            float3 O1 = O + root1 * D;
            b         = dot( O1, D );
            c         = dot( O1, O1 ) - radius * radius;
            disc      = b * b - c;

            if( disc > 0.0f )
            {
                sdisc  = sqrtf( disc );
                root11 = ( -b - sdisc );
            }
        }

        float  t;
        float3 normal;
        t = ( root1 + root11 ) * l;
        if( t > ray_tmin && t < ray_tmax )
        {
            normal = ( O + ( root1 + root11 ) * D ) / radius;
            if( optixReportIntersection( t, 0, float3_as_ints( normal ), float_as_int( radius ) ) ) {
                //printf("Hit at %d\n", primitive_index);
                check_second = false;
            }
        }

        if( check_second )
        {
            float root2 = ( -b + sdisc ) + ( do_refine ? root1 : 0 );
            t           = root2 * l;
            normal      = ( O + root2 * D ) / radius;
            if( t > ray_tmin && t < ray_tmax ) {
                //printf("Hit at %d\n", primitive_index);
                optixReportIntersection(t, 0, float3_as_ints(normal), float_as_int(radius));
            }
        }
    }
}

extern "C" {
__constant__ Params params;
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
    const uint3 idx = optixGetLaunchIndex();
    //const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    unsigned int idx1d = idx.x;
    float3 origin = params.points[idx1d];
    //float3 ray_target = origin + params.ray_direction;
    float3 direction = normalize(params.ray_direction);
    unsigned int visible = traceOcclusion(params.handle, idx1d, origin, direction, params.tolerance, 1e16f);
    //atomicAdd(&(params.visible[idx1d]), visible);
    params.visible[idx1d] = visible;
}


extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(1);
}

