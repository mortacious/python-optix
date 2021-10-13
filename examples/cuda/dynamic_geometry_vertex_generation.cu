
#include "vec_math.h"

enum struct AnimationMode: int
{
    NONE = 0,
    DEFORM = 1,
    EXPLODE = 2,
};


__forceinline__ __device__ float triangle_wave( float x, float shift = 0.f, float period = 2.f * M_PIf, float amplitude = 1.f )
{
    return fabsf( fmodf( ( 4.f / period ) * ( x - shift ), 4.f * amplitude ) - 2.f * amplitude ) - amplitude;
}

__forceinline__ __device__ void write_animated_triangle( float3* out_vertices, int tidx, float3 v0, float3 v1, float3 v2, AnimationMode mode, float time )
{
    float3 v = make_float3( 0 );

    if( mode == AnimationMode::EXPLODE )
    {
        // Generate displacement vector from triangle index
        const float theta = ( (float)M_PIf * ( ( tidx + 1 ) * ( 13 / M_PIf ) ) );
        const float phi   = ( (float)( 2.0 * M_PIf ) * ( ( tidx + 1 ) * ( 97 / M_PIf ) ) );

        // Apply displacement to the sphere triangles
        v = make_float3( triangle_wave( phi ) * triangle_wave( theta, M_PIf / 2.f ),
            triangle_wave( phi, M_PIf / 2.f ) * triangle_wave( theta, M_PIf / 2.f ), triangle_wave( theta ) )
            * triangle_wave( time, M_PIf / 2.f ) * 2.f;
    }

    out_vertices[tidx * 3 + 0] = v0 + v;
    out_vertices[tidx * 3 + 1] = v1 + v;
    out_vertices[tidx * 3 + 2] = v2 + v;
}

__forceinline__ __device__ float3 deform_vertex( const float3& c, AnimationMode mode, float time )
{
    // Apply sine wave to the y coordinate of the sphere vertices
    if( mode == AnimationMode::DEFORM )
        return make_float3( c.x, c.y * ( 0.5f + 0.4f * cosf( 4 * ( c.x + time ) ) ), c.z );
    return c;
}

extern "C" __global__ void generate_vertices(float3* out_vertices, AnimationMode mode, float time, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < width * height )
    {
        // generate a single patch (two unindexed triangles) of a tessellated sphere

        int x = idx % width;
        int y = idx / width;

        const float theta0 = ( ( float )M_PIf * ( y + 0 ) ) / height;
        const float theta1 = ( ( float )M_PIf * ( y + 1 ) ) / height;
        const float phi0 = ( ( float )( 2.0 * M_PIf ) * ( x + 0 ) ) / width;
        const float phi1 = ( ( float )( 2.0 * M_PIf ) * ( x + 1 ) ) / width;

        const float ct0 = cosf( theta0 );
        const float st0 = sinf( theta0 );
        const float ct1 = cosf( theta1 );
        const float st1 = sinf( theta1 );

        const float cp0 = cosf( phi0 );
        const float sp0 = sinf( phi0 );
        const float cp1 = cosf( phi1 );
        const float sp1 = sinf( phi1 );

        const float3 v00 = deform_vertex( make_float3( cp0 * st0, sp0 * st0, ct0 ), mode, time );
        const float3 v10 = deform_vertex( make_float3( cp0 * st1, sp0 * st1, ct1 ), mode, time );
        const float3 v01 = deform_vertex( make_float3( cp1 * st0, sp1 * st0, ct0 ), mode, time );
        const float3 v11 = deform_vertex( make_float3( cp1 * st1, sp1 * st1, ct1 ), mode, time );

        write_animated_triangle( out_vertices, idx * 2 + 0, v00, v10, v11, mode, time );
        write_animated_triangle( out_vertices, idx * 2 + 1, v00, v11, v01, mode, time );
    }
}
