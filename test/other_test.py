import optix
import cupy as cp

example_cuda_program = \
    """
    //
    // Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
    //
    // Redistribution and use in source and binary forms, with or without
    // modification, are permitted provided that the following conditions
    // are met:
    //  * Redistributions of source code must retain the above copyright
    //    notice, this list of conditions and the following disclaimer.
    //  * Redistributions in binary form must reproduce the above copyright
    //    notice, this list of conditions and the following disclaimer in the
    //    documentation and/or other materials provided with the distribution.
    //  * Neither the name of NVIDIA CORPORATION nor the names of its
    //    contributors may be used to endorse or promote products derived
    //    from this software without specific prior written permission.
    //
    // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
    // EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    // PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
    // CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    // EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    // PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    // PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
    // OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    // (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    //

    #include <optix.h>

    struct Params
    {
        uchar4* image;
        unsigned int image_width;
    };

    struct RayGenData
    {
        float r,g,b;
    };

    extern "C" {
    __constant__ Params params;
    }

    extern "C"
    __global__ void __raygen__hello()
    {
        uint3 launch_index = optixGetLaunchIndex();
        RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
        params.image[launch_index.y * params.image_width + launch_index.x] =
            make_uchar4( 
                        max( 0.0f, min( 255.0f, rtData->r*255.0f ) ), 
                        max( 0.0f, min( 255.0f, rtData->g*255.0f ) ),
                        max( 0.0f, min( 255.0f, rtData->b*255.0f ) ),
                        255
                        );
    }

    extern "C"
    __global__ void __anyhit__noop() {}

    extern "C"
    __global__ void __closesthit__noop() {}

    extern "C"
    __global__ void ___intersection__noop() {}

    extern "C"
    __global__ void ___intersect__noop() {}

    extern "C"
    __global__ void ___miss__noop() {}

    extern "C"
    __global__ void ___direct_callable__noop() {}

    extern "C"
    __global__ void ___continuation_callable__noop() {}
    """

def compile_cuda( src ):
    from pynvrtc.compiler import Program
    prog = Program( src, "default_program" )
    ptx  = prog.compile( [
        '-use_fast_math',
        '-lineinfo',
        '-default-device',
        '-std=c++11',
        '-rdc',
        'true',
        #'-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v11.1\include'
        '-I/home/mortacious/.conda/envs/hyperspace/include',
        f'-I/opt/optix/include'
        ] )
    return ptx

def optix_init():
    cp.cuda.runtime.free( 0 )
    optix.init()


def create_default_ctx():
    optix_init()
    ctx_options = optix.DeviceContextOptions()

    cu_ctx = 0
    return optix.deviceContextCreate( cu_ctx, ctx_options )


def test_create_destroy():
    ctx = create_default_ctx()

    module_opts = optix.ModuleCompileOptions()
    pipeline_opts = optix.PipelineCompileOptions()

    ptx = compile_cuda(example_cuda_program)
    print(ptx, len(ptx))
    mod, log = ctx.moduleCreateFromPTX(
        module_opts,
        pipeline_opts,
        ptx
    )
    assert type(mod) is optix.Module
    assert type(log) is str

    mod.destroy()
    ctx.destroy()

if __name__ == "__main__":
    test_create_destroy()