import os, sys, logging

import numpy as np
import cupy as cp
import optix as ox

from optix.sutils.cuda_output_buffer import CudaOutputBufferType
from optix.sutils.camera import Camera

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger()

script_dir = os.path.dirname(os.path.abspath(__file__))

class Params:
    def __init__(self):
        self.image = None
        self.image_width = None
        self.image_height = None
        self.radius = None
        self.handle = None
        self.cam_eye = None
        self.camera_u = None
        self.camera_v = None
        self.camera_w = None
        self.hit_group_record_idx_0 = None
        self.hit_group_record_stride = None

class SampleState:
    def __init__(self, width, height):
        self.params = Params()
        self.params.image_width = width
        self.params.image_height = height

        self.ctx = None
        self.gas = None
        self.ias = None
        self.module = None

        self.raygen_grp = None
        self.miss_grp = None
        self.hit_grps = None

        self.pipeline = None
        self.pipeline_opts = None


# Transforms for instances - one on the left (sphere 0), one in the centre and one on the right (sphere 2).
transforms = np.asarray([
    [1, 0, 0, -6, 
     0, 1, 0, 0, 
     0, 0, 1, -10],
    [1, 0, 0, 0, 
     0, 1, 0, 0, 
     0, 0, 1, -10],
    [1, 0, 0, 6, 
     0, 1, 0, 0, 
     0, 0, 1, -10],
], dtype=np.float32).reshape(3,3,4)

# Offsets into SBT for each instance. Hence this needs to be in sync with transforms!
# The middle sphere has two SBT records, the two other instances have one each.
sbt_offsets = np.asarray([0, 1, 3], dtype=np.uint32)

g_colors = np.asarray([[1, 0, 0],  
                       [0, 1, 0],
                       [0, 0, 1]], dtype=np.float32)

class MaterialIndex:
    def __init__(self, max_index):
        self._index = 0
        self._max_index = max_index

    def _get_index(self):
        return self._index
    def _set_index(self, value):
        assert value >= 0, value
        self._index = np.uint32(value % self._max_index)
    index = property(_get_index, _set_index)

    def nextval(self):
        self.index = self.index + 1
        return self.index

# Left sphere
g_materialIndex_0 = MaterialIndex(3)
g_hasDataChanged = False;

# Middle sphere
g_materialIndex_1 = MaterialIndex(2)
g_hasOffsetChanged = False;

# Right sphere
g_materialIndex_2 = MaterialIndex(3)
g_hasSbtChanged = False;

##------------------------------------------------------------------------------
##
## Helper Functions
##
##------------------------------------------------------------------------------

def init_camera(state):
    camera = Camera()
    camera.eye = (0, 0, 3)
    camera.look_at = (0, 0, 0)
    camera.up = (0, 1, 0)
    camera.fov_y = 60
    camera.aspect_ratio = state.params.image_width  / state.params.image_height

    u,v,w = camera.uvw_frame()
    state.params.camera_u = u
    state.params.camera_v = v
    state.params.camera_w = w
    state.params.cam_eye = camera.eye

def create_context(state):
    logger = ox.Logger(log)
    ctx = ox.DeviceContext(validation_mode=True, log_callback_function=logger, log_callback_level=4)
    ctx.cache_enabled = False
    state.ctx = ctx

def build_gas(state):
    aabb = np.asarray([[-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]], dtype=np.float32)
    build_input = ox.BuildInputCustomPrimitiveArray(aabb)
    state.gas = ox.AccelerationStructure(state.ctx, build_input, compact=True)

def build_ias(state):
    instances = []
    for i in range(transforms.shape[0]):
        instance = ox.Instance(traversable=state.gas, instance_id=0, flags=ox.InstanceFlags.NONE,
                sbt_offset=sbt_offsets[i], transform=transforms[i], visibility_mask=1)
        instances.append(instance)

    build_input = ox.BuildInputInstanceArray(instances)
    state.ias = ox.AccelerationStructure(state.ctx, build_input)

def create_module(state):
    pipeline_opts = ox.PipelineCompileOptions(
            uses_motion_blur=False,
            traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_ANY,
            num_payload_values=3,
            num_attribute_values=3,
            exception_flags=ox.ExceptionFlags.NONE,
            pipeline_launch_params_variable_name="params")

    compile_opts = ox.ModuleCompileOptions(
            max_register_count=ox.ModuleCompileOptions.DEFAULT_MAX_REGISTER_COUNT,
            opt_level=ox.CompileOptimizationLevel.DEFAULT,
            debug_level=ox.CompileDebugLevel.LINEINFO)
    
    source = os.path.join(script_dir, 'cuda', 'dynamic_materials.cu')
    state.module = ox.Module(state.ctx, source, compile_opts, pipeline_opts)
    state.pipeline_opts = pipeline_opts

def create_program_groups(state):
    ctx, module = state.ctx, state.module

    state.raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    state.miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    
    # The left sphere has a single CH progra
    # The middle sphere toggles between two CH programs
    # The right sphere uses the g_materialIndex_2.getVal()'th of these CH programs
    ch_names = ('__closesthit__ch' , 
                '__closesthit__ch', '__closesthit__normal',
                '__closesthit__blue', '__closesthit__green', '__closesthit__red')
    
    hit_grps = []
    for ch_name in ch_names:
        hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module,
                                                  entry_function_CH=ch_name,
                                                  entry_function_IS='__intersection__is')
        hit_grps.append(hit_grp)
    state.hit_grps = hit_grps

def create_pipeline(state):
    program_grps = [state.raygen_grp, state.miss_grp] + state.hit_grps
    
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1,
                                       debug_level=ox.CompileDebugLevel.FULL)
    
    pipeline = ox.Pipeline(state.ctx,
                           compile_options=state.pipeline_opts,
                           link_options=link_opts,
                           program_groups=program_grps)
    
    pipeline.compute_stack_sizes(1,  # max_trace_depth
                                 0,  # max_cc_depth
                                 0)  # max_dc_depth

    state.pipeline = pipeline

def create_sbt(state):
    raygen_grp, miss_grp, hit_grps = state.raygen_grp, state.miss_grp, state.hit_grps

    raygen_sbt = ox.SbtRecord(raygen_grp)

    miss_sbt = ox.SbtRecord(miss_grp, names=('color',), formats=('3f4',))
    miss_sbt['color'] = [0.3, 0.1, 0.2]
    
    hit_sbts = ox.SbtRecord(hit_grps[0], names=('color', 'idx'), formats=('3f4', 'u4'), size=4)

    # The left sphere cycles through three colors by updating the data field of the SBT record.
    hit_sbts['color'][0] = g_colors[0]
    hit_sbts['idx'][0] = np.uint32(0)
    
    # The middle sphere toggles between two SBT records by adjusting the SBT
    # offset field of the sphere instance. The IAS needs to be rebuilt for the
    # update to take effect.
    hit_sbts['color'][1] = g_colors[1]
    hit_sbts['idx'][1] = np.uint32(1)
    
    hit_sbts['color'][2] = g_colors[1]
    hit_sbts['idx'][2] = np.uint32(1)
    
    # The right sphere cycles through colors by modifying the SBT. On update, a
    # different pre-built CH program is packed into the corresponding SBT
    # record.
    hit_sbts['color'][3] = [0,0,0]
    hit_sbts['idx'][3] = np.uint32(2)
    
    state.sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, 
            hitgroup_records=hit_sbts)


if __name__ == '__main__':
    state = SampleState(1024, 768)
    output_buffer_type = CudaOutputBufferType.CUDA_DEVICE

    init_camera(state)
    create_context(state)
    build_gas(state)
    build_ias(state)
    create_module(state)
    create_program_groups(state)
    create_pipeline(state)
    create_sbt(state)

