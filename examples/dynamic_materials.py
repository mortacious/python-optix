import os, sys, logging, collections

import numpy as np
import cupy as cp
import optix as ox
import glfw

from optix.sutils.gui import init_ui
from optix.sutils.camera import Camera
from optix.sutils.gl_display import GLDisplay
from optix.sutils.cuda_output_buffer import CudaOutputBuffer, CudaOutputBufferType

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger()

script_dir = os.path.dirname(os.path.abspath(__file__))


class Params:
    _struct = collections.OrderedDict([
            ('image', 'u8'),
            ('image_width', 'u4'),
            ('image_height', 'u4'),
            ('radius', 'f4'),
            ('cam_eye', '3f4'),
            ('camera_u', '3f4'),
            ('camera_v', '3f4'),
            ('camera_w', '3f4'),
            ('trav_handle', 'u8'),
        ])

    @classmethod
    def names(cls):
        return tuple(cls._struct.keys())

    @classmethod
    def formats(cls):
        return tuple(cls._struct.values())
    
    def __init__(self):
        self.handle = ox.LaunchParamsRecord(names=self.names(), formats=self.formats())

    def __getattribute__(self, name):
        if name in Params._struct.keys():
            return self.__dict__['handle'][name]
        elif name in {'names', 'formats', 'handle', '__dict__'}:
            return super().__getattribute__(name)
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in Params._struct.keys():
            self.handle[name] = value
        elif name in {'handle'}:
            super().__setattr__(name, value)
        else:
            raise AttributeError(name)


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
    
        raygen_sbt = None
        miss_sbt = None
        hit_sbts = None
        sbt = None

        self.pipeline = None
        self.pipeline_opts = None

    @property
    def dimensions(self):
        return (int(self.params.image_height), int(self.params.image_width))


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

def key_callback(window, key, scancode, action, mods):
    if action == GLFW_PRESS:
        if key in {GLFW_KEY_Q, GLFW_KEY_ESCAPE}:
            glfwSetWindowShouldClose(window, True)
        elif key == GLFW_KEY_LEFT:
            g_has_data_changed = True 
        elif key == GLFW_KEY_RIGHT:
            g_has_sbt_changed = True
        elif key == GLFW_KEY_UP:
            g_has_offset_changed = True


# Transforms for instances - one on the left (sphere 0), one in the center and one on the right (sphere 2).
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

# Left sphere
g_material_index_0 = MaterialIndex(3)
g_has_data_changed = False;

# Middle sphere
g_material_index_1 = MaterialIndex(2)
g_has_offset_changed = False;

# Right sphere
g_material_index_2 = MaterialIndex(3)
g_has_sbt_changed = False;

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
    build_input = ox.BuildInputCustomPrimitiveArray(aabb_buffers=aabb, num_sbt_records=1)
    state.gas = ox.AccelerationStructure(state.ctx, build_input, compact=True)
    state.params.radius = 1.5

def build_ias(state):
    instances = []
    for i in range(transforms.shape[0]):
        instance = ox.Instance(traversable=state.gas, instance_id=0, flags=ox.InstanceFlags.NONE,
                sbt_offset=sbt_offsets[i], transform=transforms[i], visibility_mask=1)
        instances.append(instance)

    build_input = ox.BuildInputInstanceArray(instances)
    state.ias = ox.AccelerationStructure(context=state.ctx, build_inputs=build_input)
    state.params.trav_handle = state.ias.handle

def create_module(state):
    pipeline_opts = ox.PipelineCompileOptions(
            uses_motion_blur=False,
            traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_ANY,
            num_payload_values=3,
            num_attribute_values=3,
            exception_flags=ox.ExceptionFlags.DEBUG | ox.ExceptionFlags.TRACE_DEPTH | ox.ExceptionFlags.STACK_OVERFLOW,
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

    # The left sphere has a single CH program
    # The middle sphere toggles between two CH programs
    # The right sphere uses the g_material_index_2.index'th of these CH programs
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

    hit_groups = [hit_grps[0], hit_grps[1], hit_grps[2], hit_grps[g_material_index_2.index + 3]]
    hit_sbts = ox.SbtRecord(hit_groups, names=('color', 'idx'), formats=('3f4', 'u4'))

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

    state.raygen_sbt = raygen_sbt
    state.miss_sbt = miss_sbt
    state.hit_sbts = hit_sbts

    state.sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt,
            hitgroup_records=hit_sbts)


def update_state(output_buffer, state):
    # Change the material properties using one of three different approaches.
    if g_has_data_changed:
        update_hit_group_data(state)
    if g_has_offset_changed:
        update_instance_offset(state)
    if g_has_sbt_changed:
        update_sbt_header(state)

def update_hit_group_data(state):
    # Method 1:
    # Change the material parameters for the left sphere by directly modifying
    # the HitGroupData for the first SBT record.

    # Cycle through three base colors.
    material_idx = g_material_index_0.nextval()

    # Update the data field of the SBT record for the left sphere with the new base color.
    state.hit_sbts['colors'][0] = g_colors[material_index]
    state.sbt = ox.ShaderBindingTable(raygen_record=state.raygen_sbt, miss_records=state.miss_sbt,
            hitgroup_records=state.hit_sbts)

    g_has_data_changed = False

def update_instance_offset(state):
    # Method 2:
    # Update the SBT offset of the middle sphere. The offset is used to select
    # an SBT record during traversal, which dertermines the CH & AH programs
    # that will be invoked for shading.

    material_index = g_material_index_1.nextval()
    sbt_offsets[1] = 1 + material_index

    # It's necessary to rebuild the IAS for the updated offset to take effect.
    build_ias(state)

    g_has_offset_changed = False

def update_sbt_header(state):
    # Method 3:
    # Select a new material by re-packing the SBT header for the right sphere
    # with a different CH program.

    # The right sphere will use the next compiled program group.
    material_index = g_material_index_2.nextval()
    
    state.hit_groups.update_program_group(3, hit_grps[3 + material_index])
    
    state.sbt = ox.ShaderBindingTable(raygen_record=state.raygen_sbt, miss_records=state.miss_sbt,
            hitgroup_records=state.hit_sbts)

def launch(state, output_buffer):
    state.params.image = output_buffer.map()
    
    state.pipeline.launch(state.sbt, dimensions=state.dimensions, params=state.params.handle, stream=output_buffer.stream)

    output_buffer.unmap()
    

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

    window = init_ui("optixDynamicMaterials", state.params.image_width, state.params.image_height)

    output_buffer = CudaOutputBuffer(output_buffer_type, 'uchar4', state.params.image_width, state.params.image_height)
    glfw.set_key_callback(window, key_callback)
                
    gl_display = GLDisplay()

    while not glfw.window_should_close(window):
        glfw.poll_events()
        update_state(output_buffer, state)
        #launch(state, output_buffer)
        display(output_buffer, gl_display, window)
        display_usage()
        glfw.swap_buffers(window)

