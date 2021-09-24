# distutils: language = c++

from .common cimport optix_check_return, optix_init
from .context cimport DeviceContext
from .module cimport Module
from libc.string cimport memset

optix_init()

cdef class ProgramGroup(OptixObject):
    def __init__(self,
                 DeviceContext context,
                 Module raygen_module=None,
                 str raygen_entry_function_name=None,
                 Module miss_module=None,
                 str miss_entry_function_name=None,
                 Module exception_module=None,
                 str exception_entry_function_name=None,
                 Module callables_module_DC=None,
                 str callables_entry_function_name_DC=None,
                 Module callables_module_CC=None,
                 str callables_entry_function_name_CC=None,
                 Module hitgroup_module_CH=None,
                 str hitgroup_entry_function_name_CH=None,
                 Module hitgroup_module_AH=None,
                 str hitgroup_entry_function_name_AH=None,
                 Module hitgroup_module_IS=None,
                 str hitgroup_entry_function_name_IS=None
                 ):
        super().__init__(context)
        cdef bytes tmp_entry_function_name_1
        cdef bytes tmp_entry_function_name_2
        cdef bytes tmp_entry_function_name_3
        cdef OptixProgramGroupDesc desc
        memset(&desc, 0, sizeof(desc))
        desc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE

        if raygen_entry_function_name is not None:
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN
            if raygen_module is None:
                raise ValueError("raygen entry function specified but no module given")
            desc.raygen.module = raygen_module._module
            tmp_entry_function_name_1 = raygen_entry_function_name.encode('ascii')
            desc.raygen.entryFunctionName = tmp_entry_function_name_1
        elif miss_entry_function_name is not None:
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS
            if miss_module is None:
                raise ValueError("miss entry function specified but no module given")
            desc.miss.module = miss_module._module
            tmp_entry_function_name_1 = miss_entry_function_name.encode('ascii')
            desc.miss.entryFunctionName = tmp_entry_function_name_1
        elif exception_entry_function_name is not None:
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION
            if exception_module is None:
                raise ValueError("exception entry function specified but no module given")
            desc.exception.module = exception_module._module
            tmp_entry_function_name_1 = exception_entry_function_name.encode('ascii')
            desc.exception.entryFunctionName = tmp_entry_function_name_1
        elif callables_entry_function_name_DC or callables_entry_function_name_CC:
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES
            if callables_entry_function_name_DC is not None:
                if callables_module_DC is None:
                    raise ValueError("callables DC entry function specified but no module given")
                else:
                    desc.callables.moduleDC = callables_module_DC._module
                    tmp_entry_function_name_1 = callables_entry_function_name_DC.encode('ascii')
                    desc.callables.entryFunctionNameDC = tmp_entry_function_name_1
            if callables_entry_function_name_CC is not None:
                if callables_module_CC is None:
                    raise ValueError("callables CC entry function specified but no module given")
                else:
                    desc.callables.moduleCC = callables_module_CC._module
                    tmp_entry_function_name_2 = callables_entry_function_name_CC.encode('ascii')
                    desc.callables.entryFunctionNameCC = tmp_entry_function_name_2
        elif hitgroup_module_CH is not None or hitgroup_module_AH is not None or hitgroup_module_IS is not None:
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP
            if hitgroup_entry_function_name_CH is not None:
                if hitgroup_module_CH is None:
                    raise ValueError("hitgroup CH entry function specified but no module given")
                else:
                    desc.hitgroup.moduleCH = hitgroup_module_CH._module
                    tmp_entry_function_name_1 = hitgroup_entry_function_name_CH.encode('ascii')
                    desc.hitgroup.entryFunctionNameCH = tmp_entry_function_name_1

            if hitgroup_entry_function_name_AH is not None:
                if hitgroup_module_AH is None:
                    raise ValueError("hitgroup AH entry function specified but no module given")
                else:
                    desc.hitgroup.moduleAH = hitgroup_module_AH._module
                    tmp_entry_function_name_2 = hitgroup_entry_function_name_AH.encode('ascii')
                    desc.hitgroup.entryFunctionNameAH = tmp_entry_function_name_2
            if hitgroup_entry_function_name_IS is not None:
                if hitgroup_module_IS is None:
                    raise ValueError("hitgroup IS entry function specified but no module given")
                else:
                    desc.hitgroup.moduleIS = hitgroup_module_IS._module
                    tmp_entry_function_name_3 = hitgroup_entry_function_name_IS.encode('ascii')
                    desc.hitgroup.entryFunctionNameIS = tmp_entry_function_name_3

        cdef OptixProgramGroupOptions options # init to zero
        memset(&options, 0, sizeof(options))
        optix_check_return(optixProgramGroupCreate(self.context.c_context, &desc, 1, &options, NULL, NULL, &self._program_group))

    def __dealloc__(self):
        if <size_t>self._program_group != 0:
            optix_check_return(optixProgramGroupDestroy(self._program_group))

    @classmethod
    def create_raygen(cls, DeviceContext context, Module module, str entry_function_name):
        return cls(context, raygen_module=module, raygen_entry_function_name=entry_function_name)

    @classmethod
    def create_miss(cls, DeviceContext context, Module module, str entry_function_name):
        return cls(context, miss_module=module, miss_entry_function_name=entry_function_name)

    @classmethod
    def create_exception(cls, DeviceContext context, Module module, str entry_function_name):
        return cls(context, exception_module=module, exception_entry_function_name=entry_function_name)

    @classmethod
    def create_callables(cls, DeviceContext context, Module module_DC=None, str entry_function_DC=None, Module module_CC=None, str entry_function_CC=None, share_module=True):
        # implicitly reuse the module
        if share_module:
            if module_CC is None and module_DC is not None:
                module_CC = module_DC
            elif module_DC is None and module_CC is not None:
                module_DC = module_CC

        return cls(context,
                   callables_module_DC=module_DC,
                   callables_entry_function_name_DC=entry_function_DC,
                   callables_module_CC=module_CC,
                   callables_entry_function_name_CC=entry_function_CC)

    @classmethod
    def create_hitgroup(cls,
                        DeviceContext context,
                        Module module_CH=None,
                        str entry_function_CH=None,
                        Module module_AH=None,
                        str entry_function_AH=None,
                        Module module_IS=None,
                        str entry_function_IS=None,
                        share_module=True):
        if share_module:
            for module in [module_CH, module_AH, module_IS]:
                if module is not None:
                    module_CH = module
                    module_AH = module
                    module_IS = module
        return cls(context,
                   hitgroup_module_CH=module_CH,
                   hitgroup_entry_function_name_CH=entry_function_CH,
                   hitgroup_module_AH=module_AH,
                   hitgroup_entry_function_name_AH=entry_function_AH,
                   hitgroup_module_IS=module_IS,
                   hitgroup_entry_function_name_IS=entry_function_IS)

    @property
    def stack_sizes(self):
        cdef OptixStackSizes stack_sizes
        optix_check_return(optixProgramGroupGetStackSize(self._program_group, &stack_sizes))
        return stack_sizes.cssRG, stack_sizes.cssMS, stack_sizes.cssCH, stack_sizes.cssAH, stack_sizes.cssIS, stack_sizes.cssCC, stack_sizes.dssDC
