# distutils: language = c++

from .common cimport optix_check_return, optix_init
from .context cimport DeviceContext
from .module cimport Module
from libc.string cimport memset
from enum import IntEnum
optix_init()

class ProgramGroupKind(IntEnum):
    """
    Wraps the OptixProgramGroupKind enum
    """
    RAYGEN = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
    MISS = OPTIX_PROGRAM_GROUP_KIND_MISS,
    EXCEPTION = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION,
    HITGROUP = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
    CALLABLES = OPTIX_PROGRAM_GROUP_KIND_CALLABLES

cdef class ProgramGroup(OptixContextObject):
    """
    Represets a Programgroup, containing the entry functions and their module for one specific part
    of the pipeline (raygen, miss, exceptin, hitgroup and callables).

    One ProgramGroup can only represent a single part with multiple functions if necessary so the creation functions provided
    below are preferred over the generic __init__ function.

    Parameters
    ----------
    context: DeviceContext
        The context to use for this ProgramGroup.
    raygen_module: Module, optional
        The module containing the raygen function or None.
    raygen_entry_function_name: str, optional
        The name of the raygen function in the raygen module above. If this is not None the parameter raygen_module must
        also be present.
    miss_module: Module, optional
        The module containing the miss function or None.
    miss_entry_function_name: str, optional
        The name of the miss function in the miss module above. If this is not None the parameter miss_module must
        also be present.
    exception_module: Module, optional
        The module containing the exception function or None.
    exception_entry_function_name: str, optional
        The name of the exception function in the exception module above. If this is not None the parameter exception_module must
        also be present.
    callables_module_DC: Module, optional
        The module containing the direct callable (DC) function or None.
    callables_entry_function_name_DC: str, optional
        The name of the direct callable (DC) function in the callables_module_DC module above. If this is not None the parameter callables_module_DC must
        also be present.
    callables_module_CC: Module, optional
        The module containing the continuation callable (CC) function or None.
    callables_entry_function_name_CC: str, optional
        The name of the continuation callable (CC) function in the callables_module_CC module above. If this is not None the parameter callables_module_CC must
        also be present.
    hitgroup_module_CH: Module, optional
        The module containing the closest hit (CH) function or None.
    hitgroup_entry_function_name_CH: str, optional
        The name of the closest hit (CH) function in the hitgroup_module_CH module above. If this is not None the parameter hitgroup_module_CH must
        also be present.
    hitgroup_module_AH: Module, optional
        The module containing the any hit (AH) function or None.
    hitgroup_entry_function_name_AH: str, optional
        The name of the any hit (AH) function in the hitgroup_module_AH module above. If this is not None the parameter hitgroup_module_AH must
        also be present.
    hitgroup_module_IS: Module, optional
        The module containing the intersection (IS) function or None.
    hitgroup_entry_function_name_IS: str, optional
        The name of the intersection (IH) function in the hitgroup_module_IS module above. If this is not None the parameter hitgroup_module_IS must
        also be present.
    """
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
            desc.raygen.module = raygen_module.module
            tmp_entry_function_name_1 = raygen_entry_function_name.encode('ascii')
            desc.raygen.entryFunctionName = tmp_entry_function_name_1
        elif miss_entry_function_name is not None:
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS
            if miss_module is None:
                raise ValueError("miss entry function specified but no module given")
            desc.miss.module = miss_module.module
            tmp_entry_function_name_1 = miss_entry_function_name.encode('ascii')
            desc.miss.entryFunctionName = tmp_entry_function_name_1
        elif exception_entry_function_name is not None:
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION
            if exception_module is None:
                raise ValueError("exception entry function specified but no module given")
            desc.exception.module = exception_module.module
            tmp_entry_function_name_1 = exception_entry_function_name.encode('ascii')
            desc.exception.entryFunctionName = tmp_entry_function_name_1
        elif callables_entry_function_name_DC or callables_entry_function_name_CC:
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES
            if callables_entry_function_name_DC is not None:
                if callables_module_DC is None:
                    raise ValueError("callables DC entry function specified but no module given")
                else:
                    desc.callables.moduleDC = callables_module_DC.module
                    tmp_entry_function_name_1 = callables_entry_function_name_DC.encode('ascii')
                    desc.callables.entryFunctionNameDC = tmp_entry_function_name_1
            if callables_entry_function_name_CC is not None:
                if callables_module_CC is None:
                    raise ValueError("callables CC entry function specified but no module given")
                else:
                    desc.callables.moduleCC = callables_module_CC.module
                    tmp_entry_function_name_2 = callables_entry_function_name_CC.encode('ascii')
                    desc.callables.entryFunctionNameCC = tmp_entry_function_name_2
        elif hitgroup_module_CH is not None or hitgroup_module_AH is not None or hitgroup_module_IS is not None:
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP
            if hitgroup_entry_function_name_CH is not None:
                if hitgroup_module_CH is None:
                    raise ValueError("hitgroup CH entry function specified but no module given")
                else:
                    desc.hitgroup.moduleCH = hitgroup_module_CH.module
                    tmp_entry_function_name_1 = hitgroup_entry_function_name_CH.encode('ascii')
                    desc.hitgroup.entryFunctionNameCH = tmp_entry_function_name_1

            if hitgroup_entry_function_name_AH is not None:
                if hitgroup_module_AH is None:
                    raise ValueError("hitgroup AH entry function specified but no module given")
                else:
                    desc.hitgroup.moduleAH = hitgroup_module_AH.module
                    tmp_entry_function_name_2 = hitgroup_entry_function_name_AH.encode('ascii')
                    desc.hitgroup.entryFunctionNameAH = tmp_entry_function_name_2
            if hitgroup_entry_function_name_IS is not None:
                if hitgroup_module_IS is None:
                    raise ValueError("hitgroup IS entry function specified but no module given")
                else:
                    desc.hitgroup.moduleIS = hitgroup_module_IS.module
                    tmp_entry_function_name_3 = hitgroup_entry_function_name_IS.encode('ascii')
                    desc.hitgroup.entryFunctionNameIS = tmp_entry_function_name_3

        self._kind = desc.kind
        cdef OptixProgramGroupOptions options # init to zero
        memset(&options, 0, sizeof(options))
        optix_check_return(optixProgramGroupCreate(self.context.c_context, &desc, 1, &options, NULL, NULL, &self.program_group))

    def __dealloc__(self):
        if <size_t>self.program_group != 0:
            optix_check_return(optixProgramGroupDestroy(self.program_group))

    @classmethod
    def create_raygen(cls, DeviceContext context, Module module, str entry_function_name):
        """
        Create a raygen ProgramGroup.

        Parameters
        ----------
        context: DeviceContext
            The context to use for this ProgramGroup.
        module: Module
            The module containing the raygen function.
        entry_function_name: str
            The name of the raygen function in the module.

        Returns
        -------
        program_group: ProgramGroup
            The created raygen ProgramGroup.
        """
        return cls(context, raygen_module=module, raygen_entry_function_name=entry_function_name)

    @classmethod
    def create_miss(cls, DeviceContext context, Module module, str entry_function_name):
        """
        Create a miss ProgramGroup.

        Parameters
        ----------
        context: DeviceContext
            The context to use for this ProgramGroup.
        module: Module
            The module containing the miss function.
        entry_function_name: str
            The name of the miss function in the module.

        Returns
        -------
        program_group: ProgramGroup
            The created miss ProgramGroup.
        """
        return cls(context, miss_module=module, miss_entry_function_name=entry_function_name)

    @classmethod
    def create_exception(cls, DeviceContext context, Module module, str entry_function_name):
        """
        Create a exception ProgramGroup.

        Parameters
        ----------
        context: DeviceContext
            The context to use for this ProgramGroup.
        module: Module
            The module containing the exception function.
        entry_function_name: str
            The name of the exception function in the module.

        Returns
        -------
        program_group: ProgramGroup
            The created exception ProgramGroup.
        """
        return cls(context, exception_module=module, exception_entry_function_name=entry_function_name)

    @classmethod
    def create_callables(cls,
                         DeviceContext context,
                         Module module=None,
                         Module module_DC=None,
                         str entry_function_DC=None,
                         Module module_CC=None,
                         str entry_function_CC=None):
        """
        Create a callables ProgramGroup.

        Parameters
        ----------
        context: DeviceContext
            The context to use for this ProgramGroup.
        module: Module, optional
            The module to use for both the direct callables (DC) and continuation callables (CC) functions. If this is None,
            the parameters module_DC and module_CC must be specified if their functions are used.
        module_DC: Module, optional
            The module to use for the direct callables (DC) function.
        entry_function_DC: str, optional
            The name of the direct callables (DC) function in the module. If this is not None the module_DC parameter
            is also expected to be present.
        module_CC: Module, optional
            The module to use for the continuation callables (CC) function.
        entry_function_CC: str, optional
            The name of the continuation callables (CC) function in the module. If this is not None the module_CC parameter
            is also expected to be present.

        Returns
        -------
        program_group: ProgramGroup
            The created callables ProgramGroup.
        """

        # implicitly reuse the module
        if module is not None:
            module_DC = module
            module_CC = module

        return cls(context,
                   callables_module_DC=module_DC,
                   callables_entry_function_name_DC=entry_function_DC,
                   callables_module_CC=module_CC,
                   callables_entry_function_name_CC=entry_function_CC)

    @classmethod
    def create_hitgroup(cls,
                        DeviceContext context,
                        Module module=None,
                        Module module_CH=None,
                        str entry_function_CH=None,
                        Module module_AH=None,
                        str entry_function_AH=None,
                        Module module_IS=None,
                        str entry_function_IS=None):
        """
        Create a hitgroup ProgramGroup.

        Parameters
        ----------
        context: DeviceContext
            The context to use for this ProgramGroup.
        module: Module, optional
            The module to use for all hitgroup functions (closest hit, any hit and intersection). If this is None,
            the parameters module_CH, module_AH and module_IS must be specified if their functions are used.
        module_CH: Module, optional
            The module to use for the closest hit (CH) function.
        entry_function_CH: str, optional
            The name of the closest hit (CH) function in the module. If this is not None the module_CH parameter
            is also expected to be present.
        module_AH: Module, optional
            The module to use for the any hit (AH) function.
        entry_function_AH: str, optional
            The name of the any hit (AH) function in the module. If this is not None the module_AH parameter
            is also expected to be present.
        module_IS: Module, optional
            The module to use for the intersection (IS) function.
        entry_function_IS: str, optional
            The name of the intersection (IS) function in the module. If this is not None the module_IS parameter
            is also expected to be present.

        Returns
        -------
        program_group: ProgramGroup
            The created hitgroup ProgramGroup.
        """
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
        """
        Returns the stack sizes of this ProgramGroup

        Returns
        -------
        cssRG: int
           Continuation stack size of raygen (RG) programs in bytes.
        cssMS: int
           Continuation stack size of miss (MS) programs in bytes.
        cssCH: int
            Continuation stack size of closest hit (CH) programs in bytes.
        cssAH: int
            Continuation stack size of any hit (AH) programs in bytes.
        cssIS: int
            Continuation stack size of intersection (IS) programs in bytes.
        cssSS: int
            Continuation stack size of continuation callables (CC) programs in bytes.
        dssDC: int
            Direct stack size of direct callables (DC) programs in bytes.
        """
        cdef OptixStackSizes stack_sizes
        optix_check_return(optixProgramGroupGetStackSize(self.program_group, &stack_sizes))
        return stack_sizes.cssRG, stack_sizes.cssMS, stack_sizes.cssCH, stack_sizes.cssAH, stack_sizes.cssIS, stack_sizes.cssCC, stack_sizes.dssDC

    def _repr_details(self):
        return f"kind {ProgramGroupKind(self._kind).name}"

    @property
    def kind(self):
        return self._kind
