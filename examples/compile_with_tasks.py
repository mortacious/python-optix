import concurrent.futures
import optix as ox
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger()


if __name__ == "__main__":
    if ox.optix_version()[1] < 4:
        raise NotImplementedError("Parallel tasks are not implemented in optix versions < 7.3.")

    parser = argparse.ArgumentParser("Compile OptiX modules using parallel tasks")
    parser.add_argument('file', nargs=1, help="The input file (.ptx or .cu) to compile")
    parser.add_argument('-na', '--num-attributes', type=int, default=2, required=False,
                        help="Number of attribute values (up to 8, default 2)")
    parser.add_argument('-npv', '--num-payload-values', type=int, default=2, required=False,
                        help=f"Number of payload values (up to {ox.PipelineCompileOptions.DEFAULT_MAX_PAYLOAD_VALUE_COUNT}, default 2)")
    parser.add_argument('-npt', '--num-payload-types', type=int, default=1, required=False,
                        help=f"Number of payload types (up to {ox.ModuleCompileOptions.DEFAULT_MAX_PAYLOAD_TYPE_COUNT}, default 1)")
    parser.add_argument('-ni', '--num-iters', type=int, default=1, required=False,
                        help="Number of iterations to compile. > 1 disables disk cache (default 1)")
    parser.add_argument('-dt', '--disable-tasks', action='store_true', required=False,
                        help="Disable compilation with tasks (default enabled)")
    parser.add_argument('-nt', '--num-threads', type=int, default=1, required=False,
                        help="Number of threads (default 1)")
    parser.add_argument('-mt', '--max-num-tasks', type=int, default=2, required=False,
                        help="Maximum number of additional tasks (default 2)")

    args = parser.parse_args()

    logger = ox.Logger(log)
    ctx = ox.DeviceContext(validation_mode=True, log_callback_function=logger, log_callback_level=3)

    if args.num_iters > 1:
        ctx.cache_enabled = False

    # compile the file content to ptx in case a .cu file is given
    ptx = ox.Module.compile_cuda_ptx(args.file[0])

    pipeline_options = ox.PipelineCompileOptions(num_payload_values=0,
                                                 num_attribute_values=args.num_attributes)

    payload_semantics = [ox.PayloadSemantics.DEFAULT] * args.num_payload_values
    payload_types = [payload_semantics] * args.num_payload_types

    compile_opts = ox.ModuleCompileOptions(payload_types=payload_types)

    use_tasks = not args.disable_tasks

    if use_tasks:
        tic = time.time()
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            for i in range(args.num_iters):
                module, task = ox.Module.create_as_task(ctx, ptx, module_compile_options=compile_opts, pipeline_compile_options=pipeline_options)
                task_futures = {executor.submit(task.execute, args.max_num_tasks)}
            while task_futures:
                done, not_done = concurrent.futures.wait(task_futures, timeout=0.25, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    new_tasks = future.result()
                    if len(new_tasks) > 0:
                        task_futures.update({executor.submit(t.execute, args.max_num_tasks) for t in new_tasks})
                    task_futures.remove(future)

        # wait for the executor to finish here
        print("Overall run time with tasks", time.time()-tic)
    else:
        tic = time.time()
        for i in range(args.num_iters):
            module = ox.Module(ctx, ptx, module_compile_options=compile_opts, pipeline_compile_options=pipeline_options)
        print("Overall run time without tasks", time.time()-tic)
