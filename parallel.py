# from multiprocessing import Pool, cpu_count

# def parallelise(func, *args_list, **kwargs_list):
#     """
#     Executes the given function in parallel by applying it to multiple sets of arguments.

#     Args:
#         func: The function to be executed in parallel.
#         max_workers: maximum number of processes to run in parallel
#         *args_list: Variable-length argument list containing iterables, each representing a list of arguments
#                     to be passed to the function. The function will be called for each set of arguments in parallel.
#         **kwargs_list: Variable-length keyword argument list containing keyword arguments to be passed to the function.

#     Returns:
#         List: A list containing the results of executing the function for each set of arguments.

#     Example:
#         def func(filename, filepath, suffix="asdf"):
#             # Your function logic here
#             # ...
#             return result

#         filenames = ["file1.txt", "file2.txt", "file3.txt"]
#         filepaths = ["/path/to/file1", "/path/to/file2", "/path/to/file3"]
#         suffixes = ["suffix1", "suffix2", "suffix3"]

#         result = parallelise(func, filenames, filepaths, suffixes)
#     """
#     if len(args_list[0]) == 0:
#         return None
#     else:
#         # if max_workers:
#         #     num_processors = min(len(args_list[0]), max_workers)
#         # else:
#         num_processors = min(len(args_list[0]), cpu_count())
#         print(f"# Processes: {len(args_list[0])}\nProcessors available: {cpu_count()}\nCPUs used: {num_processors}")

#         # Number of processes to run in parallel
#         pool = Pool(processes=num_processors)

#         if kwargs_list:
#             results = pool.starmap(func, zip(*args_list), **kwargs_list)            
#         else:
#             results = pool.starmap(func, zip(*args_list))
#         # # Map the function call to each set of arguments using multiprocessing
#         # results = pool.starmap(func, zip(*args_list), **kwargs_list)

#         pool.close()
#         pool.join()

#     return results

from multiprocessing import Pool, cpu_count

def parallelise(func, args_list, max_workers=None, **kwargs_list):
    """
    Executes the given function in parallel by applying it to multiple sets of arguments.
    Automatically replicates single arguments to match the number of processes.

    Args:
        func: The function to be executed in parallel.
        args_list: A list of tuples, each representing a set of arguments to be passed to the function.
        max_workers (optional): Maximum number of processes to run in parallel. If not specified, 
                                uses the number of available CPUs.
        **kwargs_list: Variable-length keyword argument list containing keyword arguments to be passed to the function.

    Returns:
        List: A list containing the results of executing the function for each set of arguments.
    """
    if not args_list:
        return []

    # Replicate kwargs to match the length of args_list
    replicated_kwargs = {key: [value]*len(args_list) if not isinstance(value, list) else value 
                         for key, value in kwargs_list.items()}

    # Combine args and kwargs into a list of tuples
    combined_args = [args + tuple(replicated_kwargs[key][i] for key in sorted(replicated_kwargs)) 
                     for i, args in enumerate(args_list)]

    # Determine the number of processors to use
    num_processors = min(len(args_list), max_workers or cpu_count())
    print(f"# Processes: {len(args_list)}\nProcessors available: {cpu_count()}\nCPUs used: {num_processors}")

    # Execute the function in parallel
    with Pool(processes=num_processors) as pool:
        results = pool.starmap(func, combined_args)

    return results
# Example usage
# Instead of result = parallelise(func, [(arg1, arg2, arg3), (arg4, arg5, arg6)], flag=[True, True])
# You can now use result = parallelise(func, [(arg1, arg2, arg3), (arg4, arg5, arg6)], flag=True)

