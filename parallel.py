from multiprocessing import Pool, cpu_count

def parallelise(func, args_list, **kwargs_list):
    """
    Executes the given function in parallel by applying it to multiple sets of arguments,
    supporting both individual and list-based keyword arguments for dynamic replication across processes.

    This function facilitates parallel execution by distributing argument sets and replicated
    or singular keyword arguments to the specified function across multiple processes, optimizing
    for computational efficiency and speed. It automatically handles the replication of singular keyword
    arguments to match the number of processes and adjusts for keyword arguments provided as empty lists
    by converting them into lists of empty lists.

    Args:
        func (callable): The function to be executed in parallel.
        args_list (list of tuple): A list of tuples, each representing a set of arguments to be passed to `func`.
        **kwargs_list: Variable-length keyword argument list containing keyword arguments to be passed to `func`.
                       Includes special handling for 'max_workers', which specifies the maximum number of processes
                       to run in parallel. If not specified, it defaults to the number of available CPUs.

    Returns:
        list: A list containing the results of executing `func` for each set of arguments.

    Example Usage:
        # Define a sample function to run in parallel
        def sample_function(x, flag=False):
            return x * 2 if flag else x + 2

        # Parallel execution without 'max_workers' specified (defaults to available CPUs)
        results = parallelise(sample_function, [(1,), (2,), (3,)], flag=True)

        # Parallel execution with 'max_workers' specified
        results = parallelise(sample_function, [(4,), (5,), (6,)], flag=False, max_workers=2)

    Note:
        - The function automatically replicates single keyword arguments to match the number of argument sets in `args_list`.
        - For keyword arguments provided as lists, it ensures that each process receives the corresponding element.
          If the list length does not match `args_list`, it replicates the entire list for each argument set.
        - Empty lists as keyword arguments are treated specially and converted into a list of empty lists, one for each argument set.
    """
    if not args_list:
        return []
    
    max_workers = kwargs_list.pop('max_workers', None)
    if isinstance(max_workers, int):
        max_workers = max_workers
    else:
        max_workers = cpu_count()  # Use default CPU count if max_workers not specified or not an int
    
    # Replicate kwargs handling special cases
    replicated_kwargs = {}
    for key, value in kwargs_list.items():
        if not isinstance(value, list):
            replicated_kwargs[key] = [value] * len(args_list)
        elif len(value) == 0:  # Special handling for empty list
            replicated_kwargs[key] = [[] for _ in range(len(args_list))]
        elif isinstance(value, list) and len(value) != len(args_list):
            replicated_kwargs[key] = [value] * len(args_list)
        else:
            replicated_kwargs[key] = value

    # Combine args and kwargs for each function call
    combined_args = [
        (list(args) if isinstance(args, tuple) else [args]) + [replicated_kwargs[key][i] for key in replicated_kwargs]
        for i, args in enumerate(args_list)
    ]

    # Determine the number of processors to use
    num_processors = min(len(args_list), max_workers or cpu_count())
    print(f"# Processes: {len(args_list)}, Processors available: {cpu_count()}, CPUs used: {num_processors}")
    # Execute the function in parallel
    with Pool(processes=num_processors) as pool:
        results = pool.starmap(func, tuple(combined_args))

    return results
