Module _temporal_walk
=====================

Classes
-------

`ExponentialIndexRandomPicker(...)`
:   __init__(self: _temporal_walk.ExponentialIndexRandomPicker, use_gpu: bool = False) -> None
    
    Initialize a ExponentialIndexRandomPicker instance.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `pick_random(self: _temporal_walk.ExponentialIndexRandomPicker, start: int, end: int, prioritize_end: bool = True)`
    :   Pick a random index with exponential probabilities with index sampling.

`LinearRandomPicker(...)`
:   __init__(self: _temporal_walk.LinearRandomPicker, use_gpu: bool = False) -> None
    
    Initialize a LinearRandomPicker instance.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `pick_random(self: _temporal_walk.LinearRandomPicker, start: int, end: int, prioritize_end: bool = True)`
    :   Pick a random index with linear probabilities.

`TemporalWalk(...)`
:   __init__(self: _temporal_walk.TemporalWalk, is_directed: bool, use_gpu: Optional[bool] = False, max_time_capacity: Optional[int] = None, enable_weight_computation: Optional[bool] = None, timescale_bound: Optional[float] = None) -> None

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `add_edges_from_networkx(self: _temporal_walk.TemporalWalk, arg0: object)`
    :   Adds edges from a networkx graph to the current TemporalWalk object.
        
        Parameters:
        - nx_graph (networkx.Graph): The networkx graph to load edges from.

    `add_multiple_edges(self: _temporal_walk.TemporalWalk, arg0: list[tuple[int, int, int]])`
    :   Adds multiple directed edges to the temporal graph based on the provided vector of tuples.
        
        Parameters:
        - edge_infos (List[Tuple[int, int, int64_t]]): A list of tuples, each containing (source node, destination node, timestamp).

    `clear(self: _temporal_walk.TemporalWalk)`
    :   Clears and reinitiates the underlying graph.

    `get_edge_count(self: _temporal_walk.TemporalWalk)`
    :   Returns the total number of directed edges in the temporal graph.
        
        Returns:
        int: The total number of directed edges.

    `get_node_count(self: _temporal_walk.TemporalWalk)`
    :   Returns the total number of nodes present in the temporal graph.
        
        Returns:
        int: The total number of nodes.

    `get_node_ids(self: _temporal_walk.TemporalWalk)`
    :   get_node_ids(self: _temporal_walk.TemporalWalk) -> numpy.ndarray[numpy.int32]
        
        
        Returns a NumPy array containing the IDs of all nodes in the temporal graph.
        
        Returns:
        np.ndarray: A NumPy array with all node IDs.

    `get_random_walks(self: _temporal_walk.TemporalWalk, max_walk_len: int, walk_bias: str, num_walks_per_node: int, initial_edge_bias: str | None = None, walk_direction: str = 'Forward_In_Time')`
    :   Generates temporal random walks.
        
        Parameters:
        max_walk_len (int): Maximum length of each random walk
        walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "ExponentialIndex", "ExponentialWeight")
        num_walks_per_node (int): Number of walks per node
        initial_edge_bias (str, optional): Type of bias for selecting initial edge
        walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")
        
        Returns:
        List[List[int]]: List of walks, each containing a sequence of node IDs

    `get_random_walks_and_times(self: _temporal_walk.TemporalWalk, max_walk_len: int, walk_bias: str, num_walks_per_node: int, initial_edge_bias: str | None = None, walk_direction: str = 'Forward_In_Time')`
    :   Generates temporal random walks with timestamps.
        
        Parameters:
        max_walk_len (int): Maximum length of each random walk
        walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "ExponentialIndex", "ExponentialWeight")
        num_walks_per_node (int): Number of walks per node
        initial_edge_bias (str, optional): Type of bias for selecting initial edge
        walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")
        
        Returns:
        List[List[Tuple[int, int64_t]]]: List of walks, each containing (node_id, timestamp) pairs

    `get_random_walks_and_times_for_all_nodes(self: _temporal_walk.TemporalWalk, max_walk_len: int, walk_bias: str, num_walks_per_node: int, initial_edge_bias: str | None = None, walk_direction: str = 'Forward_In_Time')`
    :   Generates temporal random walks with timestamps for all the nodes in the graph.
        
        Parameters:
        max_walk_len (int): Maximum length of each random walk
        walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "ExponentialIndex", "ExponentialWeight")
        num_walks_per_node (int): Number of walks per node
        initial_edge_bias (str, optional): Type of bias for selecting initial edge
        walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")
        
        Returns:
        List[List[Tuple[int, int64_t]]]: List of walks, each containing (node_id, timestamp) pairs

    `get_random_walks_and_times_with_specific_number_of_contexts(self: _temporal_walk.TemporalWalk, max_walk_len: int, walk_bias: str, num_cw: int | None = None, num_walks_per_node: int | None = None, initial_edge_bias: str | None = None, walk_direction: str = 'Forward_In_Time', context_window_len: int | None = None, p_walk_success_threshold: float = 0.009999999776482582)`
    :   Generates temporal random walks with timestamps with specified number of contexts. Here number of walks can vary based on their actual lengths.
        
        Parameters:
        max_walk_len (int): Maximum length of each random walk
        walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "ExponentialIndex", "ExponentialWeight")
        num_cw (int, optional): Number of context windows to generate
        num_walks_per_node (int, optional): Number of walks per node (used if num_cw not specified)
        initial_edge_bias (str, optional): Type of bias for selecting initial edge
        walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")
        context_window_len (int, optional): Size of context window
        p_walk_success_threshold (float): Minimum proportion of successful walks (default: 0.01)
        
        Returns:
        List[List[Tuple[int, int64_t]]]: List of walks, each containing (node_id, timestamp) pairs

    `get_random_walks_for_all_nodes(self: _temporal_walk.TemporalWalk, max_walk_len: int, walk_bias: str, num_walks_per_node: int, initial_edge_bias: str | None = None, walk_direction: str = 'Forward_In_Time')`
    :   Generates temporal random walks for all the nodes in the graph.
        
        Parameters:
        max_walk_len (int): Maximum length of each random walk
        walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "ExponentialIndex" or "ExponentialWeight")
        num_walks_per_node (int): Number of walks per node
        initial_edge_bias (str, optional): Type of bias for selecting initial edge
        walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")
        
        Returns:
        List[List[int]]: List of walks, each containing a sequence of node IDs

    `get_random_walks_with_specific_number_of_contexts(self: _temporal_walk.TemporalWalk, max_walk_len: int, walk_bias: str, num_cw: int | None = None, num_walks_per_node: int | None = None, initial_edge_bias: str | None = None, walk_direction: str = 'Forward_In_Time', context_window_len: int | None = None, p_walk_success_threshold: float = 0.009999999776482582)`
    :   Generates temporal random walks with specified number of contexts. Here number of walks can vary based on their actual lengths.
        
        Parameters:
        max_walk_len (int): Maximum length of each random walk
        walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "ExponentialIndex", "ExponentialWeight")
        num_cw (int, optional): Number of context windows to generate
        num_walks_per_node (int, optional): Number of walks per node (used if num_cw not specified)
        initial_edge_bias (str, optional): Type of bias for selecting initial edge
        walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")
        context_window_len (int, optional): Size of context window
        p_walk_success_threshold (float): Minimum proportion of successful walks (default: 0.01)
        
        Returns:
        List[List[int]]: List of walks, each containing a sequence of node IDs

    `to_networkx(self: _temporal_walk.TemporalWalk)`
    :   Exports the TemporalWalk object to a networkX graph.
        
        Returns:
        networkx.Graph: The exported networkx graph.

`UniformRandomPicker(...)`
:   __init__(self: _temporal_walk.UniformRandomPicker, use_gpu: bool = False) -> None
    
    Initialize a UniformRandomPicker instance.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `pick_random(self: _temporal_walk.UniformRandomPicker, start: int, end: int, prioritize_end: bool = True)`
    :   Pick a random index with uniform probabilities.

`WeightBasedRandomPicker(...)`
:   __init__(self: _temporal_walk.WeightBasedRandomPicker) -> None
    
    Initialize a WeightBasedRandomPicker instance.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `pick_random(self: _temporal_walk.WeightBasedRandomPicker, cumulative_weights: list[float], group_start: int, group_end: int)`
    :   Pick a random index based on cumulative weights