# Temporal Walk

[![PyPI Latest Release](https://img.shields.io/pypi/v/temporal-walk.svg)](https://pypi.org/project/temporal-walk/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/temporal-walk.svg)](https://pypi.org/project/temporal-walk/)

A modified implementation of temporal walk algorithm from "Continuous-Time Dynamic Network Embeddings" paper.

---

## Introduction

This project enables the construction of large temporal networks in memory, from which temporal walks can be sampled. Temporal walks are invaluable in graph neural networks (GNNs) for learning network dynamics.

This library facilitates the creation of temporal graphs and the incremental sampling of temporal walks based on the current graph state, making it especially useful for training GNNs. PyBind is interfaced which let's the functions to be called from Python. For convenience the walks are returned as numpy arrays.

The library also supports the creation of continuous graphs with a maximum time capacity, where edges with timestamps older than the current time minus the maximum capacity are automatically deleted.

---

## Definitions

A temporal network represents a dynamic system where edges or interactions between nodes vary over time. In contrast to static networks, temporal networks capture changes and sequences in connectivity, which is essential in understanding real-world phenomena like social interactions, information spread, and transportation flows over time.

A temporal walk is a path through a temporal network that respects the order of timestamps on edges, ensuring causality. Temporal walks are particularly useful for applications in graph neural networks (GNNs), where learning temporal dependencies enhances the model's ability to predict or understand evolving network structures.

---

### Example

For a given temporal network with following edges (upstream node, downstream node, timestamp),

```pytchon
[
    (4, 5, 71),
    (3, 5, 82),
    (1, 3, 19),
    (4, 2, 34),
    (4, 3, 79),
    (2, 5, 19),
    (2, 3, 70),
    (5, 4, 97),
    (4, 6, 57),
    (6, 4, 27),
    (2, 6, 80),
    (6, 1, 42),
    (4, 6, 98),
    (1, 4, 17),
    (5, 4, 32)
]
```

<img src="https://raw.githubusercontent.com/ashfaq1701/temporal_walk/refs/heads/master/images/network.png" alt="Sample Temporal Graph" style="width: 600px; margin: auto"/>

Some forward walks (past to future), 

```
(2, -9223372036854775808), (5, 19), (4, 32), (5, 71), (4, 97), (6, 98)
(1, -9223372036854775808), (3, 19), (5, 82), (4, 97), (6, 98)
(6, -9223372036854775808), (4, 27), (3, 79), (5, 82), (4, 97), (6, 98)
(1, -9223372036854775808), (4, 17), (2, 34), (3, 70), (5, 82), (4, 97), (6, 98)
(1, -9223372036854775808), (5, 19), (4, 32), (3, 79), (5, 82), (4, 97), (6, 98)
```

Some reverse walks (Future to past and then reversed),

```
(6, 27), (4, 79), (3, 82), (5, 97), (4, 9223372036854775807)
(2, 19), (5, 32), (4, 57), (6, 9223372036854775807)
(2, 19), (5, 32), (4, 34), (2, 80), (6, 9223372036854775807)
(6, 27), (4, 98), (6, 9223372036854775807)
(2, 19), (5, 32), (4, 79), (3, 82), (5, 97), (4, 98), (6, 9223372036854775807)
```

## Installation

This project can be installed using pip.

```bash
pip install temporal-walk
```

## Functions

`TemporalWalk` class contains the public facing functions.

### Constructor

```cpp
TemporalWalk(bool is_directed, int64_t max_time_capacity=-1);
```

Initializes a (Un)directed TemporalWalk object with the maximum time capacity of the graph. The is_directed parameter defines if the graph is directed or not. The default value of `max_time_capacity` is -1, which means unlimited capacity. If set then edges older than `max_time_capacity` from the latest timestamp are deleted automatically.

### add_multiple_edges

```cpp
void add_multiple_edges(const std::vector<EdgeInfo>& edge_infos);
```

Adds multiple edges to the temporal graph based on the provided vector of EdgeInfo structures, where each structure contains the source node `u`, destination node `i`, and timestamp `t`.

### get_random_walks_for_all_nodes

```cpp
std::vector<std::vector<int>> get_random_walks_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        WalkInitEdgeTimeBias walk_init_edge_time_bias=WalkInitEdgeTimeBias::Bias_Earliest_Tim);
```

Generates temporal random walks for all nodes in the graph similar to get_random_walks_and_times_for_all_nodes but returns only the node IDs without timestamps.

Parameters:

* max_walk_len: Maximum length of each random walk
* walk_bias: Type of bias for selecting next edges during walk (Uniform, Linear, or Exponential)
* num_walks_per_node: Number of walks per node.
* initial_edge_bias: Optional bias type for selecting initial edges (Uniform, Linear, or Exponential). If nullptr, uses walk_bias
* walk_direction: Direction of temporal walks (Forward_In_Time or Backward_In_Time)
* walk_init_edge_time_bias: Bias for initial edge selection (Bias_Earliest_Time or Bias_Latest_Time)

Returns:

A vector of walks, where each walk is a vector of node IDs representing the temporal path through the network.

### get_random_walks_and_times_for_all_nodes

```cpp
std::vector<std::vector<NodeWithTime>> get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        WalkInitEdgeTimeBias walk_init_edge_time_bias=WalkInitEdgeTimeBias::Bias_Earliest_Time);
```

Generates temporal random walks for all nodes in the graph where each step contains both node ID and timestamp. Each walk respects temporal ordering based on the specified direction and biases. In this function the number of contexts remain fixed. The number of walks can vary based on their actual lengths after sampling.

Parameters:

* max_walk_len: Maximum length of each random walk
* walk_bias: Type of bias for selecting next edges during walk (Uniform, Linear, or Exponential)
* num_walks_per_node: Number of walks per node.
* initial_edge_bias: Optional bias type for selecting initial edges (Uniform, Linear, or Exponential). If nullptr, uses walk_bias
* walk_direction: Direction of temporal walks (Forward_In_Time or Backward_In_Time)
* walk_init_edge_time_bias: Bias for initial edge selection (Bias_Earliest_Time or Bias_Latest_Time)

Returns:

A vector of temporal walks, where each walk is a vector of NodeWithTime pairs containing node IDs and their corresponding timestamps.

### get_random_walks

```cpp
std::vector<std::vector<int>> get_random_walks(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        WalkInitEdgeTimeBias walk_init_edge_time_bias=WalkInitEdgeTimeBias::Bias_Earliest_Tim);
```

Generates temporal random walks similar to get_random_walks_and_times but returns only the node IDs without timestamps.

Parameters:

* max_walk_len: Maximum length of each random walk
* walk_bias: Type of bias for selecting next edges during walk (Uniform, Linear, or Exponential)
* num_walks_per_node: Number of walks per node.
* initial_edge_bias: Optional bias type for selecting initial edges (Uniform, Linear, or Exponential). If nullptr, uses walk_bias
* walk_direction: Direction of temporal walks (Forward_In_Time or Backward_In_Time)
* walk_init_edge_time_bias: Bias for initial edge selection (Bias_Earliest_Time or Bias_Latest_Time)

Returns:

A vector of walks, where each walk is a vector of node IDs representing the temporal path through the network.

### get_random_walks_and_times

```cpp
std::vector<std::vector<NodeWithTime>> get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        WalkInitEdgeTimeBias walk_init_edge_time_bias=WalkInitEdgeTimeBias::Bias_Earliest_Time);
```

Generates temporal random walks where each step contains both node ID and timestamp. Each walk respects temporal ordering based on the specified direction and biases. In this function the number of contexts remain fixed. The number of walks can vary based on their actual lengths after sampling.

Parameters:

* max_walk_len: Maximum length of each random walk
* walk_bias: Type of bias for selecting next edges during walk (Uniform, Linear, or Exponential)
* num_walks_per_node: Number of walks per node.
* initial_edge_bias: Optional bias type for selecting initial edges (Uniform, Linear, or Exponential). If nullptr, uses walk_bias
* walk_direction: Direction of temporal walks (Forward_In_Time or Backward_In_Time)
* walk_init_edge_time_bias: Bias for initial edge selection (Bias_Earliest_Time or Bias_Latest_Time)

Returns:

A vector of temporal walks, where each walk is a vector of NodeWithTime pairs containing node IDs and their corresponding timestamps.

### get_random_walks_with_specific_number_of_contexts

```cpp
std::vector<std::vector<int>> get_random_walks_with_specific_number_of_contexts(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        long num_cw=-1,
        int num_walks_per_node=-1,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        WalkInitEdgeTimeBias walk_init_edge_time_bias=WalkInitEdgeTimeBias::Bias_Earliest_Time,
        int context_window_len=-1,
        float p_walk_success_threshold=0.01);
```

Generates temporal random walks similar to get_random_walks_and_times_with_specific_number_of_contexts but returns only the node IDs without timestamps. In this function the number of contexts remain fixed. The number of walks can vary based on their actual lengths after sampling.

Parameters:

* max_walk_len: Maximum length of each random walk
* walk_bias: Type of bias for selecting next edges during walk (Uniform, Linear, or Exponential)
* num_cw: Number of context windows to generate. If -1, calculated using num_walks_per_node
* num_walks_per_node: Number of walks per node. Used only if num_cw is -1
* initial_edge_bias: Optional bias type for selecting initial edges (Uniform, Linear, or Exponential). If nullptr, uses walk_bias
* walk_direction: Direction of temporal walks (Forward_In_Time or Backward_In_Time)
* walk_init_edge_time_bias: Bias for initial edge selection (Bias_Earliest_Time or Bias_Latest_Time)
* context_window_len: Minimum length of walks. Default is 2 if -1 provided
* p_walk_success_threshold: Minimum required success rate for walk generation (default 0.01)

Returns:

A vector of walks, where each walk is a vector of node IDs representing the temporal path through the network.

### get_random_walks_and_times_with_specific_number_of_contexts

```cpp
std::vector<std::vector<NodeWithTime>> get_random_walks_and_times_with_specific_number_of_contexts(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        long num_cw=-1,
        int num_walks_per_node=-1,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        WalkInitEdgeTimeBias walk_init_edge_time_bias=WalkInitEdgeTimeBias::Bias_Earliest_Time,
        int context_window_len=-1,
        float p_walk_success_threshold=0.01);
```

Generates temporal random walks where each step contains both node ID and timestamp. Each walk respects temporal ordering based on the specified direction and biases. In this function the number of contexts remain fixed. The number of walks can vary based on their actual lengths after sampling.

Parameters:

* max_walk_len: Maximum length of each random walk
* walk_bias: Type of bias for selecting next edges during walk (Uniform, Linear, or Exponential)
* num_cw: Number of context windows to generate. If -1, calculated using num_walks_per_node
* num_walks_per_node: Number of walks per node. Used only if num_cw is -1
* initial_edge_bias: Optional bias type for selecting initial edges (Uniform, Linear, or Exponential). If nullptr, uses walk_bias
* walk_direction: Direction of temporal walks (Forward_In_Time or Backward_In_Time)
* walk_init_edge_time_bias: Bias for initial edge selection (Bias_Earliest_Time or Bias_Latest_Time)
* context_window_len: Minimum length of walks. Default is 2 if -1 provided
* p_walk_success_threshold: Minimum required success rate for walk generation (default 0.01)

Returns:

A vector of temporal walks, where each walk is a vector of NodeWithTime pairs containing node IDs and their corresponding timestamps.

### get_edge_count

```cpp
size_t get_edge_count();
```

Returns the total number of directed edges in the temporal graph.

### get_node_count

```cpp
size_t get_node_count();
```

Returns the total number of nodes present in the temporal graph.

### get_node_ids

```cpp
std::vector<int> get_node_ids();
```

Returns a vector containing the IDs of all nodes in the temporal graph.

### clear

```cpp
void clear();
```

Clears and reinitiates the underlying temporal graph, removing all edges and nodes.

---

## Python Interfaces

The Python bindings for the `TemporalWalk` class provide a seamless way to interact with the C++ implementation from Python. The bindings are created using the `pybind11` library, enabling easy access to the functionality of the `TemporalWalk` class.

### Constructor

```python
TemporalWalk(bool is_directed, max_time_capacity: int=-1):
```

Initializes a (Un)directed TemporalWalk object with the maximum time capacity of the graph. The is_directed parameter defines if the graph is directed or not.

### add_multiple_edges

```python
def add_multiple_edges(edge_infos: List[Tuple[int, int, int64_t]]):
```

Adds multiple directed edges to the temporal graph based on the provided list of tuples. Each tuple should contain three elements: the source node `u`, the destination node `i`, and the timestamp `t`.

### get_random_walks_with_specific_number_of_contexts

```python
get_random_walks_with_specific_number_of_contexts(
    max_walk_len: int,
    walk_bias: str,
    num_cw: Optional[int] = None,
    num_walks_per_node: Optional[int] = None,
    initial_edge_bias: Optional[str] = None,
    walk_direction: str = "Forward_In_Time",
    walk_init_edge_time_bias: str = "Bias_Earliest_Time",
    context_window_len: Optional[int] = None,
    p_walk_success_threshold: float = 0.01
) -> List[List[int]]:
```

Generates temporal random walks from the graph using parallel processing with hardware concurrency. In this function the number of contexts remain fixed. The number of walks can vary based on their actual lengths after sampling.

Parameters

* max_walk_len - Maximum length of each random walk
* walk_bias - Type of bias for selecting next edges during walk:
  * "Uniform": Equal probability for all valid edges
  * "Linear": Linear decay based on time
  * "Exponential": Exponential decay based on time

* num_cw - Number of context windows to generate. If None, calculated using num_walks_per_node
* num_walks_per_node - Number of walks per node. Used only if num_cw is None
* initial_edge_bias - Optional bias type for selecting initial edges. Uses walk_bias if None

* walk_direction - Direction of temporal walks:
  * "Forward_In_Time": Walks progress from past to future
  * "Backward_In_Time": Walks progress from future to past

* walk_init_edge_time_bias - Bias for initial edge selection:

  * "Bias_Earliest_Time": Prioritize earlier timestamps
  * "Bias_Latest_Time": Prioritize later timestamps

* context_window_len - Minimum length of walks (default 2 if None provided)
* p_walk_success_threshold - Minimum required success rate for walk generation (default 0.01)

Returns

List of walks, where each walk is a list of node IDs representing the temporal path through the network.

### get_random_walks_and_times_with_specific_number_of_contexts

```python
get_random_walks_and_times_with_specific_number_of_contexts(
    max_walk_len: int,
    walk_bias: str,
    num_cw: Optional[int] = None,
    num_walks_per_node: Optional[int] = None,
    initial_edge_bias: Optional[str] = None,
    walk_direction: str = "Forward_In_Time",
    walk_init_edge_time_bias: str = "Bias_Earliest_Time",
    context_window_len: Optional[int] = None,
    p_walk_success_threshold: float = 0.01
) -> List[List[Tuple[int, int64_t]]]:
```

Similar to get_random_walks but includes timestamps with each node in the walks. Uses parallel processing with hardware concurrency. In this function the number of contexts remain fixed. The number of walks can vary based on their actual lengths after sampling.

Parameters

* max_walk_len - Maximum length of each random walk
* walk_bias - Type of bias for selecting next edges during walk:
  * "Uniform": Equal probability for all valid edges
  * "Linear": Linear decay based on time
  * "Exponential": Exponential decay based on time

* num_cw - Number of context windows to generate. If None, calculated using num_walks_per_node
* num_walks_per_node - Number of walks per node. Used only if num_cw is None
* initial_edge_bias - Optional bias type for selecting initial edges. Uses walk_bias if None
* walk_direction - Direction of temporal walks:
  * "Forward_In_Time": Walks progress from past to future
  * "Backward_In_Time": Walks progress from future to past

* walk_init_edge_time_bias - Bias for initial edge selection:
  * "Bias_Earliest_Time": Prioritize earlier timestamps
  * "Bias_Latest_Time": Prioritize later timestamps

* context_window_len - Minimum length of walks (default 2 if None provided)
* p_walk_success_threshold - Minimum required success rate for walk generation (default 0.01)

Returns

List of walks, where each walk is a list of tuples containing (node_id, timestamp) pairs, representing the temporal path through the network with corresponding timestamps.

### get_node_count

```python
get_node_count() -> int:
```

Returns the total number of nodes present in the temporal graph.

### get_edge_count

```python
get_edge_count() -> int:
```

Returns the total number of directed edges in the temporal graph.

### get_node_ids

```python
get_node_ids() -> np.ndarray:
```
Returns a numpy array containing the IDs of all nodes in the temporal graph.

### clear

```python
clear():
```

Clears and reinitiates the underlying temporal graph, removing all edges and nodes.

### add_edges_from_networkx

```python
add_edges_from_networkx(nx_graph: nx.DiGraph):
```

Adds edges from a NetworkX directed graph to the current TemporalWalk object. The NetworkX graph must have a 'timestamp' attribute for each edge.

### to_networkx

```python
to_networkx() -> nx.DiGraph:
```

Exports the current temporal graph to a NetworkX DiGraph object. Each edge in the resulting graph will have a 'timestamp' attribute containing the temporal information.

---

Nguyen, Giang Hoang et al. “Continuous-Time Dynamic Network Embeddings.” Companion Proceedings of the The Web Conference 2018 (2018).