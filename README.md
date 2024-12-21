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

Five walks staring at node `2` with linear probability along with their timestamps,

```
(2, -9223372036854775808), (3, 70), (5, 82), (4, 97), (6, 98)
(2, -9223372036854775808), (5, 19), (4, 32), (5, 71), (4, 97), (6, 98)
(2, -9223372036854775808), (5, 19), (4, 32), (2, 34), (3, 70), (5, 82), (4, 97), (6, 98)
(2, -9223372036854775808), (5, 19), (4, 97), (6, 98)
(2, -9223372036854775808), (3, 70), (5, 82), (4, 97), (6, 98)
```

Five walks ending at node `2` with linear probability,

```
(6, 27), (4, 34), (2, 9223372036854775807)
(2, 19), (5, 32), (4, 34), (2, 9223372036854775807)
(6, 27), (4, 34), (2, 9223372036854775807)
(6, 27), (4, 34), (2, 9223372036854775807)
(1, 17), (4, 34), (2, 9223372036854775807)
```

Five walks staring at node `2` with exponential probability,

```
(2, -9223372036854775808), (6, 80)
(2, -9223372036854775808), (5, 19), (4, 32), (2, 34), (3, 70), (5, 82), (4, 97), (6, 98)
(2, -9223372036854775808), (5, 19), (4, 32), (2, 34), (6, 80)
(2, -9223372036854775808), (5, 19), (4, 32), (5, 71), (4, 97), (6, 98)
(2, -9223372036854775808), (5, 19), (4, 32), (2, 34), (6, 80)
```

Five walks ending at node `2` with exponential probability,

```
(2, 19), (5, 32), (4, 34), (2, 9223372036854775807)
(2, 19), (5, 32), (4, 34), (2, 9223372036854775807)
(6, 27), (4, 34), (2, 9223372036854775807)
(2, 19), (5, 32), (4, 34), (2, 9223372036854775807)
(2, 19), (5, 32), (4, 34), (2, 9223372036854775807)
```

Five walks staring at node `2` with uniform probability,

```
(2, -9223372036854775808), (3, 70), (5, 82), (4, 97), (6, 98)
(2, -9223372036854775808), (5, 19), (4, 97), (6, 98)
(2, -9223372036854775808), (5, 19), (4, 97), (6, 98)
(2, -9223372036854775808), (6, 80)
(2, -9223372036854775808), (5, 19), (4, 32), (2, 34), (6, 80)
```

Five walks ending at node `2` with uniform probability,

```
(6, 27), (4, 34), (2, 9223372036854775807)
(6, 27), (4, 34), (2, 9223372036854775807)
(2, 19), (5, 32), (4, 34), (2, 9223372036854775807)
(1, 17), (4, 34), (2, 9223372036854775807)
(6, 27), (4, 34), (2, 9223372036854775807)
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
TemporalWalk(int64_t max_time_capacity=-1);
```

Initializes a TemporalWalk object with the maximum time capacity of the graph. The default value of `max_time_capacity` is -1, which means unlimited capacity. If set then edges older than `max_time_capacity` from the latest timestamp are deleted automatically.

### add_multiple_edges

```cpp
void add_multiple_edges(const std::vector<EdgeInfo>& edge_infos);
```

Adds multiple edges to the temporal graph based on the provided vector of EdgeInfo structures, where each structure contains the source node `u`, destination node `i`, and timestamp `t`.

### get_random_walks

```cpp
std::vector<std::vector<int>> get_random_walks(WalkStartAt walk_start_at, int num_walks, int len_walk, RandomPickerType* edge_picker_type, int end_node, RandomPickerType* start_picker_type);
```

The get_random_walks method generates a specified number of random walks from the temporal graph, with each walk having a maximum length defined by len_walk. The direction of the walks can be controlled using the walk_start_at parameter, which accepts Begin, End, or Random to determine whether the walk starts from the beginning, end, or a random position within the graph. The edge_picker_type parameter, passed as a pointer, specifies the sampling strategy for edges during the walk and can be Linear, Exponential, or Uniform. Optionally, an end_node can be provided to define a fixed node to start or end the walks, with a default value of -1 selecting a random node. Additionally, the start_picker_type parameter, also a pointer, specifies the sampling strategy for the starting edge and defaults to the value of edge_picker_type if set to nullptr.

### get_random_walks_for_nodes

```cpp
std::unordered_map<int, std::vector<std::vector<int>>> get_random_walks_for_nodes(WalkStartAt walk_start_at, std::vector<int>& end_nodes, int num_walks, int len_walk, RandomPickerType* edge_picker_type, RandomPickerType* start_picker_type);
```

The get_random_walks_for_nodes method generates random walks for multiple specified nodes in the temporal graph. The direction of the walks is controlled by the walk_start_at parameter, which can take the values Begin, End, or Random to determine whether the walks start from the beginning, end, or a random position in the graph. The end_nodes parameter is a vector of node IDs for which the walks will be generated, with each node producing a specified number of random walks (num_walks) of a given maximum length (len_walk). The edge_picker_type parameter, passed as a pointer, defines the sampling strategy for edges during the walk and supports Linear, Exponential, or Uniform sampling. Similarly, the start_picker_type parameter, also a pointer, specifies the sampling strategy for the starting edge and defaults to the value of edge_picker_type if set to nullptr.

### get_random_walks_with_times

```cpp
std::vector<std::vector<std::pair<int, int64_t>>> get_random_walks_with_times(WalkStartAt walk_start_at, int num_walks, int len_walk, RandomPickerType* edge_picker_type, int end_node, RandomPickerType* start_picker_type)
```


The get_random_walks_with_times method generates random walks from the temporal graph, where each step in the walk includes both the node ID and its corresponding timestamp. The direction of the walks is controlled by the walk_start_at parameter, which can take the values Begin, End, or Random to specify whether the walks start from the beginning, end, or a random position in the graph. The num_walks parameter defines the number of walks to generate, while len_walk specifies the maximum length of each walk. The edge_picker_type parameter, passed as a pointer, determines the sampling strategy for edges during the walk and supports Linear, Exponential, or Uniform sampling methods. Similarly, the start_picker_type parameter, also a pointer, specifies the sampling strategy for the starting edge and defaults to the value of edge_picker_type if set to nullptr. An optional end_node parameter can be provided to define a specific node for starting or ending the walks; if set to -1, the end-node is selected randomly.

### get_random_walks_for_nodes_with_times

```cpp
std::unordered_map<int, std::vector<std::vector<std::pair<int, int64_t>>>> get_random_walks_for_nodes_with_times(const WalkStartAt walk_start_at, const std::vector<int>& end_nodes, const int num_walks, const int len_walk, const RandomPickerType* edge_picker_type, const RandomPickerType* start_picker_type)
```

The get_random_walks_for_nodes_with_times method generates timestamped random walks for multiple specified nodes in the temporal graph. The walk_start_at parameter controls the starting point of the walks, allowing values Begin, End, or Random to specify whether the walks start from the beginning, end, or a random position in the graph. The end_nodes parameter is a vector of node IDs for which the walks will be generated. For each node in end_nodes, the function generates a specified number of walks (num_walks) with a maximum length of len_walk. The edge_picker_type parameter, passed as a pointer, determines the sampling strategy for edges during the walk and can be Linear, Exponential, or Uniform. The start_picker_type parameter, also passed as a pointer, specifies the sampling strategy for the starting edge and defaults to the value of edge_picker_type if set to nullptr.

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
TemporalWalk(max_time_capacity: int=-1):
```

Initializes a TemporalWalk object with the maximum time capacity of the graph.

### add_multiple_edges

```python
def add_multiple_edges(edge_infos: List[Tuple[int, int, int64_t]]):
```

Adds multiple directed edges to the temporal graph based on the provided list of tuples. Each tuple should contain three elements: the source node `u`, the destination node `i`, and the timestamp `t`.

### get_random_walks

```python
get_random_walks(
    num_walks: int, len_walk: int, edge_picker_type: str, 
    end_node: int = -1, start_picker_type: Optional[str] = None, 
    walk_start_at: str = "Random",fill_value: int = 0
) -> np.ndarray:
```

Generates random walks from the temporal graph. The walk_start_at parameter controls the starting point for the walks, which can be "Begin", "End", or "Random". The num_walks specifies how many random walks to generate, and len_walk determines the length of each walk. The edge_picker_type parameter defines the sampling strategy for selecting edges during the walk and can be "Linear", "Exponential", or "Uniform". Optionally, an end_node can be provided to specify a fixed node to start or end the walks; if set to -1, the end node is chosen randomly. The start_picker_type can be used to define a separate sampling strategy for the starting edge; if not provided, it defaults to the same value as edge_picker_type. The function returns a 2D NumPy array containing the generated walks, padded with the fill_value where necessary to ensure all walks are of the same length.

### get_random_walks_for_nodes

```python
get_random_walks_for_nodes(
    end_nodes: List[int], num_walks: int, len_walk: int,
    edge_picker_type: str, start_picker_type: Optional[str] = None,
    walk_start_at: str = "Random", fill_value: int = 0
) -> Dict[int, np.ndarray]:
```

Generates random walks for multiple specified nodes in the temporal graph. The walk_start_at parameter controls the starting point for the walks, which can be "Begin", "End", or "Random". The end_nodes parameter is a list of node IDs for which the walks will be generated. For each node in end_nodes, the method generates num_walks random walks, each of length len_walk. The edge_picker_type determines the sampling strategy for selecting edges during the walk, with possible values of "Linear", "Exponential", or "Uniform". The start_picker_type allows the specification of a different sampling strategy for the starting edges; if not provided, it defaults to edge_picker_type. The function returns a dictionary where the keys are node IDs, and the values are 2D NumPy arrays containing the random walks for those nodes. If any walk is shorter than len_walk, it will be padded with fill_value to ensure uniform length across all walks.

### get_random_walks_with_times

```python
get_random_walks_with_times(
    num_walks: int, len_walk: int, edge_picker_type: str,
    walk_start_at: str = "Random", end_node: int = -1,
    start_picker_type: Optional[str] = None
) -> List[List[Tuple[int, int]]]:
```

Generates timestamped random walks from the temporal graph, where each step in the walk consists of a node ID and its corresponding timestamp. The walk_start_at parameter controls the sampling point for the walk, which can be "Begin", "End", or "Random". An optional end_node can be specified to control the endpoint of the walk. If -1 is provided, the endpoint is selected randomly. The edge_picker_type parameter determines the sampling strategy for selecting edges during the walk, with possible values of "Linear", "Exponential", or "Uniform". The start_picker_type allows the specification of a different edge picker type for the starting edges; if not provided, it defaults to edge_picker_type. This function returns a list of walks, where each walk is represented as a list of tuples, with each tuple containing the node ID and the corresponding timestamp.

### get_random_walks_for_nodes_with_times

```python
get_random_walks_for_nodes_with_times(
    num_walks: int, len_walk: int, edge_picker_type: str,
    walk_start_at: str = "Random", end_nodes: List[int],
    start_picker_type: Optional[str] = None
) -> Dict[int, List[List[Tuple[int, int]]]]:
```

Generates timestamped random walks for each specified node in end_nodes. The walk_start_at parameter controls the sampling point for the walk, which can be "Begin", "End", or "Random". The end_nodes parameter is a list of node IDs for which the walks are generated. The edge_picker_type determines the sampling strategy for selecting edges during the walk, with possible values of "Linear", "Exponential", or "Uniform". If start_picker_type is provided, it specifies the edge picker type for the starting edges, otherwise, it defaults to edge_picker_type. This method returns a dictionary that maps each node ID to a list of walks, where each walk is represented as a list of tuples, containing the node ID and timestamp.

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