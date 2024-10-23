# Temporal Walk

A modified implementation of temporal walk algorithm from "Continuous-Time Dynamic Network Embeddings" paper.

---

## Introduction

This project enables the construction of large temporal networks in memory, from which temporal walks can be sampled. Temporal walks are invaluable in graph neural networks (GNNs) for learning network dynamics.

This library facilitates the creation of temporal graphs and the incremental sampling of temporal walks based on the current graph state, making it especially useful for training GNNs. PyBind is interfaced which let's the functions to be called from Python. For convenience the walks are returned as numpy arrays.

---

## Functions

`TemporalWalk` class contains the public facing functions.

### Constructor

```cpp
TemporalWalk(int num_walks, int len_walk, RandomPickerType picker_type);
```

Initializes a TemporalWalk object with the specified number of walks, length of each walk, and the type of random picker to be used. Three random pickers are available `Exponential`, `Linear` and `Uniform`.

### add_edge

```cpp
void add_edge(int u, int i, int64_t t);
```

Adds a directed edge from node u to node i at the specified timestamp t in the temporal graph.

### add_multiple_edges

```cpp
void add_multiple_edges(const std::vector<EdgeInfo>& edge_infos);
```

Adds multiple edges to the temporal graph based on the provided vector of EdgeInfo structures, where each structure contains the source node `u`, destination node `i`, and timestamp `t`.

### get_random_walks

```cpp
std::vector<std::vector<int>> get_random_walks(WalkStartAt walk_start_at, int end_node=-1);
```

Generates a specified number of random walks from the temporal graph. The walks can be sampled from destination to source or source to destination. This can be controlled using `walk_start_at`, which can have values `Begin`, `End` or `Random`. An end-node can be specified to start or end the walks. The default value `-1` picks the end-node randomly.

### get_random_walks_for_nodes

```cpp
std::unordered_map<int, std::vector<std::vector<int>>> get_random_walks_for_nodes(WalkStartAt walk_start_at, const std::vector<int>& end_nodes);
```

Generates random walks for multiple specified nodes in the temporal graph. The walks can be sampled from destination to source or source to destination, controlled by the `walk_start_at` parameter, which can take the values `Begin`, `End`, or `Random`. The end_nodes parameter is a vector containing the IDs of the nodes for which the walks will be generated. For each node in end_nodes, the function produces random walks based on the specified starting point, returning the walks as a mapping of node IDs to their corresponding random walks.

### get_len_walk

```cpp
int get_len_walk();
```

Returns the length of the random walks that will be generated by the `TemporalWalk` object.

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

---

## Python Interfaces

The Python bindings for the `TemporalWalk` class provide a seamless way to interact with the C++ implementation from Python. The bindings are created using the `pybind11` library, enabling easy access to the functionality of the `TemporalWalk` class.

### Constructor

```python
TemporalWalk(num_walks: int, len_walk: int, picker_type: str):
```

Initializes a TemporalWalk object with the specified number of walks, length of each walk, and the type of random picker to be used. The picker_type should be one of the following strings: `"Uniform"`, `"Linear"`, or `"Exponential"`.

### add_edge

```python
add_edge(u: int, i: int, t: int):
```

Adds a directed edge from node u to node i at the specified timestamp t in the temporal graph.

### add_multiple_edges

```python
def add_multiple_edges(edge_infos: List[Tuple[int, int, int64_t]]):
```

Adds multiple directed edges to the temporal graph based on the provided list of tuples. Each tuple should contain three elements: the source node `u`, the destination node `i`, and the timestamp `t`.

### get_random_walks

```python
get_random_walks(walk_start_at: str, end_node: int = -1, fill_value: int = -1) -> np.ndarray:
```

Generates random walks from the temporal graph. The walks can be sampled from destination to source or source to destination, controlled by the `walk_start_at` parameter, which can be `"Begin"`, `"End"`, or `"Random"`. An `end_node` can be specified to start or end the walks. The default value `-1` picks the end-node randomly. Returns a 2D NumPy array containing the generated walks, padded with `fill_value` where necessary.

### get_random_walks_for_nodes

```python
get_random_walks_for_nodes(walk_start_at: str, end_nodes: List[int], fill_value: int = -1) -> List[np.ndarray]:
```

Generates random walks for multiple specified nodes in the temporal graph. Similar to get_random_walks, the walks can be sampled from destination to source or source to destination, controlled by the `walk_start_at` parameter. The end_nodes parameter is a list of integers representing the IDs of the nodes for which the walks will be generated. Returns a list of 2D NumPy arrays, each corresponding to the walks for a specific node, padded with `fill_value` where necessary.

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

---

Nguyen, Giang Hoang et al. “Continuous-Time Dynamic Network Embeddings.” Companion Proceedings of the The Web Conference 2018 (2018).