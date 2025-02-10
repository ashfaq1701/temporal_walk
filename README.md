# 🚀 Temporal Walk

[![PyPI Latest Release](https://img.shields.io/pypi/v/temporal-walk.svg)](https://pypi.org/project/temporal-walk/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/temporal-walk.svg)](https://pypi.org/project/temporal-walk/)

**A high-performance temporal walk sampler for dynamic networks with GPU acceleration. Built for scale.**

---

## 🔥 Why Temporal Walk?
✅ **Performance First** – GPU-accelerated sampling for massive networks  
✅ **Memory Efficient** – Smart memory management for large graphs  
✅ **Flexible Integration** – Easy Python bindings with **NumPy/NetworkX** support  
✅ **Production Ready** – Developed by [Packets Research Lab](https://packets-lab.github.io/)

---

## ⚡ Quick Start

```python
from temporal_walk import TemporalWalk

# Create a directed temporal graph
walker = TemporalWalk(is_directed=True, use_gpu=False)

# Add edges: (source, target, timestamp)
edges = [
    (4, 5, 71), (3, 5, 82), (1, 3, 19),
    (4, 2, 34), (4, 3, 79), (2, 5, 19)
]
walker.add_multiple_edges(edges)

# Sample walks with exponential time bias
walks = walker.get_random_walks_for_all_nodes(
    max_walk_len=5,
    walk_bias="ExponentialWeight",
    num_walks_per_node=10,
    initial_edge_bias="Uniform"
)
```

## ✨ Key Features
- ⚡ **GPU acceleration** for large graphs
- 🎯 **Multiple sampling strategies** – Uniform, Linear, Exponential
- 🔄 **Forward & backward** temporal walks
- 📡 **Rolling window support** for streaming data
- 🔗 **NetworkX integration**
- 🛠️ **Efficient memory management**

---

## 📦 Installation

```sh
pip install temporal-walk
```

## 📖 Documentation

📌 **[C++ Documentation →](https://htmlpreview.github.io/?https://github.com/ashfaq1701/temporal_walk/blob/master/docs/html/class_temporal_walk.html)**
📌 **[Python Interface Documentation →](docs/_temporal_walk.md)**

---

## 📚 Inspired By

**Nguyen, Giang Hoang, et al.**  
*"Continuous-Time Dynamic Network Embeddings."*  
*Companion Proceedings of The Web Conference 2018.*

## 👨‍🔬 Built by [Packets Research Lab](https://packets-lab.github.io/)

🚀 **Contributions welcome!** Open a PR or issue if you have suggestions.  

