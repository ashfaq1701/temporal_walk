import csv
from temporal_walk import TemporalWalk

def print_walks_for_nodes(node_walks_map):
    for node_id, walks in node_walks_map.items():
        print(f'Walks for node: {node_id}')
        for idx, walk in enumerate(walks):
            print(f"Walk {idx}: {','.join(map(str, walk))}")

    print("\n-----------------------------------------------------------------------------\n")

if __name__ == '__main__':
    temporal_walk_obj = TemporalWalk()

    data_file_path = 'data/sample_data.csv'

    data_tuples = []

    with open(data_file_path, mode='r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            u = int(row[0])
            i = int(row[1])
            ts = int(row[2])
            data_tuples.append((u, i, ts))

    temporal_walk_obj.add_multiple_edges(data_tuples)

    print(f'Total edges: {temporal_walk_obj.get_edge_count()}')
    print(f'Total nodes: {temporal_walk_obj.get_node_count()}')

    walks = temporal_walk_obj.get_random_walks_for_nodes(20, "Exponential", -1, 10, "Linear", "Forward_In_Time", "Bias_Earliest_Time", 4, 0.5)
    print(f"Number of walks {walks}")
    print_walks_for_nodes(walks[:100])
