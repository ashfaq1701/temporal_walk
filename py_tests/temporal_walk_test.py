import csv
from temporal_walk import TemporalWalk


USE_GPU = False

def print_walks_for_nodes(walks):
    for idx, walk in enumerate(walks):
        print(f"Walk {idx}: {','.join(map(str, walk))}")

if __name__ == '__main__':
    temporal_walk_obj = TemporalWalk(False, USE_GPU)

    data_file_path = '../data/sample_data.csv'

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

    walks = temporal_walk_obj.get_random_walks_and_times_for_all_nodes(
        max_walk_len=20,
        walk_bias="ExponentialIndex",
        num_walks_per_node=10,
        initial_edge_bias="Linear",
        walk_direction="Forward_In_Time"
    )

    print(f"Number of walks {len(walks)}")
    print_walks_for_nodes(walks[:100])
