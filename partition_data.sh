# Partitions data into data/<node_id>/<label>/<i>.png

#python3 ./mnist_to_dataset_split.py <input_dir> <output_dir> <num_nodes> <percentage_uniform> <datapoints>
python3 ./mnist_to_dataset_split.py mnist $1 10 0.5 -1