## Prepares dataset from scratch

# Requires MNIST files:
#   - mnist/t10k-images-idx3-ubyte
#   - mnist/t10k-labels-idx1-ubyte
#   - mnist/train_images-idx3-ubyte
#   - mnist/train_labels_idx1-ubyte

# Saves data to
#   - data/<node_id>/mnist_png_training_shuffled.tar.gz
#   - data/<node_id>/mnist_png_testing_shuffled.tar.gz

echo "Cleaning dataset... "
rm -r ./data/*

echo "Partitioning dataset... "
./partition_data.sh

echo "Shuffling dataset... "
./shuffle_dataset.sh

# Remove intermediate files / non tar.gz files?