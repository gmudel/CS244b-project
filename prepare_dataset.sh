## Prepares dataset from scratch

# Requires MNIST files:
#   - mnist/t10k-images-idx3-ubyte
#   - mnist/t10k-labels-idx1-ubyte
#   - mnist/train_images-idx3-ubyte
#   - mnist/train_labels_idx1-ubyte

# Saves data to
#   - data/<node_id>/mnist_png_training_shuffled.tar.gz
#   - data/<node_id>/mnist_png_testing_shuffled.tar.gz

datadir=$1

echo "Cleaning dataset... "
rm -r ./$datadir/*

echo "Partitioning dataset... "
./partition_data.sh $datadir

echo "Shuffling dataset... "
./shuffle_dataset.sh $datadir

# Remove intermediate files / non tar.gz files?
echo "Removing temporary test files... "
rm -r ./$datadir/*-test*