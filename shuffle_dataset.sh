curr_dir=$(pwd)

# Merge each label directory into a label.tar.gz file
for label_dir in data/*/*/*; do
    # echo "$label_dir/"
    tar czf "$label_dir.tar.gz" "$label_dir/"
done

# Shuffle the label directories into data/<node_id>/mnist_png_training_shuffled.tar.gz
for node_id in data/[0-9]; do
    
    # Process Training Data
    data_dir="$node_id/training"

    # Shuffles data/<node_id>/training/[0-9].tar.gz -> data/<node_id>/mnist_png_training_shuffled.tar.gz
    cd $data_dir
    go run $GOPATH/pkg/mod/github.com/wangkuiyi/gotorch@v0.0.0-20201028015551-9afed2f3ad7b/tool/tarball_merge/tarball_merge.go \
            -out="mnist_png_training_shuffled.tar.gz" [0-9].tar.gz
    mv "mnist_png_training_shuffled.tar.gz" ..
    cd $curr_dir

done


# Process Testing Data
cd ./data

for i in [0-9]; do

    tar czf "${i}-test.tar.gz" "${i}-test/"

    go run $GOPATH/pkg/mod/github.com/wangkuiyi/gotorch@v0.0.0-20201028015551-9afed2f3ad7b/tool/tarball_merge/tarball_merge.go \
            -out="mnist_png_testing_shuffled.tar.gz" "${i}-test.tar.gz"
done