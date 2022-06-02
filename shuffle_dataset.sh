curr_dir=$(pwd)

# echo $curr_dir

# Merge each label directory into a label.tar.gz file
for label_dir in data/*/*/*; do
    # echo "$label_dir/"
    tar czf "$label_dir.tar.gz" "$label_dir/"
done

# Shuffle the label directories into data/<node_id>/mnist_png_training_shuffled.tar.gz
for node_id in data/*; do
    # echo "$node_id/"
    for data_dir in $node_id/*; do

        # Get basename
        # echo "$data_dir/"
        dataset=$(basename $data_dir .tar.gz)
        # echo "$dataset"

        # Shuffles data/<node_id>/training/[0-9].tar.gz -> data/<node_id>/mnist_png_training_shuffled.tar.gz
        cd $data_dir
        go run $GOPATH/pkg/mod/github.com/wangkuiyi/gotorch@v0.0.0-20201028015551-9afed2f3ad7b/tool/tarball_merge/tarball_merge.go \
                -out="mnist_png_${dataset}_shuffled.tar.gz" [0-9].tar.gz
        mv "mnist_png_${dataset}_shuffled.tar.gz" ..
        cd $curr_dir
    done
done