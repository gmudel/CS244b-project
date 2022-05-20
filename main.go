package main

import (
	"flads/ds"
	"flads/ds/network"
	"flads/ml"
	"flads/util"
	"flag"
	"log"
	"os"
	"strconv"
	"strings"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

var device torch.Device

func makeModel() ml.MLProcess {
	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		device = torch.NewDevice("cuda")
	} else {
		log.Println("No CUDA found; CPU only")
		device = torch.NewDevice("cpu")
	}

	initializer.ManualSeed(1)

	trainCmd := flag.NewFlagSet("train", flag.ExitOnError)
	trainTar := trainCmd.String("data", "./data/mnist_png/mnist_png_training_shuffled.tar.gz", "data tarball")
	testTar := trainCmd.String("test", "./data/mnist_png/mnist_png_testing_shuffled.tar.gz", "data tarball")
	save := trainCmd.String("save", "/tmp/mnist_model.gob", "the model file")

	// predictCmd := flag.NewFlagSet("predict", flag.ExitOnError)
	// load := predictCmd.String("load", "/tmp/mnist_model.gob", "the model file")

	lr := trainCmd.Float64("lr", .01, "learning rate")
	epochs := trainCmd.Int("epochs", 10, "number of epochs")

	// if len(os.Args) < 2 {
	// 	fmt.Fprintf(os.Stderr, "Usage: %s needs subcomamnd train or predict\n", os.Args[0])
	// 	os.Exit(1)
	// }

	trainCmd.Parse(os.Args[2:])
	model := ml.MakeSimpleNN(*lr, *epochs, device)
	go model.Train(*trainTar, *testTar, *save)
	return model
}

func main() {
	numNodes := 3
	curNodeId, err := strconv.Atoi(os.Args[1])
	if err != nil || curNodeId >= numNodes || curNodeId < 0 {
		panic("Cannot get the node id or node id out or range")
	}
	util.Log("hello")
	mlp := makeModel()
	networkTable := map[int]string{ // nodeId : ipAddr
		0: "localhost:7003",
		1: "localhost:7004",
		2: "localhost:7005",
	}

	port := ":" + strings.Split(networkTable[curNodeId], ":")[1]
	net := network.Network{}
	net.Initialize(curNodeId, port, make([]network.Message, 0), networkTable)
	err = net.Listen()
	if err != nil {
		panic("Network not able to listen")
	}

	x := &ds.DistributedSystem{}
	x.Initialize(
		numNodes,
		mlp,
		net,
		1,
	)
	x.Run()
}
