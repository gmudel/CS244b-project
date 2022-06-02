package main

import (
	"flads/ds/network"
	"flads/ds/protocols"
	"flads/ml"
	"flads/util"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

type dssMode int16

const (
	ALGO1 = iota
	ALGO2
	ZAB
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
	save := trainCmd.String("save", "./ml/mnist_model.gob", "the model file")

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

func setup[T any](numNodes int, port string, curNodeId int, networkTable map[int]string) network.Network[T] {
	net := network.NetworkClass[T]{}
	net.Initialize(curNodeId, port, make([]T, 0), networkTable)
	err := net.Listen()
	if err != nil {
		panic("Network not able to listen")
	}
	return &net
}

func main() {

	numNodes := 2
	curNodeId, err := strconv.Atoi(os.Args[1])
	dssMode := ZAB
	util.InitLogger(curNodeId)
	if err != nil || curNodeId >= numNodes || curNodeId < 0 {
		panic("Cannot get the node id or node id out or range")
	}

	networkTable := map[int]string{ // nodeId : ipAddr
		0: "localhost:7009",
		1: "localhost:7010",
		// 2: "localhost:7007",
	}

	mlp := makeModel()
	util.Logger.Println("made model and began training")

	port := ":" + strings.Split(networkTable[curNodeId], ":")[1]

	if dssMode == ALGO1 {
		net := setup[protocols.Algo1Message](numNodes, port, curNodeId, networkTable)
		nodes := make([]protocols.Node[protocols.Algo1Message], numNodes)
		for i := 0; i < numNodes; i++ {
			nodes[i] = &protocols.Algo1Node{}
			nodes[i].Initialize(i, strconv.Itoa(i), mlp, net, numNodes)
		}
		for {
			for i := 0; i < numNodes; i++ {
				nodes[i].Run()
			}
		}
	} else if dssMode == ALGO2 {
		net := setup[protocols.Algo2Message](numNodes, port, curNodeId, networkTable)
		nodes := make([]protocols.Node[protocols.Algo2Message], numNodes)
		for i := 0; i < numNodes; i++ {
			nodes[i] = &protocols.Algo2Node{}
			nodes[i].Initialize(i, strconv.Itoa(i), mlp, net, numNodes)
		}
		for {
			for i := 0; i < numNodes; i++ {
				nodes[i].Run()
			}
		}
	} else if dssMode == ZAB {
		fmt.Println("running zab")
		net := setup[protocols.ZabMessage](numNodes, port, curNodeId, networkTable)
		nodes := make([]protocols.Node[protocols.ZabMessage], numNodes)
		for i := 0; i < numNodes; i++ {
			nodes[i] = &protocols.ZabNode{}
			nodes[i].Initialize(i, strconv.Itoa(i), mlp, net, numNodes)
		}
		for {
			nodes[curNodeId].Run()
		}
	}
}
