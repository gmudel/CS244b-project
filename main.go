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
	"time"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/initializer"
	"github.com/wangkuiyi/gotorch/vision/imageloader"
)

type dssMode int16

const (
	ALGO1 = iota
	ALGO2
	ZAB
)

var device torch.Device

func makeModel() (ml.MLProcess, string, string, string) {
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
	model := ml.MakeSmallNN(*lr, *epochs, device)
	return model, *trainTar, *testTar, *save
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

	numNodes := 3
	curNodeId, err := strconv.Atoi(os.Args[1])
	dssMode := ZAB
	util.InitLogger(curNodeId)
	if err != nil || curNodeId >= numNodes || curNodeId < 0 {
		panic("Cannot get the node id or node id out or range")
	}

	networkTable := map[int]string{ // nodeId : ipAddr
		0: "localhost:7009",
		1: "localhost:7010",
		2: "localhost:7011",
		// 3: "localhost:7012",
	}

	heartbeatNetworkTable := map[int]string{ // nodeId : ipAddr
		0: "localhost:8009",
		1: "localhost:8010",
		2: "localhost:8011",
		// 3: "localhost:7012",
	}

	mlp, trainPath, testPath, _ := makeModel()
	util.Logger.Println("made model and began training")
	vocab, e := imageloader.BuildLabelVocabularyFromTgz(trainPath)
	if e != nil {
		panic(e)
	}

	var totalSamples int
	var samples int
	var trainLoss float32

	port := ":" + strings.Split(networkTable[curNodeId], ":")[1]
	heartbeatPort := ":" + strings.Split(heartbeatNetworkTable[curNodeId], ":")[1]

	if dssMode == ALGO1 {
		net := setup[protocols.Algo1Message](numNodes, port, curNodeId, networkTable)
		node := &protocols.Algo1Node{}
		node.Initialize(curNodeId, strconv.Itoa(curNodeId), mlp, net, net, numNodes)
		for epoch := 0; epoch < 10; epoch++ {
			startTime := time.Now()
			totalSamples = 0
			trainLoader := ml.MNISTLoader(trainPath, vocab)
			testLoader := ml.MNISTLoader(testPath, vocab)
			for trainLoader.Scan() {
				samples, trainLoss = mlp.TrainBatch(trainLoader)
				totalSamples += samples
				node.Run()
			}
			throughput := float64(totalSamples) / time.Since(startTime).Seconds()
			log.Printf("Train Epoch: %d, Loss: %.4f, throughput: %f samples/sec", epoch, trainLoss, throughput)
			mlp.Test(testLoader)
		}
	} else if dssMode == ALGO2 {
		net := setup[protocols.Algo2Message](numNodes, port, curNodeId, networkTable)
		node := &protocols.Algo2Node{}
		node.Initialize(curNodeId, strconv.Itoa(curNodeId), mlp, net, net, numNodes)
		for epoch := 0; epoch < 10; epoch++ {
			startTime := time.Now()
			totalSamples = 0
			trainLoader := ml.MNISTLoader(trainPath, vocab)
			testLoader := ml.MNISTLoader(testPath, vocab)
			for trainLoader.Scan() {
				samples, trainLoss = mlp.TrainBatch(trainLoader)
				totalSamples += samples
				node.Run()
			}
			throughput := float64(totalSamples) / time.Since(startTime).Seconds()
			log.Printf("Train Epoch: %d, Loss: %.4f, throughput: %f samples/sec", epoch, trainLoss, throughput)
			mlp.Test(testLoader)
		}
	} else if dssMode == ZAB {
		fmt.Println("running zab")
		net := setup[protocols.ZabMessage](numNodes, port, curNodeId, networkTable)
		heartbeatNet := setup[protocols.ZabMessage](numNodes, heartbeatPort, curNodeId, heartbeatNetworkTable)
		node := &protocols.ZabNode{}
		node.Initialize(curNodeId, strconv.Itoa(curNodeId), mlp, net, heartbeatNet, numNodes)
		for epoch := 0; epoch < 10; epoch++ {
			startTime := time.Now()
			totalSamples = 0
			trainLoader := ml.MNISTLoader(trainPath, vocab)
			testLoader := ml.MNISTLoader(testPath, vocab)
			for trainLoader.Scan() {
				samples, trainLoss = mlp.TrainBatch(trainLoader)
				totalSamples += samples
				node.Run()
				// time.Sleep(time.Second)
			}
			throughput := float64(totalSamples) / time.Since(startTime).Seconds()
			log.Printf("Train Epoch: %d, Loss: %.4f, throughput: %f samples/sec", epoch, trainLoss, throughput)
			mlp.Test(testLoader)
			// nodes[curNodeId].Run()
		}
		for {
			node.Run()
		}
	}
}
