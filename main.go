package main

import (
	"flads/ds/network"
	"flads/ds/protocols"
	"flads/ml"
	"flads/util"
	"flag"
	"fmt"
	"log"
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

func makeModel(trainDir string, nodeId int, useWholeDataset bool) (ml.MLProcess, string, string, string) {
	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		device = torch.NewDevice("cuda")
	} else {
		log.Println("No CUDA found; CPU only")
		device = torch.NewDevice("cpu")
	}

	initializer.ManualSeed(1)

	var trainPath string
	if useWholeDataset {
		trainPath = "./test_data/mnist_png_training_shuffled.tar.gz"
	} else {
		trainPath = fmt.Sprintf("./%s/%d/mnist_png_training_shuffled.tar.gz", trainDir, nodeId)
	}

	fmt.Println(trainPath)
	trainCmd := flag.NewFlagSet("train", flag.ExitOnError)
	trainTar := trainCmd.String("data", trainPath, "data tarball")
	testTar := trainCmd.String("test", "./test_data/mnist_png_testing_shuffled.tar.gz", "data tarball")
	save := trainCmd.String("save", "./ml/mnist_model.gob", "the model file")

	// predictCmd := flag.NewFlagSet("predict", flag.ExitOnError)
	// load := predictCmd.String("load", "/tmp/mnist_model.gob", "the model file")

	lr := trainCmd.Float64("lr", .01, "learning rate")
	epochs := trainCmd.Int("epochs", 10, "number of epochs")

	// if len(os.Args) < 2 {
	// 	fmt.Fprintf(os.Stderr, "Usage: %s needs subcomamnd train or predict\n", os.Args[0])
	// 	os.Exit(1)
	// }

	// trainCmd.Parse(os.Args[2:])
	model := ml.MakeSimpleNN(*lr, *epochs, device)
	return model, *trainTar, *testTar, *save
}

func setup[T any](numNodes int, port string, curNodeId int, networkTable map[int]string, protocol string) network.Network[T] {
	net := network.NetworkClass[T]{}
	net.Initialize(curNodeId, port, make([]T, 0), networkTable, protocol)
	err := net.Listen()
	if err != nil {
		panic("Network not able to listen")
	}
	return &net
}

func main() {

	numNodesPtr := flag.Int("numNodes", 3, "Number of nodes in the network")
	curNodeIdPtr := flag.Int("id", -1, "Current node id")
	leaderIdPtr := flag.Int("leader", -1, "leaderId")
	trainDirPtr := flag.String("trainDir", "data", "directory which contains node_id/mnist_png_training_shuffled.tar.gz")

	flag.Parse()

	numNodes := *numNodesPtr
	curNodeId := *curNodeIdPtr
	leaderId := *leaderIdPtr
	dssMode := ZAB

	util.InitPlotLogger(curNodeId, *trainDirPtr)

	util.InitLogger(curNodeId)
	if curNodeId >= numNodes || curNodeId < 0 {
		panic("Cannot get the node id or node id out or range")
	}

	useWholeDataset := false
	if *trainDirPtr == "all_data" {
		useWholeDataset = true
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
		// 3: "localhost:8012",
	}

	mlp, trainPath, testPath, _ := makeModel(*trainDirPtr, curNodeId, useWholeDataset)
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
		net := setup[protocols.Algo1Message](numNodes, port, curNodeId, networkTable, "tcp")
		node := &protocols.Algo1Node{}
		node.Initialize(curNodeId, strconv.Itoa(curNodeId), mlp, net, net, numNodes, 0)
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
			mlp.Test(testLoader, util.PlotLogger, epoch)
		}
	} else if dssMode == ALGO2 {
		net := setup[protocols.Algo2Message](numNodes, port, curNodeId, networkTable, "tcp")
		node := &protocols.Algo2Node{}
		node.Initialize(curNodeId, strconv.Itoa(curNodeId), mlp, net, net, numNodes, 0)
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
			mlp.Test(testLoader, util.PlotLogger, epoch)
		}
	} else if dssMode == ZAB {
		fmt.Println("running zab")
		net := setup[protocols.ZabMessage](numNodes, port, curNodeId, networkTable, "tcp")
		heartbeatNet := setup[int](numNodes, heartbeatPort, curNodeId, heartbeatNetworkTable, "udp")
		node := &protocols.ZabNode{}
		node.Initialize(curNodeId, strconv.Itoa(curNodeId), mlp, net, heartbeatNet, numNodes, leaderId)
		for epoch := 0; epoch < 10; epoch++ {
			startTime := time.Now()
			totalSamples = 0
			trainLoader := ml.MNISTLoader(trainPath, vocab)
			testLoader := ml.MNISTLoader(testPath, vocab)
			for trainLoader.Scan() {
				samples, trainLoss = mlp.TrainBatch(trainLoader)
				totalSamples += samples
				node.Run()
				time.Sleep(50 * time.Millisecond)
			}
			throughput := float64(totalSamples) / time.Since(startTime).Seconds()
			log.Printf("Train Epoch: %d, Loss: %.4f, throughput: %f samples/sec", epoch, trainLoss, throughput)
			mlp.Test(testLoader, util.PlotLogger, epoch)
			// nodes[curNodeId].Run()
		}
		for {
			node.Run()
		}
	}
}
