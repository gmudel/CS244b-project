package main

import (
	"flads/ds"
	"flads/ds/network"
	"flads/ml"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main() {
	numNodes := 3
	curNodeId, err := strconv.Atoi(os.Args[1])
	if err != nil || curNodeId >= numNodes || curNodeId < 0 {
		panic("Cannot get the node id or node id out or range")
	}
	fmt.Println(curNodeId)
	mlp := &ml.DumbMLProcess{}
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
		2,
	)
	x.Run()
}
