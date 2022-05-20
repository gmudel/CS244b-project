package main

import (
	"flads/ds"
	"flads/ds/network"
	"flads/ml"
	"strings"
)

func main() {
	numNodes := 3
	curNodeId := 0
	mlp := &ml.DumbMLProcess{}
	networkTable := map[int]string{ // nodeId : ipAddr
		0: "localhost:7003",
		1: "localhost:7004",
		2: "localhost:7005",
	}

	port := ":" + strings.Split(networkTable[curNodeId], ":")[1]
	net := network.Network{}
	net.Initialize(curNodeId, port, make([]network.Message, 0), networkTable)
	err := net.Listen()
	if err != nil {
		panic("Network not able to listen")
	}

	x := &ds.DistributedSystem{}
	x.Initialize(
		numNodes,
		mlp,
		net,
		3,
	)
	x.Run()
}
