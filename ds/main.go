package main

import (
	"ds/basic_pbft"
	"ds/network"
	"strconv"
)

func main() {
	numNodes := 4
	var net network.Network = network.DumbNetwork{}
	net.Initialize(numNodes)
	nodes := make([]basic_pbft.Node, numNodes)
	for i := 0; i < numNodes; i++ {
		nodes[i].Initialize(i, strconv.Itoa(i), i == 0, net)
	}
	// Next Step : Run the main loop
}
