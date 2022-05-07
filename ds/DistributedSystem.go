package ds

import (
	"flads/ds/network"
	"flads/ds/protocols"
	"flads/ml"
	"strconv"
)

type DistributedSystem struct {
	ml ml.MLProcess
}

func (ds DistributedSystem) Run() {
	numNodes := 4
	var net network.Network = network.DumbNetwork{}
	net.Initialize(numNodes)
	nodes := make([]protocols.PbftNode, numNodes)
	for i := 0; i < numNodes; i++ {
		nodes[i].Initialize(i, strconv.Itoa(i), i == 0, net)
	}
	// Next Step : Run the main loop
}
