package ds

import (
	"flads/ds/network"
	"flads/ds/protocols"
	"flads/ml"
	"fmt"
	"strconv"
)

type DistributedSystem struct {
	numNodes int
	mlp      ml.MLProcess
	net      network.Network
	mode     int
}

func (dss *DistributedSystem) Initialize(numNodes int, mlp ml.MLProcess, net network.Network, mode int) {
	dss.numNodes = numNodes
	dss.mlp = mlp
	dss.net = net
	dss.mode = mode
}

func (dss *DistributedSystem) Run() {
	fmt.Println(dss.numNodes)
	nodes := make([]protocols.Node, dss.numNodes)

	for i := 0; i < dss.numNodes; i++ {
		if dss.mode == 1 {
			nodes[i] = &protocols.Algo1Node{}
		} else if dss.mode == 2 {
			nodes[i] = &protocols.Algo2Node{}
		} else {
			nodes[i] = &protocols.ZabNode{}
		}

		nodes[i].Initialize(i, strconv.Itoa(i), dss.mlp, dss.net)
	}

	// Step 1: Let every node run in order on same process
	for {
		for i := 0; i < dss.numNodes; i++ {
			nodes[i].Run()
		}
	}

	// Next Step : Run the main loop
}
