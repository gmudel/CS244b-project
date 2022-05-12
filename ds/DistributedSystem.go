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
}

func (dss *DistributedSystem) Initialize(numNodes int, mlp ml.MLProcess, net network.Network) {
	dss.numNodes = numNodes
	dss.mlp = mlp
	dss.net = net
}

func (dss *DistributedSystem) Run() {
	fmt.Println(dss.numNodes)
	nodes := make([]*protocols.Algo1Node, dss.numNodes)
	for i := 0; i < dss.numNodes; i++ {
		nodes[i] = &protocols.Algo1Node{}
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
