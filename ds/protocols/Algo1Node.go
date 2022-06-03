package protocols

import (
	"flads/ds/network"
	"flads/ml"
)

type Algo1Message struct {
}

type Algo1Node struct {
	id            int
	name          string
	ml            ml.MLProcess
	net           network.Network[Algo1Message]
	timeoutInSecs int
}

func (node *Algo1Node) Initialize(id int, name string, mlp ml.MLProcess, net network.Network[Algo1Message], heartbeatNet network.Network[Algo1Message], numNodes int, leaderId int) {
	node.id = id
	node.name = name
	node.ml = mlp
	node.net = net
	node.timeoutInSecs = 2
}

func (node *Algo1Node) Run() {
	if ready, grads := node.ml.GetGradients(); ready {
		node.ml.UpdateModel(grads)
	}
}
