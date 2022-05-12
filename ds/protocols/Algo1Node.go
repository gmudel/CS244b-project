package protocols

import (
	"flads/ds/network"
	"flads/ml"
)

type Algo1Node struct {
	id            int
	name          string
	ml            ml.MLProcess
	net           network.Network
	timeoutInSecs int
}

func (node *Algo1Node) Initialize(id int, name string, mlp ml.MLProcess, net network.Network) {
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
