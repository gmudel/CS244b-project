package protocols

import (
	"flads/ds/network"
	"flads/ml"
	"flads/util"
	"time"
)

type Algo2Message struct {
	id    int
	grads ml.Gradients
}

type Algo2Node struct {
	id            int
	name          string
	ml            ml.MLProcess
	net           network.Network[Algo2Message]
	timeoutInSecs int
}

func (node *Algo2Node) Initialize(id int, name string, mlp ml.MLProcess, net network.Network[Algo2Message]) {
	node.id = id
	node.name = name
	node.ml = mlp
	node.net = net
	node.timeoutInSecs = 2
}

func (node *Algo2Node) Run() {
	allGrads := make([]ml.Gradients, 0)
	if ready, grads := node.ml.GetGradients(); ready {
		allGrads = append(allGrads, grads)
		err := node.net.Broadcast(Algo2Message{
			id:    node.id,
			grads: grads,
		})

		if err == nil {
			util.Logger.Println("broadcast failed")
		}
	}
	msg, received := node.net.Receive()
	for received {
		time.Sleep(time.Second)
		var grads ml.Gradients = msg.grads
		allGrads = append(allGrads, grads)
		msg, received = node.net.Receive()
	}
	for _, grads := range allGrads {
		node.ml.UpdateModel(grads)
	}
}
