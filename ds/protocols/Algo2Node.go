package protocols

import (
	"encoding/json"
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
	net           network.Network
	timeoutInSecs int
}

func (node *Algo2Node) Initialize(id int, name string, mlp ml.MLProcess, net network.Network) {
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
		serializedMsg, err := json.Marshal(Algo2Message{
			id:    node.id,
			grads: grads,
		})
		// util.Logger.Println(err)
		if err == nil {
			// util.Logger.Println("preparing msg")
			node.net.Broadcast(network.Message{
				Text: string(serializedMsg),
			})
		}
	}
	msg, received := node.net.Receive()
	for received {
		util.Logger.Println("msg.text")
		util.Logger.Println(msg.Text)
		time.Sleep(time.Second)
		var grads ml.Gradients
		err := json.Unmarshal([]byte(msg.Text), &grads)
		if err != nil {
			allGrads = append(allGrads, grads)
		}
		msg, received = node.net.Receive()
	}
	for _, grads := range allGrads {
		node.ml.UpdateModel(grads)
	}
}
