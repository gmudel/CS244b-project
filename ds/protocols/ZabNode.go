package protocols

import (
	"encoding/json"
	"flads/ds/network"
	"flads/ml"
	"flads/util"
)

type ZabMessage struct {
	id       int
	msgType  int
	myGrads  ml.Gradients
	allGrads []ml.Gradients
}

type ZabNode struct {
	id            int
	name          string
	leaderId      int
	leaderGrads   []ml.Gradients
	ml            ml.MLProcess
	net           network.Network
	timeoutInSecs int
}

func (node *ZabNode) Initialize(id int, name string, mlp ml.MLProcess, net network.Network) {
	node.id = id
	node.name = name
	node.leaderId = 0
	node.ml = mlp
	node.net = net
	node.timeoutInSecs = 2
}

func (node *ZabNode) Run() {
	if ready, grads := node.ml.GetGradients(); ready {
		serializedMsg, err := json.Marshal(ZabMessage{
			id:      node.id,
			msgType: 1,
			myGrads: grads,
		})
		if err != nil {
			node.net.Send(node.leaderId, network.Message{
				Text: string(serializedMsg),
			})
		}
	}
	util.Debug("here")

	valid, msg := node.net.Receive(node.id)
	for valid {
		util.Debug("here")
		var zabMsg ZabMessage
		err := json.Unmarshal([]byte(msg.Text), &zabMsg)
		if err != nil {
			if zabMsg.msgType == 1 {
				node.processLeaderMessage(zabMsg)
			} else {
				node.processFollowerMessage(zabMsg)
			}
		}
		valid, msg = node.net.Receive(node.id)
	}
}

func (node *ZabNode) isLeader() bool {
	return node.leaderId == node.id
}

func (node *ZabNode) processLeaderMessage(msg ZabMessage) {
	assert(msg.msgType == 1, "Wrong Message Type")
	assert(len(msg.allGrads) == 0, "Wrong Message")
	assert(node.isLeader(), "Not leader but processing leader messages")
	node.leaderGrads = append(node.leaderGrads, msg.myGrads)
	node.applyGrads(node.leaderGrads)
	if len(node.leaderGrads) > 10 {
		serializedMsg, err := json.Marshal(ZabMessage{
			id:       node.id,
			msgType:  2,
			allGrads: node.leaderGrads,
		})
		node.leaderGrads = make([]ml.Gradients, 0)
		if err != nil {
			node.net.Broadcast(network.Message{
				Text: string(serializedMsg),
			})
		}
	}

}

func (node *ZabNode) processFollowerMessage(msg ZabMessage) {
	assert(msg.msgType == 2, "Wrong Message Type")
	assert(!node.isLeader(), "Leader but getting non-leader messages")
	node.applyGrads(msg.allGrads)
}

func (node *ZabNode) applyGrads(allGrads []ml.Gradients) {
	for grad := range allGrads {
		node.ml.UpdateModel(ml.Gradients(grad))
	}
}
