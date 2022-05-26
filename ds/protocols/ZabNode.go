package protocols

import (
	"errors"
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
	net           network.Network[ZabMessage]
	timeoutInSecs int
}

func (node *ZabNode) Initialize(id int, name string, mlp ml.MLProcess, net network.Network[ZabMessage]) {
	node.id = id
	node.name = name
	node.leaderId = 0
	node.ml = mlp
	node.net = net
	node.timeoutInSecs = 2
}

func (node *ZabNode) Run() {
	if ready, grads := node.ml.GetGradients(); ready {
		err := node.net.Send(node.leaderId, ZabMessage{
			id:      node.id,
			msgType: 1,
			myGrads: grads,
		})
		if err != nil {
			util.Logger.Println("Msg send failed")
		}
	}
	util.Debug("here")

	zabMsg, received := node.net.Receive()
	for received {
		if zabMsg.msgType == 1 {
			node.processLeaderMessage(zabMsg)
		} else {
			node.processFollowerMessage(zabMsg)
		}
		zabMsg, received = node.net.Receive()
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
		node.leaderGrads = make([]ml.Gradients, 0)
		err := node.net.Broadcast(ZabMessage{
			id:       node.id,
			msgType:  2,
			allGrads: node.leaderGrads,
		})
		if err != nil {
			util.Logger.Println("Broadcast failed")
		}
	}

}

func (node *ZabNode) processFollowerMessage(msg ZabMessage) {
	assert(msg.msgType == 2, "Wrong Message Type")
	assert(!node.isLeader(), "Leader but getting non-leader messages")
	node.applyGrads(msg.allGrads)
}

func (node *ZabNode) applyGrads(allGrads []ml.Gradients) {
	for _, grad := range allGrads {
		node.ml.UpdateModel(grad)
	}
}

func assert(assertion bool, errorText string) error {
	if !assertion {
		return errors.New(errorText)
	}
	return nil
}
