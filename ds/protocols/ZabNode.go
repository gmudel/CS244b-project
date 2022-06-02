package protocols

import (
	"errors"
	"flads/ds/network"
	"flads/ml"
	"flads/util"
)

type MsgType int16

const (
	PROPOSAL = iota
	ACK
	COMMIT
	WRITE_REQUEST
)

type ZabMessage struct {
	senderId int
	msgType  MsgType
	ZabProposalAckCommit
}

type ZabProposalAckCommit struct {
	Counter int
	Grads   ml.Gradients
}

type ZabNode struct {
	id              int
	numNodes        int
	name            string
	leaderId        int
	ml              ml.MLProcess
	net             network.Network[ZabMessage]
	leaderCounter   int
	proposalCounter int
	commitCounter   int
	history         []*ZabProposalAckCommit
	pendingCommits  map[int]*ZabProposalAckCommit
	ackCounter      []int
}

func (node *ZabNode) Initialize(id int, name string, mlp ml.MLProcess, net network.Network[ZabMessage], numNodes int) {
	node.id = id
	node.numNodes = numNodes
	node.name = name
	node.leaderId = 1
	node.ml = mlp
	node.net = net
	node.leaderCounter = 0
	node.proposalCounter = 0
	node.commitCounter = 0
}

func (node *ZabNode) Run() {
	if ready, localGrads := node.ml.GetGradients(); ready && node.leaderId != node.id {
		err := node.net.Send(node.leaderId, ZabMessage{
			senderId: node.id + 2,
			msgType:  ACK,
			ZabProposalAckCommit: ZabProposalAckCommit{
				Counter: -5,
				Grads:   localGrads,
			},
		})
		if err != nil {
			util.Logger.Println("From ZabNode Run(): Msg send failed", err)
		} else {
			util.Logger.Println("From ZabNode Run(): sent grads to leader from ", node.id)
		}
	}

	zabMsg, received := node.net.Receive()
	for received {
		util.Logger.Println("From ZabNode Run(): received ", zabMsg.msgType, " from  ", zabMsg.senderId)
		switch zabMsg.msgType {
		case PROPOSAL:
			node.handleProposal(&zabMsg.ZabProposalAckCommit)
		case COMMIT:
			node.handleCommit(&zabMsg.ZabProposalAckCommit)
		case WRITE_REQUEST:
			node.handleWriteRequest(&zabMsg.ZabProposalAckCommit)
		case ACK:
			node.handleAck(&zabMsg.ZabProposalAckCommit)
		}
		zabMsg, received = node.net.Receive()
	}
	node.processPendingCommits()
}

/****************************************************************************************************/
/***************************************Follower*****************************************************/
/****************************************************************************************************/

func (node *ZabNode) processPendingCommits() {
	msg, ok := node.pendingCommits[node.commitCounter+1]
	if ok {
		node.commit(msg)
	}
}

func (node *ZabNode) handleProposal(p *ZabProposalAckCommit) {
	node.history = append(node.history, p)
	if p.Counter > node.proposalCounter {
		node.proposalCounter = p.Counter
	}
	node.net.Send(node.leaderId, ZabMessage{
		senderId: node.id,
		msgType:  ACK,
		ZabProposalAckCommit: ZabProposalAckCommit{
			Counter: p.Counter,
			Grads:   p.Grads,
		},
	})
}

func (node *ZabNode) handleCommit(c *ZabProposalAckCommit) {
	for c.Counter > node.commitCounter+1 {
		// wait
		node.pendingCommits[c.Counter] = c
	}
	node.commit(c)
}

func (node *ZabNode) commit(c *ZabProposalAckCommit) {
	node.ml.UpdateModel(c.Grads)
}

/****************************************************************************************************/
/***************************************Leader*******************************************************/
/****************************************************************************************************/

func (node *ZabNode) handleWriteRequest(msg *ZabProposalAckCommit) {
	// propose to all followers in Q
	msg.Counter = node.leaderCounter
	node.leaderCounter += 1
	node.net.Broadcast(ZabMessage{
		senderId:             node.id,
		msgType:              PROPOSAL,
		ZabProposalAckCommit: *msg,
	})
	node.ackCounter = append(node.ackCounter, 0)
}

func (node *ZabNode) handleAck(msg *ZabProposalAckCommit) {
	// increment counter for this message
	node.ackCounter[msg.Counter] += 1
	// if we have a quorum, send commit
	if node.ackCounter[msg.Counter] > node.numNodes/2 {
		node.net.Broadcast(ZabMessage{
			senderId:             node.id,
			msgType:              COMMIT,
			ZabProposalAckCommit: *msg,
		})
	}
	// TODO: check if we want to commit ourselves
	node.commit(msg)
}

/****************************************************************************************************/
/***************************************Helper*******************************************************/
/****************************************************************************************************/

func assert(assertion bool, errorText string) error {
	if !assertion {
		return errors.New(errorText)
	}
	return nil
}
