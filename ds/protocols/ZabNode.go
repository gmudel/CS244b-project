package protocols

import (
	"errors"
	"flads/ds/network"
	"flads/ml"
	"flads/util"
	"fmt"
	"runtime/internal/atomic"
	"time"
)

type MsgType int16

const (
	WRITE_REQUEST = iota
	PROPOSAL
	ACK
	COMMIT
	HEARTBEAT
)

type ZabMessage struct {
	SenderId int
	MsgType  MsgType
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
	heartbeats      []time.Time
	timeout         time.Duration
	phase           atomic.Uint8
}

func (node *ZabNode) Initialize(id int, name string, mlp ml.MLProcess, net network.Network[ZabMessage], numNodes int) {
	node.id = id
	node.numNodes = numNodes
	node.name = name
	node.leaderId = 0
	node.ml = mlp
	node.net = net
	node.leaderCounter = 0
	node.proposalCounter = 0
	node.commitCounter = 0
	node.pendingCommits = make(map[int]*ZabProposalAckCommit)
	node.timeout = 5 * time.Second
	node.heartbeats = make([]time.Time, numNodes)
	node.phase = 3
	for i, _ := range node.heartbeats {
		node.heartbeats[i] = time.Now()
	}
	go node.heartbeatSender()
}

func (node *ZabNode) Run() {
	if node.phase == 0 {
		node.leaderId++
		node.phase++
	} else if node.phase == 3 {
		if ready, localGrads := node.ml.GetGradients(); ready {
			err := node.net.Send(node.leaderId, ZabMessage{
				SenderId: node.id,
				MsgType:  WRITE_REQUEST,
				ZabProposalAckCommit: ZabProposalAckCommit{
					Counter: -1, // can be anything
					Grads:   localGrads,
				},
			})
			if err != nil {
				util.Logger.Println("From ZabNode Run(): Msg send failed", err)
			} else {
				util.Logger.Println("From ZabNode Run(): sent grads to leader from", node.id)
			}
		}

		// for {
		zabMsg, received := node.net.Receive()
		for received {
			util.Logger.Println("From ZabNode Run(): received", zabMsg.MsgType, "from", zabMsg.SenderId, "with counter", zabMsg.Counter)
			switch zabMsg.MsgType {
			case PROPOSAL:
				node.handleProposal(&zabMsg.ZabProposalAckCommit)
			case COMMIT:
				node.handleCommit(&zabMsg.ZabProposalAckCommit)
			case WRITE_REQUEST:
				node.handleWriteRequest(&zabMsg.ZabProposalAckCommit)
			case ACK:
				node.handleAck(&zabMsg.ZabProposalAckCommit)
			case HEARTBEAT:
				node.handleHeartbeat(&zabMsg)
			}
			zabMsg, received = node.net.Receive()
		}
		// }
		node.processPendingCommits()
	}
}

/****************************************************************************************************/
/***************************************Follower*****************************************************/
/****************************************************************************************************/

// Phase 3
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
	if p.Counter == -1 {
		fmt.Println(node.id, "has a counter of -1")
	}
	util.Logger.Println("sending ack to leader with counter", p.Counter)
	node.net.Send(node.leaderId, ZabMessage{
		SenderId: node.id,
		MsgType:  ACK,
		ZabProposalAckCommit: ZabProposalAckCommit{
			Counter: p.Counter,
			Grads:   p.Grads,
		},
	})
}

func (node *ZabNode) handleCommit(c *ZabProposalAckCommit) {
	if c.Counter > node.commitCounter+1 {
		// wait
		node.pendingCommits[c.Counter] = c
	}
	node.commit(c)
}

func (node *ZabNode) commit(c *ZabProposalAckCommit) {
	node.ml.UpdateModel(c.Grads)
}

// Heartbeat
func (node *ZabNode) heartbeatSender() {
	for {
		if node.id != node.leaderId {
			fmt.Println("in follower")
			node.net.Send(node.leaderId, ZabMessage{
				SenderId: node.id,
				MsgType:  HEARTBEAT,
			})
			if time.Time.Sub(time.Now(), node.heartbeats[node.leaderId]) >= node.timeout {
				// go to phase 0
				panic("follower didn't receive heartbeat")
				node.phase = 0
				return
			}
		} else {
			fmt.Println("in leader")
			node.net.Broadcast(ZabMessage{
				SenderId: node.id,
				MsgType:  HEARTBEAT,
			})
			count := 0
			for _, t := range node.heartbeats {
				if time.Time.Sub(time.Now(), t) < node.timeout {
					count++
				}
			}
			if count <= node.numNodes/2 {
				// go to phase 0
				panic("leader didn't receive heartbeat")
				node.phase = 0
				return
			}
		}

		time.Sleep(time.Second)
	}
}

func (node *ZabNode) handleHeartbeat(msg *ZabMessage) {
	if node.id != node.leaderId {
		if msg.SenderId == node.leaderId {
			node.heartbeats[node.leaderId] = time.Now()
		} else {
			util.Logger.Println("follower got heartbeat from non-leader")
		}
	} else {
		node.heartbeats[msg.SenderId] = time.Now()

	}
}

/****************************************************************************************************/
/***************************************Leader*******************************************************/
/****************************************************************************************************/

// Phase 3
func (node *ZabNode) handleWriteRequest(msg *ZabProposalAckCommit) {
	// propose to all followers in Q
	msg.Counter = node.leaderCounter
	node.leaderCounter += 1
	util.Logger.Println("broadcasting with counter", msg.Counter)
	node.net.Broadcast(ZabMessage{
		SenderId:             node.id,
		MsgType:              PROPOSAL,
		ZabProposalAckCommit: *msg,
	})
	node.ackCounter = append(node.ackCounter, 0)
}

func (node *ZabNode) handleAck(msg *ZabProposalAckCommit) {
	// increment counter for this message
	util.Logger.Println("counter is", msg.Counter)
	node.ackCounter[msg.Counter] += 1
	// if we have a quorum, send commit
	if node.ackCounter[msg.Counter] > node.numNodes/2 {
		node.net.Broadcast(ZabMessage{
			SenderId:             node.id,
			MsgType:              COMMIT,
			ZabProposalAckCommit: *msg,
		})
	}
	// TODO: check if we want to commit ourselves
	// node.commit(msg)
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
