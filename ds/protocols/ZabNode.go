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
	FOLLOWERINFO
	NEWEPOCH
	ACKEPOCH
)

type ZabMessage struct {
	SenderId int
	epoch    int
	MsgType  MsgType
	ZabProposalAckCommit
	ZabViewChange
}

type ZabProposalAckCommit struct {
	epoch   int
	Counter int
	Grads   ml.Gradients
}

type ZabViewChange struct {
	history      []ZabProposalAckCommit
	currentEpoch int
	lastZxid     struct {
		epoch   int
		counter int
	}
}

type ZabNode struct {
	// follower
	id              int
	numNodes        int
	name            string
	leaderId        int
	ml              ml.MLProcess
	net             network.Network[ZabMessage]
	leaderCounter   int
	proposalCounter int
	commitCounter   int
	history         []ZabProposalAckCommit
	pendingCommits  map[int]*ZabProposalAckCommit
	ackCounter      []int
	heartbeats      []time.Time
	timeout         time.Duration
	phase           atomic.Uint8
	acceptedEpoch   int
	currentEpoch    int

	// leader
	followerInfos     map[int]int
	followerAckEpochs map[int]*ZabViewChange
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
	node.acceptedEpoch = 0
	node.currentEpoch = 0
	node.followerInfos = make(map[int]int)
	node.followerAckEpochs = make(map[int]*ZabViewChange)
	for i, _ := range node.heartbeats {
		node.heartbeats[i] = time.Now()
	}
	go node.heartbeatSender()
}

func (node *ZabNode) Run() {
	if node.phase == 0 {
		node.leaderId = (node.leaderId + 1) % node.numNodes
		// follower sends info to leader
		// TODO: Check if leader should send to itself
		node.net.Send(node.leaderId, ZabMessage{
			epoch:   node.acceptedEpoch,
			MsgType: FOLLOWERINFO,
		})

		node.phase = 1
	} else if node.phase == 1 {
		zabMsg, received := node.net.Receive()
		for received {
			switch zabMsg.MsgType {
			case FOLLOWERINFO:
				node.handleFollowerInfo(&zabMsg)
			case NEWEPOCH:
				node.handleNewEpoch(&zabMsg)
			default:
				util.Logger.Println("got message type", zabMsg.MsgType, "in phase 1")
			}
			zabMsg, received = node.net.Receive()
		}
	} else if node.phase == 3 {
		if ready, localGrads := node.ml.GetGradients(); ready {
			err := node.net.Send(node.leaderId, ZabMessage{
				SenderId: node.id,
				MsgType:  WRITE_REQUEST,
				ZabProposalAckCommit: ZabProposalAckCommit{
					Counter: -1, // can be anything
					epoch:   -1, // can be anything
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
			default:
				util.Logger.Println("got message type", zabMsg.MsgType, "in phase 3")
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

// Phase 1
func (node *ZabNode) handleNewEpoch(msg *ZabMessage) {
	if msg.SenderId != node.leaderId {
		return
	}
	// note acceptedEpoch should go to non-volatile mem
	if msg.epoch > node.acceptedEpoch {
		node.acceptedEpoch = msg.epoch
		maxEpoch, maxCounter := node.getLastZxid()
		node.net.Send(node.leaderId, ZabMessage{
			MsgType: ACKEPOCH,
			ZabViewChange: ZabViewChange{
				history:      node.history,
				currentEpoch: node.currentEpoch,
				lastZxid: struct {
					epoch   int
					counter int
				}{maxEpoch, maxCounter},
			},
		})
		node.phase = 2
	} else if msg.epoch < node.acceptedEpoch {
		node.phase = 0
	}
}

func (node *ZabNode) handleAckEpoch(msg *ZabMessage) {

}

// Phase 3
func (node *ZabNode) processPendingCommits() {
	msg, ok := node.pendingCommits[node.commitCounter+1]
	if ok && msg.epoch == node.currentEpoch {
		node.commit(msg)
	}
}

func (node *ZabNode) handleProposal(p *ZabProposalAckCommit) {
	node.history = append(node.history, *p)
	if p.Counter > node.proposalCounter {
		node.proposalCounter = p.Counter
	}
	if p.Counter == -1 {
		fmt.Println(node.id, "has a counter of -1")
	}
	util.Logger.Println("sending ack to leader with counter", p.Counter)
	node.net.Send(node.leaderId, ZabMessage{
		SenderId:             node.id,
		MsgType:              ACK,
		ZabProposalAckCommit: *p,
	})
}

func (node *ZabNode) handleCommit(c *ZabProposalAckCommit) {
	if c.epoch > node.currentEpoch || c.Counter > node.commitCounter+1 {
		// wait
		node.pendingCommits[c.Counter] = c
	} else {
		node.commit(c)
	}
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

// Phase 1
func (node *ZabNode) handleFollowerInfo(msg *ZabMessage) {
	// may need to handle this in phase 3 as well
	if len(node.followerInfos) > node.numNodes/2+1 {
		return
	}
	node.followerInfos[msg.SenderId] = msg.epoch
	if len(node.followerInfos) == node.numNodes/2+1 {
		maxEpoch := 0
		for _, epoch := range node.followerInfos {
			if epoch > maxEpoch {
				maxEpoch = epoch
			}
		}
		newEpoch := maxEpoch + 1
		for nodeId, _ := range node.followerInfos {
			node.net.Send(nodeId, ZabMessage{
				epoch:   newEpoch,
				MsgType: NEWEPOCH,
			})
		}

	}
}

// Phase 3
func (node *ZabNode) handleWriteRequest(msg *ZabProposalAckCommit) {
	// propose to all followers in Q
	msg.Counter = node.leaderCounter
	msg.epoch = node.currentEpoch
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

func (node *ZabNode) getLastZxid() (int, int) {
	maxEpoch := -1
	maxCounter := -1
	for _, val := range node.history {
		if val.epoch > maxEpoch || (val.epoch == maxEpoch && val.Counter > maxCounter) {
			maxEpoch = val.epoch
			maxCounter = val.Counter
		}
	}
	return maxEpoch, maxCounter
}

func assert(assertion bool, errorText string) error {
	if !assertion {
		return errors.New(errorText)
	}
	return nil
}
