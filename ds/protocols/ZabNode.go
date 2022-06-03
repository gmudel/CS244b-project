package protocols

import (
	"errors"
	"flads/ds/network"
	"flads/ml"
	"flads/util"
	"fmt"
	"time"
)

type MsgType string

const (
	WRITE_REQUEST   = "WRITE_REQUEST"
	PROPOSAL        = "PROPOSAL"
	ACK             = "ACK"
	COMMIT          = "COMMIT"
	FOLLOWERINFO    = "FOLLOWERINFO"
	NEWEPOCH        = "NEWEPOCH"
	ACKEPOCH        = "ACKEPOCH"
	NEWLEADER       = "NEWLEADER"
	ACKNEWLEADER    = "ACKNEWLEADER"
	COMMITNEWLEADER = "COMMITNEWLEADER"
)

type ZabMessage struct {
	SenderId int
	Epoch    int
	MsgType  MsgType
	ZabProposalAckCommit
	ZabViewChange
}

type ZabProposalAckCommit struct {
	Epoch   int
	Counter int
	Grads   ml.Gradients
}

type ZabViewChange struct {
	History      []ZabProposalAckCommit
	CurrentEpoch int
	LastZxid     struct {
		Epoch   int
		Counter int
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
	heartbeatNet    network.Network[int]
	leaderCounter   int
	proposalCounter int
	commitEpoch     int
	commitCounter   int
	history         []ZabProposalAckCommit
	pendingCommits  map[int]map[int]*ZabProposalAckCommit
	ackCounter      []int
	heartbeats      []time.Time
	timeout         time.Duration
	phase           int
	acceptedEpoch   int
	currentEpoch    int
	reset           bool

	// leader
	followerInfos         map[int]int
	followerAckEpochs     map[int]*ZabViewChange
	followerAckNewLeaders map[int]bool
}

func (node *ZabNode) Initialize(id int, name string, mlp ml.MLProcess, net network.Network[ZabMessage], heartbeatNet network.Network[int], numNodes int, leaderId int) {
	node.id = id
	node.numNodes = numNodes
	node.name = name
	node.leaderId = leaderId
	node.ml = mlp
	node.net = net
	node.heartbeatNet = heartbeatNet
	node.leaderCounter = 0
	node.proposalCounter = 0
	node.commitEpoch = 0
	node.commitCounter = 0
	node.pendingCommits = make(map[int]map[int]*ZabProposalAckCommit)
	node.timeout = 5 * time.Second
	node.heartbeats = make([]time.Time, numNodes)
	node.phase = 0
	node.acceptedEpoch = 0
	node.currentEpoch = 0
	node.followerInfos = make(map[int]int)
	node.reset = true
	node.followerAckEpochs = make(map[int]*ZabViewChange)
	node.followerAckNewLeaders = make(map[int]bool)
	// go node.heartbeat()
}

func (node *ZabNode) Run() {
	// TODO: Outer for received {} loop, with nested phase conditions
	if node.reset {
		fmt.Println("in reset")
		node.phase = 0
		node.reset = false
		node.leaderId = (node.leaderId + 1) % node.numNodes
		fmt.Println("leaderId is", node.leaderId)
		// follower sends info to leader
		if node.leaderId != node.id {
			node.SendHelper(node.leaderId, ZabMessage{
				SenderId: node.id,
				Epoch:    node.acceptedEpoch,
				MsgType:  FOLLOWERINFO,
			})
		}
		node.phase = 1
		fmt.Println("Entering phase 1")
	} else {
		if node.phase == 3 {
			if ready, localGrads := node.ml.GetGradients(); ready {
				err := node.SendHelper(node.leaderId, ZabMessage{
					SenderId: node.id,
					MsgType:  WRITE_REQUEST,
					ZabProposalAckCommit: ZabProposalAckCommit{
						Counter: -1, // can be anything
						Epoch:   -1, // can be anything
						Grads:   localGrads,
					},
				})
				if err != nil {
					util.Logger.Println("From ZabNode Run(): Msg send failed", err)
				} else {
					util.Logger.Println("From ZabNode Run(): sent grads to leader from", node.id)
				}
			}
		}

		zabMsg, received := node.ReceiveHelper()
		for received {
			if node.phase == 1 {
				switch zabMsg.MsgType {
				case FOLLOWERINFO:
					node.handleFollowerInfo(&zabMsg)
				case NEWEPOCH:
					node.handleNewEpoch(&zabMsg)
				case ACKEPOCH:
					node.handleAckEpoch(&zabMsg)
				default:
					util.Logger.Println("got message type", zabMsg.MsgType, "in phase 1")
				}
			} else if node.phase == 2 {
				switch zabMsg.MsgType {
				case NEWLEADER:
					fmt.Println("got new leader")
					node.handleNewLeader(&zabMsg)
				case ACKNEWLEADER:
					node.handleAckNewLeader(&zabMsg)
				case COMMITNEWLEADER:
					node.handleCommitNewLeader(&zabMsg)
				}
			} else if node.phase == 3 {
				util.Logger.Println("From ZabNode Run(): received", zabMsg.MsgType, "from", zabMsg.SenderId, "with counter", zabMsg.Counter)
				switch zabMsg.MsgType {
				case PROPOSAL:
					node.handleProposal(&zabMsg)
				case COMMIT:
					node.handleCommit(&zabMsg.ZabProposalAckCommit)
				case WRITE_REQUEST:
					node.handleWriteRequest(&zabMsg.ZabProposalAckCommit)
				case ACK:
					node.handleAck(&zabMsg.ZabProposalAckCommit)
				case FOLLOWERINFO:
					node.handleIncomingFollower(&zabMsg)
				case ACKNEWLEADER:
					node.handleIncomingFollowerAck(&zabMsg)
				default:
					util.Logger.Println("got message type", zabMsg.MsgType, "in phase 3")
				}
				node.processPendingCommits()
			}
			zabMsg, received = node.ReceiveHelper()
		}
	}

}

/****************************************************************************************************/
/***************************************Follower*****************************************************/
/****************************************************************************************************/

// Phase 1
func (node *ZabNode) handleNewEpoch(msg *ZabMessage) {
	fmt.Println("in handleNewEpoch")
	if msg.SenderId != node.leaderId {
		fmt.Println("return early from handleNewEpoch")
		return
	}
	// note acceptedEpoch should go to non-volatile mem
	if msg.Epoch > node.acceptedEpoch {
		fmt.Println("Entering phase 2")
		node.acceptedEpoch = msg.Epoch
		maxEpoch, maxCounter := node.getLastZxid()
		node.SendHelper(node.leaderId, ZabMessage{
			SenderId: node.id,
			MsgType:  ACKEPOCH,
			Epoch:    msg.Epoch,
			ZabViewChange: ZabViewChange{
				History:      node.history,
				CurrentEpoch: node.currentEpoch,
				LastZxid: struct {
					Epoch   int
					Counter int
				}{maxEpoch, maxCounter},
			},
		})
		node.phase = 2
	} else if msg.Epoch < node.acceptedEpoch {
		fmt.Println("phase 0")
		node.phase = 0
	}
	fmt.Println("end handleNewEpoch")
}

// Phase 2
func (node *ZabNode) handleNewLeader(msg *ZabMessage) {
	if node.acceptedEpoch == msg.Epoch {
		fmt.Println("handle new leader", node.acceptedEpoch, msg.Epoch)
		// begin atomic
		// fmt.Println("begin atomic")
		node.currentEpoch = msg.Epoch
		// TODO: Supposed to accept the proposal, doing nothing for now

		node.history = msg.History
		// fmt.Println("end atomic")
		// end atomic
		node.SendHelper(node.leaderId, ZabMessage{
			SenderId: node.id,
			MsgType:  ACKNEWLEADER,
			ZabViewChange: ZabViewChange{
				CurrentEpoch: msg.Epoch,
				History:      node.history,
			},
		})
	} else {
		fmt.Println("Entering phase 0")
		node.phase = 0
	}
}

func (node *ZabNode) handleCommitNewLeader(msg *ZabMessage) {
	// TODO: this is inefficient
	// Might be buggy but check again
	for {
		found := false
		for _, local_zpac := range node.history {
			if local_zpac.Counter == node.commitCounter+1 && local_zpac.Epoch == node.commitEpoch {
				found = true
				node.commit(&local_zpac)
				break
			}
		}
		if !found {
			node.commitEpoch++
			fmt.Println(node.commitEpoch)
			node.commitCounter = 0
			break
		}
	}
	for {
		found := false
		for _, local_zpac := range node.history {
			if local_zpac.Counter == node.commitCounter+1 && local_zpac.Epoch == node.commitEpoch {
				found = true
				node.commit(&local_zpac)
				break
			}
		}
		if !found {
			// node.commitEpoch++
			fmt.Println(node.commitEpoch)
			// node.commitCounter = 0
			break
		}
	}
	// Go to phase 3
	fmt.Println("go to phase 3")
	go node.heartbeat()
	node.phase = 3
}

// Phase 3
func (node *ZabNode) processPendingCommits() {
	msg, ok := node.getFromPendingCommit(node.commitEpoch, node.commitCounter)
	for ok && msg.Epoch == node.currentEpoch {
		node.commit(msg)
		msg, ok = node.getFromPendingCommit(node.commitEpoch, node.commitCounter)
	}
}

func (node *ZabNode) handleProposal(p *ZabMessage) {
	if p.SenderId != node.leaderId {
		return
	}
	node.history = append(node.history, p.ZabProposalAckCommit)
	if p.Counter > node.proposalCounter {
		node.proposalCounter = p.Counter
	}
	if p.Counter == -1 {
		fmt.Println(node.id, "has a counter of -1")
	}
	util.Logger.Println("sending ack to leader with counter", p.Counter)
	node.SendHelper(node.leaderId, ZabMessage{
		SenderId:             node.id,
		MsgType:              ACK,
		ZabProposalAckCommit: p.ZabProposalAckCommit,
	})
}

func (node *ZabNode) handleCommit(c *ZabProposalAckCommit) {
	if c.Epoch > node.commitEpoch || (c.Epoch == node.commitEpoch && c.Counter > node.commitCounter+1) {
		// wait
		fmt.Printf("handle commit wait, c.e: %d node.ce %d c.c %d node.cc %d\n", c.Epoch, node.commitEpoch, c.Counter, node.commitCounter)
		node.setFromPendingCommit(c.Epoch, c.Counter, c)
	} else {
		node.commit(c)
	}
}

func (node *ZabNode) commit(c *ZabProposalAckCommit) {
	node.ml.UpdateModel(c.Grads)
	node.commitCounter++
	// trainBatch.train()
}

// Heartbeat
func (node *ZabNode) heartbeat() {
	for i, _ := range node.heartbeats {
		node.heartbeats[i] = time.Now()
	}
	for {
		if node.id != node.leaderId {
			node.heartbeatNet.Send(node.leaderId, node.id)
			util.Logger.Println("sent heartbeat to leader at", time.Now())
			if time.Time.Sub(time.Now(), node.heartbeats[node.leaderId]) >= node.timeout {
				// go to phase 0
				// panic("follower didn't receive heartbeat")
				node.reset = true
				return
			}
		} else {
			node.heartbeatNet.Broadcast(node.id)
			util.Logger.Println("sent heartbeat to followers at", time.Now())
			count := 0
			for nodeId, t := range node.heartbeats {
				if time.Time.Sub(time.Now(), t) < node.timeout {
					count++
				} else {
					util.Logger.Println("leader received stale heartbeat from node", nodeId, "at time", time.Now())
				}
			}
			if count <= node.numNodes/2 {
				// go to phase 0
				fmt.Println("leader didn't receive enough heartbeats")
				node.reset = true
				return
			}
		}

		sender, received := node.heartbeatNet.Receive()
		for received {
			node.handleHeartbeat(sender)
			sender, received = node.heartbeatNet.Receive()
		}

		time.Sleep(100 * time.Millisecond)
	}
}

func (node *ZabNode) handleHeartbeat(sender int) {
	if node.id != node.leaderId {
		if sender == node.leaderId {
			node.heartbeats[node.leaderId] = time.Now()
			util.Logger.Println("received heartbeat at", time.Now())
		} else {
			util.Logger.Println("follower got heartbeat from non-leader")
		}
	} else {
		t := time.Now()
		node.heartbeats[sender] = t
		util.Logger.Println("received heartbeat from", sender, "at", t)

	}
}

/****************************************************************************************************/
/***************************************Leader*******************************************************/
/****************************************************************************************************/

// Phase 1
func (node *ZabNode) handleFollowerInfo(msg *ZabMessage) {
	// may need to handle this in phase 3 as well
	if len(node.followerInfos) > node.numNodes/2 {
		return
	}
	node.followerInfos[msg.SenderId] = msg.Epoch
	fmt.Println("length of followerinfos:", len(node.followerInfos))
	if len(node.followerInfos) == node.numNodes/2 {
		maxEpoch := 0
		for _, epoch := range node.followerInfos {
			if epoch > maxEpoch {
				maxEpoch = epoch
			}
		}
		fmt.Println("maxepoch", maxEpoch)
		newEpoch := maxEpoch + 1
		for nodeId, _ := range node.followerInfos {
			err := node.SendHelper(nodeId, ZabMessage{
				SenderId: node.id,
				Epoch:    newEpoch,
				MsgType:  NEWEPOCH,
			})
			if err != nil {
				fmt.Println("error in handleFollowerInfo", err)
			}
			fmt.Println("sent to", nodeId)
		}
		node.leaderId = node.id
		node.phase = 1
	}
}

func (node *ZabNode) handleAckEpoch(msg *ZabMessage) {
	fmt.Println("in handleAckEpoch")
	node.followerAckEpochs[msg.SenderId] = &msg.ZabViewChange

	if len(node.followerAckEpochs) == len(node.followerInfos) {
		maxCurrEpoch := -1
		maxEpoch := -1
		maxCounter := -1
		maxNodeId := -1
		for nodeId, followerAckEpoch := range node.followerAckEpochs {
			ce := followerAckEpoch.CurrentEpoch
			e := followerAckEpoch.LastZxid.Epoch
			c := followerAckEpoch.LastZxid.Counter
			if ce > maxCurrEpoch || (ce == maxCurrEpoch && e > maxEpoch) || (ce == maxCurrEpoch && e == maxEpoch && c > maxCounter) {
				maxCurrEpoch = ce
				maxEpoch = e
				maxCounter = c
				maxNodeId = nodeId
			}
		}
		node.history = node.followerAckEpochs[maxNodeId].History
		node.phase = 2
		for nodeId, _ := range node.followerInfos {
			fmt.Println("sending new leader to", nodeId)
			node.SendHelper(nodeId, ZabMessage{
				SenderId: node.id,
				Epoch:    msg.Epoch,
				MsgType:  NEWLEADER,
				ZabViewChange: ZabViewChange{
					History: node.history,
				},
			})
		}
		node.followerInfos = make(map[int]int)
		node.followerAckEpochs = make(map[int]*ZabViewChange)
		fmt.Println("Entering phase 2")
	} else if len(node.followerAckEpochs) > len(node.followerInfos) {
		panic("received more ackepochs than followerInfos")
	}
}

// Phase 2
func (node *ZabNode) handleAckNewLeader(msg *ZabMessage) {
	if len(node.followerAckNewLeaders) > node.numNodes/2 {
		return
	}
	node.followerAckNewLeaders[msg.SenderId] = true
	if len(node.followerAckNewLeaders) == node.numNodes/2 {
		node.net.BroadcastToRest(ZabMessage{
			SenderId: node.id,
			MsgType:  COMMITNEWLEADER,
		})
		// Go to phase 3
		node.phase = 3
		node.currentEpoch = msg.CurrentEpoch
		go node.heartbeat()
		node.followerAckEpochs = make(map[int]*ZabViewChange)
		fmt.Println("Entering phase 3")
	} else {
		fmt.Println("Length of followerAckNewLeader", len(node.followerAckNewLeaders))
	}
}

// Phase 3
func (node *ZabNode) handleWriteRequest(msg *ZabProposalAckCommit) {
	// propose to all followers in Q
	msg.Counter = node.leaderCounter
	msg.Epoch = node.currentEpoch
	node.leaderCounter += 1
	util.Logger.Println("broadcasting with counter", msg.Counter)
	node.net.BroadcastToRest(ZabMessage{
		SenderId:             node.id,
		MsgType:              PROPOSAL,
		ZabProposalAckCommit: *msg,
	})
	node.ackCounter = append(node.ackCounter, 0)
}

func (node *ZabNode) handleAck(msg *ZabProposalAckCommit) {
	if node.ackCounter[msg.Counter] > node.numNodes/2 {
		return
	}
	// increment counter for this message
	util.Logger.Println("counter is", msg.Counter)
	node.ackCounter[msg.Counter] += 1
	// if we have a quorum, send commit
	if node.ackCounter[msg.Counter] == node.numNodes/2 {
		node.net.BroadcastToRest(ZabMessage{
			SenderId:             node.id,
			MsgType:              COMMIT,
			ZabProposalAckCommit: *msg,
		})
		node.commit(msg)
	}
}

func (node *ZabNode) handleIncomingFollower(msg *ZabMessage) {
	if node.id == node.leaderId {
		node.SendHelper(msg.SenderId, ZabMessage{
			SenderId: node.id,
			MsgType:  NEWEPOCH,
			Epoch:    node.currentEpoch,
		})
		node.SendHelper(msg.SenderId, ZabMessage{
			SenderId: node.id,
			MsgType:  NEWLEADER,
			Epoch:    node.currentEpoch,
			ZabViewChange: ZabViewChange{
				History: node.history,
			},
		})
	} else {
		node.handleFollowerInfo(msg)
	}
}

func (node *ZabNode) handleIncomingFollowerAck(msg *ZabMessage) {
	node.SendHelper(msg.SenderId, ZabMessage{
		SenderId: node.id,
		MsgType:  COMMITNEWLEADER,
	})
}

/****************************************************************************************************/
/***************************************Helper*******************************************************/
/****************************************************************************************************/

func (node *ZabNode) getFromPendingCommit(epoch int, counter int) (*ZabProposalAckCommit, bool) {
	nextMap, ok := node.pendingCommits[epoch]
	if !ok {
		return nil, ok
	}
	val, ok2 := nextMap[counter]
	if !ok2 {
		return nil, ok2
	}
	return val, true
}

func (node *ZabNode) SendHelper(receivingNodeId int, msg ZabMessage) error {
	util.Logger.Printf("Sending from %d to %d of type %s at time %s\n", node.id, receivingNodeId, msg.MsgType, time.Now())
	return node.net.Send(receivingNodeId, msg)
}

func (node *ZabNode) ReceiveHelper() (ZabMessage, bool) {
	msg, received := node.net.Receive()
	if received {
		util.Logger.Printf("Receiving from %d of type %s at time %s\n", msg.SenderId, msg.MsgType, time.Now())
	}
	return msg, received
}

func (node *ZabNode) setFromPendingCommit(epoch int, counter int, val *ZabProposalAckCommit) {
	_, ok := node.pendingCommits[epoch]
	if !ok {
		node.pendingCommits[epoch] = make(map[int]*ZabProposalAckCommit)
	}
	node.pendingCommits[epoch][counter] = val
}

func (node *ZabNode) getLastZxid() (int, int) {
	maxEpoch := -1
	maxCounter := -1
	for _, val := range node.history {
		if val.Epoch > maxEpoch || (val.Epoch == maxEpoch && val.Counter > maxCounter) {
			maxEpoch = val.Epoch
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
