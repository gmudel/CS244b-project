package protocols

import (
	"ds/network"
	"errors"
	"math"
)

type opState struct {
	text  string
	phase int
}

type PbftNode struct {
	id           int
	name         string
	isPrimary    bool
	logs         []network.Message
	commits      []string
	seqToTextMap map[int]opState
	network      network.Network
}

func (node PbftNode) Initialize(id int, name string, isPrimary bool, net network.Network) {
	node.id = id
	node.name = name
	node.isPrimary = isPrimary
	node.network = net
}

func (node PbftNode) Propose(text string) {
	var seqNumber int = math.MinInt32
	for n := range node.seqToTextMap {
		if n > seqNumber {
			seqNumber = n
		}
	}
	seqNumber += 1
	var prePrepareMsg network.Message = network.Message{
		SendingNodeId: node.id,
		Seq:           seqNumber,
		Phase:         1,
		Text:          text,
	}
	node.log(prePrepareMsg)
	node.seqToTextMap[seqNumber] = opState{
		text:  text,
		phase: 2,
	}
	node.network.Broadcast(prePrepareMsg)
}

func (node PbftNode) Run() {
	valid, recMsg := node.network.Receive(node.id)
	for valid {
		switch recMsg.Phase {
		case 1: // Pre-prepare msg
			node.processPrePrepare(recMsg)
		case 2: // Prepare msg
			node.processPrepare(recMsg)
		default:
		}
		valid, recMsg = node.network.Receive(node.id)
	}
	// Next Step : Add check for whether committed is reached
}

func (node PbftNode) ReadLog() []network.Message {
	return node.logs
}

func (node PbftNode) ReadCommits() []string {
	return node.commits
}

func (node PbftNode) processPrePrepare(msg network.Message) {
	val, ok := node.seqToTextMap[msg.Seq]
	if ok && val.text != msg.Text {
		return
	} else {
		node.log(msg)
		prepareMsg := network.Message{
			SendingNodeId: node.id,
			Seq:           msg.Seq,
			Phase:         2,
			Text:          msg.Text,
		}
		node.log(prepareMsg)
		node.seqToTextMap[msg.Seq] = opState{
			text:  msg.Text,
			phase: 2,
		}
		node.network.Broadcast(prepareMsg)
	}
}

func (node PbftNode) processPrepare(msg network.Message) {
	node.log(msg)
}

func (node PbftNode) log(msg network.Message) {
	node.logs = append(node.logs, msg)
}

func assert(assertion bool, errorText string) error {
	if !assertion {
		return errors.New(errorText)
	}
	return nil
}
