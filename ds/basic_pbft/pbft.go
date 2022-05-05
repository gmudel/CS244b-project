package basic_pbft

import (
	"ds/network"
	"errors"
	"math"
)

type opState struct {
	text  string
	phase int
}

type Node struct {
	id           int
	name         string
	isPrimary    bool
	logs         []network.Message
	commits      []string
	seqToTextMap map[int]opState
}

func (node Node) Propose(text string) {
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
	network.Broadcast(prePrepareMsg)
}

func (node Node) Run() {
	valid, recMsg := network.Receive(node.id)
	for valid {
		switch recMsg.Phase {
		case 1: // Pre-prepare msg
			node.processPrePrepare(recMsg)
		case 2: // Prepare msg
			node.processPrepare(recMsg)
		default:
		}
		valid, recMsg = network.Receive(node.id)
	}
	// Next Step : Add check for whether committed is reached
}

func (node Node) ReadLog() []network.Message {
	return node.logs
}

func (node Node) ReadCommits() []string {
	return node.commits
}

func (node Node) processPrePrepare(msg network.Message) {
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
		network.Broadcast(prepareMsg)
	}
}

func (node Node) processPrepare(msg network.Message) {
	node.log(msg)
}

func (node Node) log(msg network.Message) {
	node.logs = append(node.logs, msg)
}

func assert(assertion bool, errorText string) error {
	if !assertion {
		return errors.New(errorText)
	}
	return nil
}

func Initialize(numNodes int) {
	network.Initialize(numNodes)
}
