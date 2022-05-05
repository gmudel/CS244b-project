package network

var n = 3

type Message struct {
	SendingNodeId, Seq, Phase int
	Text                      string
}

var internalQueue [][]Message

func Initialize(numNodes int) {
	n = numNodes
	internalQueue = make([][]Message, numNodes)
}

func Send(recNodeId int, msg Message) {
	internalQueue[recNodeId] = append(internalQueue[recNodeId], msg)
}

func Broadcast(msg Message) {
	for i := 0; i < n; i++ {
		if i != msg.SendingNodeId {
			internalQueue[i] = append(internalQueue[i], msg)
		}
	}
}

func Receive(nodeId int) (valid bool, msg Message) {
	if len(internalQueue[nodeId]) == 0 {
		return false, Message{}
	}
	msg, internalQueue[nodeId] = internalQueue[nodeId][0], internalQueue[nodeId][1:]
	return true, msg
}
