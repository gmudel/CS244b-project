package network

type DumbNetwork struct {
	numNodes      int
	internalQueue [][]Message
}

func (net *DumbNetwork) Initialize(numNodes int) {
	net.numNodes = numNodes
	net.internalQueue = make([][]Message, numNodes)
	for i := 0; i < numNodes; i++ {
		net.internalQueue[i] = make([]Message, 0)
	}
}

func (net *DumbNetwork) Send(recNodeId int, msg Message) {
	net.internalQueue[recNodeId] = append(net.internalQueue[recNodeId], msg)
}

func (net *DumbNetwork) Broadcast(msg Message) {
	for i := 0; i < net.numNodes; i++ {
		if i != msg.SendingNodeId {
			net.internalQueue[i] = append(net.internalQueue[i], msg)
		}
	}
}

func (net *DumbNetwork) Receive(nodeId int) (valid bool, msg Message) {
	if len(net.internalQueue[nodeId]) == 0 {
		return false, Message{}
	}
	msg, net.internalQueue[nodeId] = net.internalQueue[nodeId][0], net.internalQueue[nodeId][1:]
	return true, msg
}
