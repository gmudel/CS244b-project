package network

type DumbNetwork[T any] struct {
	numNodes      int
	internalQueue [][]T
}

func (net *DumbNetwork[T]) Initialize(numNodes int) {
	net.numNodes = numNodes
	net.internalQueue = make([][]T, numNodes)
	for i := 0; i < numNodes; i++ {
		net.internalQueue[i] = make([]T, 0)
	}
}

func (net *DumbNetwork[T]) Send(recNodeId int, msg T) {
	net.internalQueue[recNodeId] = append(net.internalQueue[recNodeId], msg)
}

func (net *DumbNetwork[T]) Broadcast(msg T) {
	for i := 0; i < net.numNodes; i++ {
		net.internalQueue[i] = append(net.internalQueue[i], msg)
	}
}

func (net *DumbNetwork[T]) Receive(nodeId int) (valid bool, msg T) {
	if len(net.internalQueue[nodeId]) == 0 {
		var t T
		return false, t
	}
	msg, net.internalQueue[nodeId] = net.internalQueue[nodeId][0], net.internalQueue[nodeId][1:]
	return true, msg
}
