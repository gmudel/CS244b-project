package network

type Message struct {
	SendingNodeId, Seq, Phase int
	Text                      string
}

type Network interface {
	Initialize(numNodes int)

	Send(recNodeId int, msg Message)

	Broadcast(msg Message)

	Receive(nodeId int) (valid bool, msg Message)
}
