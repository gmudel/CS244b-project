package network

type Message struct {
	Text string
}

// Network Interface
type INetwork interface {
	Initialize(nodeId int, port string,
		queue []Message, nodeIdTable map[int]string)

	Listen() error

	ListenOnPort(port string) error

	Send(nodeId int, msg Message) error

	Broadcast(msg Message) error

	ProcessMessage(msg Message) error

	Multicast(nodeIds []int, msg Message) error

	Receive() (msg Message, ok bool)
}
