package network

type Message struct {
	Text                      string
}

// Network Interface
type INetwork interface {

    Initialize(port string, queue []Message, nodeIdTable map[int]string) error

    ProcessMessage(msg Message) error

    Send(recNodeId int, msg Message) error

    Broadcast(msg Message) error

    Multicast(nodeIds []int, msg Message) error

    Receive() (Message, error)
}
