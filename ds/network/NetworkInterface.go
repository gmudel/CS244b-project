package network

import "encoding/gob"

// Network Interface
type INetwork[T any] interface {
	Initialize(nodeId int, port string,
		queue []T, nodeIdTable map[int]string)

	Listen() error

	ListenOnPort(port string) error

	Send(nodeId int, msg T) error

	Broadcast(msg T) error

	ProcessMessage(msg T) error

	Multicast(nodeIds []int, msg T) error

	Receive() (msg T, ok bool)
}

type serializable interface {
	gob.GobEncoder
	gob.GobDecoder
}
