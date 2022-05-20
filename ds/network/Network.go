package network

//TODO:
//   - Factor classes so nodes have a network interface
//   - Create a node table in main
//   - Test on localhost (hardcoded to main okay)
//   - Create new Listener thread
//   - Channel for Network/Node interface
//   - Negotiate processMessage() interface
//   - Test on AWS

import (
	"encoding/gob"
	"fmt"
	"net"
)

// Network Type
type Network struct {
	nodeId int
	port   string
	queue  []Message

	nodeIdTable map[int]string
}

func (network *Network) Initialize(nodeId int, port string,
	queue []Message, nodeIdTable map[int]string) {

	network.nodeId = nodeId
	network.port = port
	network.queue = queue
	network.nodeIdTable = nodeIdTable

}

// Create node and listen on port
func (network *Network) Listen() error {
	return network.ListenOnPort(network.port)
}

func (network *Network) ListenOnPort(port string) error {

	// Listen on port
	listener, err := net.Listen("tcp", port)
	if err != nil {
		return fmt.Errorf("error opening port: %v ", err)
	}

	go network.listenForever(listener)

	return nil

}

func (network *Network) listenForever(listener net.Listener) {

	// Handle incoming connections concurrently
	for {
		// fmt.Printf("node %d listening...\n", network.nodeId)

		conn, err := listener.Accept()
		if err != nil {
			// No error handling -- if we cannot accept a
			//   connection, we ignore it.
			fmt.Println(err)
		}

		go network.handleConnection(conn)
	}
}

// Creates TCP connection to the address mapped to, encodes the
//   message, then closes the connection
func (network *Network) Send(nodeId int, msg Message) error {

	// Get address of target node
	address, ok := network.nodeIdTable[nodeId]
	if !ok {
		return fmt.Errorf("node %d error: nodeId %d does not exist in network id table.",
			network.nodeId, nodeId)
	}

	// Initialize TCP connection with target
	conn, err := net.Dial("tcp", address)
	if err != nil {
		return err
	}
	defer conn.Close()

	// Encode the message over the connection
	encoder := gob.NewEncoder(conn)
	err = encoder.Encode(&msg) // Might want encoder.Encode(msg)
	return err
}

// Unoptimized broadcast -- simple calls send for every node in the
//   nodeIdTable
func (network *Network) Broadcast(msg Message) error {

	var err error
	for nodeId, _ := range network.nodeIdTable {
		err = network.Send(nodeId, msg)
		if err != nil {
			return err
		}
	}

	return err
}

func (network *Network) Multicast(nodeIds []int, msg Message) error {

	var err error
	for _, nodeId := range nodeIds {
		err = network.Send(nodeId, msg)
		if err != nil {
			return err
		}
	}

	return err
}

// Pops the head of the message queue
func (network *Network) Receive() (msg Message, ok bool) {

	// fmt.Println(network.nodeId, "Receive, ", network.queue)

	if len(network.queue) == 0 {
		return Message{}, false
	}

	msg = network.queue[0]
	network.queue = network.queue[1:]

	return msg, true

}

// handleConnection adds the msg to queue and closes the connection
//   The network class does not check that the message is well-formed,
//   and will add it to the queue for downstream processing
func (network *Network) handleConnection(conn net.Conn) error {

	decoder := gob.NewDecoder(conn)
	var msg Message
	err := decoder.Decode(&msg)

	conn.Close()

	if err == nil {
		// TODO: Locking
		network.queue = append(network.queue, msg)
	}

	return err
}

// Node Type
type INode interface {
	Initialize(network INetwork)

	ProcessMessage(msg Message)
}

type Node struct {
	network INetwork
}

func (node Node) Initialize(network INetwork) {
	node.network = network
}

func (node Node) ProcessMessage(msg Message) {

	// Placeholder -- Prints the message with fields
	fmt.Printf("%+v\n", msg)
}
