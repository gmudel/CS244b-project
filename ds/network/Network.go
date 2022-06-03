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
	"flads/util"
	"fmt"
	"net"

	"github.com/cenkalti/backoff/v4"
)

// Network Type
type NetworkClass[T any] struct {
	nodeId int
	port   string
	queue  []T

	nodeIdTable map[int]string
}

func (network *NetworkClass[T]) Initialize(nodeId int, port string,
	queue []T, nodeIdTable map[int]string) {

	network.nodeId = nodeId
	network.port = port
	network.queue = queue
	network.nodeIdTable = nodeIdTable

}

// Create node and listen on port
func (network *NetworkClass[T]) Listen() error {
	return network.ListenOnPort(network.port)
}

func (network *NetworkClass[T]) ListenOnPort(port string) error {

	// Listen on port
	listener, err := net.Listen("tcp", port)
	if err != nil {
		return fmt.Errorf("error opening port: %v ", err)
	}

	go network.listenForever(listener)

	return nil

}

func (network *NetworkClass[T]) listenForever(listener net.Listener) {

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
func (network *NetworkClass[T]) Send(nodeId int, msg T) error {

	// Get address of target node
	address, ok := network.nodeIdTable[nodeId]
	if !ok {
		return fmt.Errorf("node %d error: nodeId %d does not exist in network id table.",
			network.nodeId, nodeId)
	}

	// Initialize TCP connection with target
	backoff.Retry(func() error {
		conn, err := net.Dial("tcp", address)
		// defer conn.Close()
		if err != nil {
			// if this is a temporary error, then retry
			if terr, ok := err.(TemporaryError); ok && terr.Temporary() {
				return err
			}
		}
		// if we were successful, or there was a non-temporary error, fail
		// Encode the message over the connection
		encoder := gob.NewEncoder(conn)
		err = encoder.Encode(msg) // Might want encoder.Encode(msg)
		if err != nil {
			util.Logger.Println("error encoding msg")
		}
		return nil
	}, backoff.NewExponentialBackOff())
	return nil

	// util.Logger.Println("In Send(), sent msg", msg)
}

// Unoptimized broadcast -- simple calls send for every node in the
//   nodeIdTable
func (network *NetworkClass[T]) Broadcast(msg T) error {

	var err error
	for nodeId, _ := range network.nodeIdTable {
		// if nodeId != network.nodeId {
		err = network.Send(nodeId, msg)
		if err != nil {
			util.Logger.Println("Error sending to ", nodeId, "error msg:", err)
		}
		// }
	}

	return err
}

func (network *NetworkClass[T]) Multicast(nodeIds []int, msg T) error {

	var err error
	for _, nodeId := range nodeIds {
		err = network.Send(nodeId, msg)
		if err != nil {
			return err
		}
	}

	return err
}

func (network *NetworkClass[T]) Receive() (msg T, ok bool) {

	// fmt.Println(network.nodeId, "Receive, ", network.queue)

	// util.Logger.Println("net queue length is ", len(network.queue))
	if len(network.queue) == 0 {
		var t T
		return t, false
	}

	msg = network.queue[0]
	network.queue = network.queue[1:]

	// util.Logger.Println("In Receive(), received msg", msg)
	return msg, true

}

// handleConnection adds the msg to queue and closes the connection
//   The network class does not check that the message is well-formed,
//   and will add it to the queue for downstream processing
func (network *NetworkClass[T]) handleConnection(conn net.Conn) error {

	decoder := gob.NewDecoder(conn)
	var msg T
	err := decoder.Decode(&msg)

	conn.Close()

	if err == nil {
		// TODO: Locking
		network.queue = append(network.queue, msg)
		// util.Logger.Println("msg contains ", msg)
	} else {
		util.Logger.Println("Error in handleConnection: ", err)
	}

	return err
}
