package network

import (
	"fmt"
	"strings"
	"time"
)

func TestPairwiseMessages[T any]() {

	networkTable := map[int]string{ // nodeId : ipAddr
		0: "localhost:7003",
		1: "localhost:7004",
		2: "localhost:7005"}

	var nodeTable map[int]*NetworkClass[T] = createNodesFromTable[T](networkTable)

	fmt.Println(nodeTable)

	// Listen
	for _, node := range nodeTable {
		fmt.Println(node)
		err := node.Listen()
		if err != nil {
			fmt.Println("Error", err)
		}
		time.Sleep(1 * time.Second)
	}

	time.Sleep(2 * time.Second)

	fmt.Println("Sending pairwise Messages:")
	sendAllPairwiseMessages(nodeTable)
}

func createNodesFromTable[T any](networkTable map[int]string) map[int]*NetworkClass[T] {

	nodeTable := make(map[int]*NetworkClass[T])

	for nodeId, address := range networkTable {
		port := ":" + strings.Split(address, ":")[1]

		newNode := NetworkClass[T]{}
		newNode.Initialize(nodeId, port, make([]T, 0), networkTable, "tcp")

		nodeTable[nodeId] = &newNode
	}

	return nodeTable
}

// Synchronized -- test that messages are being passed across all edges
func sendAllPairwiseMessages[T any](nodeTable map[int]*NetworkClass[T]) {

	for srcId, srcNode := range nodeTable {
		for dstId, dstNode := range nodeTable {

			if srcId != dstId {

				fmt.Printf("Sending from %d to %d\n", srcId, dstId)

				// msgText := fmt.Sprintf("(node %d -> %d)", srcId, dstId)
				var msg T

				srcNode.Send(dstId, msg)

				time.Sleep(1 * time.Second)
				// fmt.Println("getting message:")

				recMsg, ok := dstNode.Receive()

				if !ok {
					fmt.Printf("dstNode %d has no messages\n", dstId)
				}
				fmt.Println("Success: ", recMsg)
			}
		}
	}
}
