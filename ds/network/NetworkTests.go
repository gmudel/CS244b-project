package network

import (
	"fmt"
	"strings"
	"time"
)

func TestPairwiseMessages() {

	networkTable := map[int]string{ // nodeId : ipAddr
		0: "localhost:7003",
		1: "localhost:7004",
		2: "localhost:7005"}

	var nodeTable map[int]*Network = createNodesFromTable(networkTable)

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

func createNodesFromTable(networkTable map[int]string) map[int]*Network {

	nodeTable := make(map[int]*Network)

	for nodeId, address := range networkTable {
		port := ":" + strings.Split(address, ":")[1]

		newNode := Network{}
		newNode.Initialize(nodeId, port, make([]Message, 0), networkTable)

		nodeTable[nodeId] = &newNode
	}

	return nodeTable
}

// Synchronized -- test that messages are being passed across all edges
func sendAllPairwiseMessages(nodeTable map[int]*Network) {

	for srcId, srcNode := range nodeTable {
		for dstId, dstNode := range nodeTable {

			if srcId != dstId {

				fmt.Printf("Sending from %d to %d\n", srcId, dstId)

				msgText := fmt.Sprintf("(node %d -> %d)", srcId, dstId)
				msg := Message{msgText}

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
