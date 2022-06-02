package protocols

// import (
// 	"flads/ds/network"
// 	"flads/ml"
// 	"math/rand"
// 	"sync"
// )

// type AvgServerMessage struct {
// 	model ml.SimpleNN
// }

// type AvgServer struct {
// 	id            int
// 	name          string
// 	model         ml.SimpleNN
// 	net           network.Network[AvgServerMessage]
// 	timeoutInSecs int
// 	total_clients int
// 	train_clients int
// 	num_rounds    int
// 	buf           []ml.SimpleNN
// }

// func (node *AvgServer) Initialize(id int, name string, model ml.SimpleNN, net network.Network[AvgServerMessage]) {
// 	node.id = id
// 	node.name = name
// 	node.model = model
// 	node.net = net
// 	node.timeoutInSecs = 2
// }

// func (node *AvgServer) Run(t int) {
// 	for i := 0; i < t; i++ {
// 		train_client_set := rand.Perm(node.total_clients)[:node.train_clients]
// 		var wg sync.WaitGroup
// 		for _, k := range train_client_set {
// 			wg.Add(1)
// 			go node.ClientUpdate(k, &wg)
// 		}
// 		wg.Wait()
// 	}
// }

// func (node *AvgServer) aggregateWeights() {

// }

// func (node *AvgServer) ClientUpdate(k int, wg *sync.WaitGroup) {
// 	defer wg.Done()

// 	// server sends model state to client
// 	node.net.Send(k, AvgServerMessage{
// 		model: node.model,
// 	})

// 	// client recvs and performs training on a minibatch

// 	// server waits for response, appends to buffer when received

// }
