package main

import (
	"flads/ds"
	"flads/ds/network"
	"flads/ml"
)

func main() {
	numNodes := 4
	mlp := &ml.DumbMLProcess{}
	net := &network.DumbNetwork{}
	net.Initialize(numNodes)
	x := &ds.DistributedSystem{}
	x.Initialize(
		numNodes,
		mlp,
		net,
		2,
	)
	x.Run()
}
