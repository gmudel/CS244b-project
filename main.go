package main

import (
	"flads/ds"
	"flads/ds/network"
	"flads/ml"
)

func main() {
	mlp := &ml.DumbMLProcess{}
	net := &network.DumbNetwork{}
	x := &ds.DistributedSystem{}
	x.Initialize(
		4,
		mlp,
		net,
	)
	x.Run()
}
