package protocols

import (
	"flads/ds/network"
	"flads/ml"
)

type Node[T any] interface {
	Initialize(id int, name string, mlp ml.MLProcess, net network.Network[T])
	Run()
}
