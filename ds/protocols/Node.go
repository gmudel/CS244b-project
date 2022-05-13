package protocols

import (
	"flads/ds/network"
	"flads/ml"
)

type Node interface {
	Initialize(id int, name string, mlp ml.MLProcess, net network.Network)
	Run()
}
