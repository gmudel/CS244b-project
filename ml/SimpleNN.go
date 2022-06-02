package ml

import (
	"sync"

	torch "github.com/wangkuiyi/gotorch"
	models "github.com/wangkuiyi/gotorch/vision/models"
)

// simple MLP NN, trained with vanilla SGD
type SimpleNN struct {
	net    *models.MLPModule
	lr     float64
	epochs int
	grads  Gradients
	lock   sync.Mutex
	device torch.Device
}

func MakeSimpleNN(lr float64, epochs int, device torch.Device) *SimpleNN {
	nn := SimpleNN{models.MLP(), lr, epochs, Gradients{}, sync.Mutex{}, device}
	nn.net.To(device)
	return &nn
}
