package ml

import (
	"sync"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
	models "github.com/wangkuiyi/gotorch/vision/models"
)

func smallMLP() *models.MLPModule {
	r := &models.MLPModule{
		FC1: nn.Linear(28*28, 10, true),
	}
	r.Init(r)
	return r
}

func MakeSmallNN(lr float64, epochs int, device torch.Device) *SimpleNN {
	nn := SimpleNN{smallMLP(), lr, epochs, Gradients{}, sync.Mutex{}, device}
	nn.net.To(device)
	return &nn
}
