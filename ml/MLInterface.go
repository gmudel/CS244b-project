package ml

import (
	"sync"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/vision/models"
)

type MLPGrads struct {
	W1, W2, W3 torch.Tensor
	B1, B2, B3 torch.Tensor
}

type GradientWrapper struct {
	gradBuffer []MLPGrads
	lock       *sync.Mutex
}

type MLProcess interface {
	GetGradients() (ready bool, grads []MLPGrads)
	UpdateModel(net *models.MLPModule, grads []MLPGrads)
}
