package ml

import (
	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/vision/models"
)

type MLPGrads struct {
	W1, W2, W3 torch.Tensor
	B1, B2, B3 torch.Tensor
}

type Gradients struct {
	gradBuffer []MLPGrads
}

type MLProcess interface {
	GetGradients() (ready bool, grads []MLPGrads)
	UpdateModel(net *models.MLPModule, grads []MLPGrads)
}
