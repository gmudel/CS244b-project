package ml

import (
	torch "github.com/wangkuiyi/gotorch"
)

type MLPGrads struct {
	W1, W2, W3 torch.Tensor
	B1, B2, B3 torch.Tensor
}

type Gradients struct {
	GradBuffer []MLPGrads
}

type MLProcess interface {
	GetGradients() (ready bool, gradients Gradients)
	UpdateModel(incomingGradients Gradients)
}
