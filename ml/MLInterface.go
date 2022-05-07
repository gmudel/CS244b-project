package ml

type Gradients struct {
	gradients []float32
}

type MLProcess interface {
	GetGradients() (ready bool, grads Gradients)
	UpdateModel(grads Gradients)
}
