package ml

type Gradients int

type MLProcess interface {
	Initialize()
	GetGradients() (ready bool, grads Gradients)
	UpdateModel(grads Gradients)
}
