package ml

import (
	"flads/util"
	"math/rand"
)

type DumbMLProcess struct {
	model int
}

func (ml DumbMLProcess) Initialize() {
	ml.model = 0
}

func (ml *DumbMLProcess) GetGradients() (bool, Gradients) {
	// util.Debug("getting gradients")
	isReady := rand.Intn(2)
	grads := Gradients{}
	return isReady == 1, Gradients(grads)
}

func (ml *DumbMLProcess) UpdateModel(grads Gradients) {
	util.Debug("updating model")
	ml.model = ml.model + 1
}
