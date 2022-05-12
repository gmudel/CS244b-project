package ml

import (
	"fmt"
	"math/rand"
)

type DumbMLProcess struct {
	model int
}

func (ml DumbMLProcess) Initialize() {
	ml.model = 0
}

func (ml *DumbMLProcess) GetGradients() (bool, Gradients) {
	fmt.Println("getting gradients")
	isReady := rand.Intn(2)
	grads := rand.Intn(100)
	return isReady == 1, Gradients(grads)
}

func (ml *DumbMLProcess) UpdateModel(grads Gradients) {
	fmt.Println("updating model")
	ml.model = int(grads)
}
