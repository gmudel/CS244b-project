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

func (model *SimpleNN) ZeroGrad() {
	W1Shape := model.net.FC1.Weight.Grad().Shape()
	model.net.FC1.Weight.Grad().SetData(torch.Full(W1Shape, 0, true))

	W2Shape := model.net.FC2.Weight.Grad().Shape()
	model.net.FC2.Weight.Grad().SetData(torch.Full(W2Shape, 0, true))

	W3Shape := model.net.FC3.Weight.Grad().Shape()
	model.net.FC3.Weight.Grad().SetData(torch.Full(W3Shape, 0, true))

	b1Shape := model.net.FC1.Bias.Grad().Shape()
	model.net.FC1.Bias.Grad().SetData(torch.Full(b1Shape, 0, true))

	b2Shape := model.net.FC2.Bias.Grad().Shape()
	model.net.FC2.Bias.Grad().SetData(torch.Full(b2Shape, 0, true))

	b3Shape := model.net.FC3.Bias.Grad().Shape()
	model.net.FC3.Bias.Grad().SetData(torch.Full(b3Shape, 0, true))
	// fmt.Println("out ZeroGrad")
}

func (model *SimpleNN) addGradientsToBuffer() {
	model.lock.Lock()
	defer model.lock.Unlock()
	mlpgrads := MLPGrads{
		model.net.FC1.Weight.Grad(),
		model.net.FC2.Weight.Grad(),
		model.net.FC3.Weight.Grad(),
		model.net.FC1.Bias.Grad(),
		model.net.FC2.Bias.Grad(),
		model.net.FC3.Bias.Grad(),
	}
	model.grads.GradBuffer = append(model.grads.GradBuffer, mlpgrads)
}

func (model *SimpleNN) UpdateModel(incomingGradients Gradients) {
	// Run SGD for each incoming grad
	incomingGrads := incomingGradients.GradBuffer
	model.lock.Lock()
	defer model.lock.Unlock()

	for _, mlpgrad := range incomingGrads {
		newW1 := torch.Sub(model.net.FC1.Weight, mlpgrad.W1, float32(model.lr))
		newW2 := torch.Sub(model.net.FC2.Weight, mlpgrad.W2, float32(model.lr))
		newW3 := torch.Sub(model.net.FC3.Weight, mlpgrad.W3, float32(model.lr))
		model.net.FC1.Weight.SetData(newW1)
		model.net.FC2.Weight.SetData(newW2)
		model.net.FC3.Weight.SetData(newW3)

		newB1 := torch.Sub(model.net.FC1.Bias, mlpgrad.B1, float32(model.lr))
		newB2 := torch.Sub(model.net.FC2.Bias, mlpgrad.B2, float32(model.lr))
		newB3 := torch.Sub(model.net.FC3.Bias, mlpgrad.B3, float32(model.lr))
		model.net.FC1.Bias.SetData(newB1)
		model.net.FC2.Bias.SetData(newB2)
		model.net.FC3.Bias.SetData(newB3)
	}
}

func (model *SimpleNN) GetGradients() (ready bool, gradients Gradients) {
	// flush gradient buffer
	model.lock.Lock()
	defer model.lock.Unlock()

	if len(model.grads.GradBuffer) != 0 {
		GradBufferCopy := make([]MLPGrads, len(model.grads.GradBuffer))
		copy(GradBufferCopy, model.grads.GradBuffer)

		model.grads.GradBuffer = nil
		return true, Gradients{GradBuffer: GradBufferCopy}
	} else {
		return false, Gradients{}
	}
}
