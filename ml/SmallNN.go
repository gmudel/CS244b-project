package ml

import (
	"sync"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
	models "github.com/wangkuiyi/gotorch/vision/models"
)

// simple MLP NN, trained with vanilla SGD
type SmallNN struct {
	net    *models.MLPModule
	lr     float64
	epochs int
	grads  Gradients
	lock   sync.Mutex
	device torch.Device
}

func smallMLP() *models.MLPModule {
	r := &models.MLPModule{
		FC1: nn.Linear(28*28, 128, true),
		FC2: nn.Linear(128, 128, true),
		FC3: nn.Linear(128, 10, true)}
	r.Init(r)
	return r
}

func MakeSmallNN(lr float64, epochs int, device torch.Device) *SmallNN {
	nn := SmallNN{smallMLP(), lr, epochs, Gradients{}, sync.Mutex{}, device}
	nn.net.To(device)
	return &nn
}

func (model *SmallNN) ZeroGrad() {
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

func (model *SmallNN) addGradientsToBuffer() {
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

	// hack to copy
	W1Shape := model.net.FC1.Weight.Grad().Shape()
	W1Copy := torch.Full(W1Shape, 0, true)
	W1Copy = torch.Add(W1Copy, mlpgrads.W1, 1.)

	W2Shape := model.net.FC2.Weight.Grad().Shape()
	W2Copy := torch.Full(W2Shape, 0, true)
	W2Copy = torch.Add(W2Copy, mlpgrads.W2, 1.)

	W3Shape := model.net.FC3.Weight.Grad().Shape()
	W3Copy := torch.Full(W3Shape, 0, true)
	W3Copy = torch.Add(W3Copy, mlpgrads.W3, 1.)

	B1Shape := model.net.FC1.Bias.Grad().Shape()
	B1Copy := torch.Full(B1Shape, 0, true)
	B1Copy = torch.Add(B1Copy, mlpgrads.B1, 1.)

	B2Shape := model.net.FC2.Bias.Grad().Shape()
	B2Copy := torch.Full(B2Shape, 0, true)
	B2Copy = torch.Add(B2Copy, mlpgrads.B2, 1.)

	B3Shape := model.net.FC3.Bias.Grad().Shape()
	B3Copy := torch.Full(B3Shape, 0, true)
	B3Copy = torch.Add(B3Copy, mlpgrads.B3, 1.)

	mlpgradsCopy := MLPGrads{
		W1Copy,
		W2Copy,
		W3Copy,
		B1Copy,
		B2Copy,
		B3Copy,
	}

	model.grads.GradBuffer = append(model.grads.GradBuffer, mlpgradsCopy)
}

func (model *SmallNN) UpdateModel(incomingGradients Gradients) {
	// Run SGD for each incoming grad
	incomingGrads := incomingGradients.GradBuffer
	// model.lock.Lock()
	// defer model.lock.Unlock()

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

func (model *SmallNN) GetGradients() (ready bool, gradients Gradients) {
	// flush gradient buffer
	// model.lock.Lock()
	// defer model.lock.Unlock()

	if len(model.grads.GradBuffer) != 0 {
		GradBufferCopy := make([]MLPGrads, len(model.grads.GradBuffer))
		copy(GradBufferCopy, model.grads.GradBuffer)

		model.grads.GradBuffer = nil
		return true, Gradients{GradBuffer: GradBufferCopy}
	} else {
		return false, Gradients{}
	}
}
