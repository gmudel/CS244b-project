package ml

import (
	"flads/util"
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

	util.Logger.Println("W1 in buffer:", torch.Sum(mlpgrads.W1))
	util.Logger.Println("W2 in buffer:", torch.Sum(mlpgrads.W2))
	util.Logger.Println("W3 in buffer:", torch.Sum(mlpgrads.W3))
	util.Logger.Println("B1 in buffer:", torch.Sum(mlpgrads.B1))
	util.Logger.Println("B2 in buffer:", torch.Sum(mlpgrads.B2))
	util.Logger.Println("B3 in buffer:", torch.Sum(mlpgrads.B3))
	model.grads.GradBuffer = append(model.grads.GradBuffer, mlpgrads)
}

func (model *SimpleNN) UpdateModel(incomingGradients Gradients) {
	// Run SGD for each incoming grad
	incomingGrads := incomingGradients.GradBuffer
	model.lock.Lock()
	defer model.lock.Unlock()

	for _, mlpgrad := range incomingGrads {
		util.Logger.Println("W1 sum:", torch.Sum(mlpgrad.W1))
		util.Logger.Println("W2 sum:", torch.Sum(mlpgrad.W2))
		util.Logger.Println("W3 sum:", torch.Sum(mlpgrad.W3))
		util.Logger.Println("B1 sum:", torch.Sum(mlpgrad.B1))
		util.Logger.Println("B2 sum:", torch.Sum(mlpgrad.B2))
		util.Logger.Println("B3 sum:", torch.Sum(mlpgrad.B3))

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
		util.Logger.Println("length of gradbuffer", len(model.grads.GradBuffer))
		GradBufferCopy := make([]MLPGrads, len(model.grads.GradBuffer))
		copy(GradBufferCopy, model.grads.GradBuffer)
		util.Logger.Println("length of gradbuffer copy", len(GradBufferCopy))

		for _, mlpgrad := range model.grads.GradBuffer {
			util.Logger.Println("gradbuffer W1 sum:", torch.Sum(mlpgrad.W1))
			util.Logger.Println("gradbuffer W2 sum:", torch.Sum(mlpgrad.W2))
			util.Logger.Println("gradbuffer W3 sum:", torch.Sum(mlpgrad.W3))
			util.Logger.Println("gradbuffer B1 sum:", torch.Sum(mlpgrad.B1))
			util.Logger.Println("gradbuffer B2 sum:", torch.Sum(mlpgrad.B2))
			util.Logger.Println("gradbuffer B3 sum:", torch.Sum(mlpgrad.B3))
		}
		model.grads.GradBuffer = nil
		for _, mlpgrad := range GradBufferCopy {
			util.Logger.Println("sending W1 sum:", torch.Sum(mlpgrad.W1))
			util.Logger.Println("sending W2 sum:", torch.Sum(mlpgrad.W2))
			util.Logger.Println("sending W3 sum:", torch.Sum(mlpgrad.W3))
			util.Logger.Println("sending B1 sum:", torch.Sum(mlpgrad.B1))
			util.Logger.Println("sending B2 sum:", torch.Sum(mlpgrad.B2))
			util.Logger.Println("sending B3 sum:", torch.Sum(mlpgrad.B3))
		}
		return true, Gradients{GradBuffer: GradBufferCopy}
	} else {
		return false, Gradients{}
	}
}
