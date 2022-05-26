package ml

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	torch "github.com/wangkuiyi/gotorch"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/vision/imageloader"
	"github.com/wangkuiyi/gotorch/vision/models"
	"github.com/wangkuiyi/gotorch/vision/transforms"
	"gocv.io/x/gocv"
)

// func listenForGradRequest() {

// }

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

func (model *SimpleNN) gradStep(grads []MLPGrads) {
	model.lock.Lock()
	defer model.lock.Unlock()
}

func (model *SimpleNN) Train(trainPath, testPath, savePath string) {
	vocab, e := imageloader.BuildLabelVocabularyFromTgz(trainPath)
	if e != nil {
		panic(e)
	}
	defer torch.FinishGC()

	for epoch := 0; epoch < model.epochs; epoch++ {
		var trainLoss float32
		startTime := time.Now()
		trainLoader := MNISTLoader(trainPath, vocab)
		testLoader := MNISTLoader(testPath, vocab)
		totalSamples := 0
		for trainLoader.Scan() {
			data, label := trainLoader.Minibatch()
			totalSamples += int(data.Shape()[0])
			pred := model.net.Forward(data.To(model.device, data.Dtype()))
			loss := F.NllLoss(pred, label.To(model.device, label.Dtype()), torch.Tensor{}, -100, "mean")
			loss.Backward()

			// fmt.Println(type(net.FC1.Weight.Grad()))
			// fmt.Println(reflect.TypeOf(net.FC1.Weight.Grad()))
			// TODO: Gradients for our layers are computed at this point. Send them.
			model.addGradientsToBuffer()
			localGrad := MLPGrads{
				model.net.FC1.Weight.Grad(),
				model.net.FC2.Weight.Grad(),
				model.net.FC3.Weight.Grad(),
				model.net.FC1.Bias.Grad(),
				model.net.FC2.Bias.Grad(),
				model.net.FC3.Bias.Grad(),
			}
			localGradSlice := []MLPGrads{localGrad}
			model.UpdateModel(Gradients{GradBuffer: localGradSlice})
			trainLoss = loss.Item().(float32)
			model.ZeroGrad()
		}
		throughput := float64(totalSamples) / time.Since(startTime).Seconds()
		log.Printf("Train Epoch: %d, Loss: %.4f, throughput: %f samples/sec", epoch, trainLoss, throughput)
		model.Test(testLoader)
	}
	saveModel(model.net, savePath)
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

// func train(net *models.MLPModule, opt torch.Optimizer, trainFn, testFn string, epochs int, save string) {
// 	vocab, e := imageloader.BuildLabelVocabularyFromTgz(trainFn)
// 	if e != nil {
// 		panic(e)
// 	}
// 	defer torch.FinishGC()

// 	for epoch := 0; epoch < epochs; epoch++ {
// 		var trainLoss float32
// 		startTime := time.Now()
// 		trainLoader := MNISTLoader(trainFn, vocab)
// 		testLoader := MNISTLoader(testFn, vocab)
// 		totalSamples := 0
// 		for trainLoader.Scan() {
// 			data, label := trainLoader.Minibatch()
// 			totalSamples += int(data.Shape()[0])
// 			opt.ZeroGrad()
// 			pred := net.Forward(data.To(device, data.Dtype()))
// 			loss := F.NllLoss(pred, label.To(device, label.Dtype()), torch.Tensor{}, -100, "mean")
// 			// fmt.Println(net.FC1.Weight.Grad())
// 			loss.Backward()
// 			// fmt.Println(type(net.FC1.Weight.Grad()))
// 			// fmt.Println(reflect.TypeOf(net.FC1.Weight.Grad()))
// 			// TODO: Gradients for our layers are computed at this point. Send them.
// 			addGradients(net)
// 			opt.Step()
// 			trainLoss = loss.Item().(float32)
// 		}
// 		throughput := float64(totalSamples) / time.Since(startTime).Seconds()
// 		log.Printf("Train Epoch: %d, Loss: %.4f, throughput: %f samples/sec", epoch, trainLoss, throughput)
// 		test(net, testLoader)
// 	}
// 	saveModel(net, save)
// }

// MNISTLoader returns a ImageLoader with MNIST training or testing tgz file
func MNISTLoader(fn string, vocab map[string]int) *imageloader.ImageLoader {
	trans := transforms.Compose(transforms.ToTensor(), transforms.Normalize([]float32{0.1307}, []float32{0.3081}))
	loader, e := imageloader.New(fn, vocab, trans, 64, 64, time.Now().UnixNano(), torch.IsCUDAAvailable(), "gray")
	if e != nil {
		panic(e)
	}
	return loader
}

func (model *SimpleNN) Test(loader *imageloader.ImageLoader) {
	testLoss := float32(0)
	correct := int64(0)
	samples := 0
	for loader.Scan() {
		data, label := loader.Minibatch()
		data = data.To(model.device, data.Dtype())
		label = label.To(model.device, label.Dtype())
		output := model.net.Forward(data)
		loss := F.NllLoss(output, label, torch.Tensor{}, -100, "mean")
		pred := output.Argmax(1)
		testLoss += loss.Item().(float32)
		correct += pred.Eq(label.View(pred.Shape()...)).Sum(map[string]interface{}{"dim": 0, "keepDim": false}).Item().(int64)
		samples += int(label.Shape()[0])
	}
	log.Printf("Test average loss: %.4f, Accuracy: %.2f%%\n",
		testLoss/float32(samples), 100.0*float32(correct)/float32(samples))
}

func saveModel(model *models.MLPModule, modelFn string) {
	log.Println("Saving model to", modelFn)
	f, e := os.Create(modelFn)
	if e != nil {
		log.Fatalf("Cannot create file to save model: %v", e)
	}
	defer f.Close()

	d := torch.NewDevice("cpu")
	model.To(d)
	if e := gob.NewEncoder(f).Encode(model.StateDict()); e != nil {
		log.Fatal(e)
	}
}

func predict(modelFn string, inputs []string) {
	net := loadModel(modelFn)

	for _, in := range inputs {
		for _, pa := range strings.Split(in, ":") {
			fns, e := filepath.Glob(pa)
			if e != nil {
				log.Fatal(e)
			}

			for _, fn := range fns {
				predictFile(fn, net)
			}
		}
	}
}

func loadModel(modelFn string) *models.MLPModule {
	f, e := os.Open(modelFn)
	if e != nil {
		log.Fatal(e)
	}
	defer f.Close()

	states := make(map[string]torch.Tensor)
	if e := gob.NewDecoder(f).Decode(&states); e != nil {
		log.Fatal(e)
	}

	net := models.MLP()
	net.SetStateDict(states)
	return net
}

func predictFile(fn string, m *models.MLPModule) {
	img := gocv.IMRead(fn, gocv.IMReadGrayScale)
	t := transforms.ToTensor().Run(img)
	n := transforms.Normalize([]float32{0.1307}, []float32{0.3081}).Run(t)
	fmt.Println(m.Forward(n).Argmax().Item())
}
