package ml

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	torch "github.com/wangkuiyi/gotorch"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/vision/imageloader"
	models "github.com/wangkuiyi/gotorch/vision/models"
	"github.com/wangkuiyi/gotorch/vision/transforms"
	"gocv.io/x/gocv"
)

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
			// localGrad := MLPGrads{
			// 	model.net.FC1.Weight.Grad(),
			// 	model.net.FC2.Weight.Grad(),
			// 	model.net.FC3.Weight.Grad(),
			// 	model.net.FC1.Bias.Grad(),
			// 	model.net.FC2.Bias.Grad(),
			// 	model.net.FC3.Bias.Grad(),
			// }
			// localGradSlice := []MLPGrads{localGrad}
			// model.UpdateModel(Gradients{GradBuffer: localGradSlice})
			trainLoss = loss.Item().(float32)
			model.ZeroGrad()
		}
		throughput := float64(totalSamples) / time.Since(startTime).Seconds()
		log.Printf("Train Epoch: %d, Loss: %.4f, throughput: %f samples/sec", epoch, trainLoss, throughput)
		model.Test(testLoader)
	}
	saveModel(model.net, savePath)
}

func (model *SimpleNN) TrainBatch(trainLoader *imageloader.ImageLoader) (int, float32) {

	data, label := trainLoader.Minibatch()
	numSamples := int(data.Shape()[0])
	pred := model.net.Forward(data.To(model.device, data.Dtype()))
	loss := F.NllLoss(pred, label.To(model.device, label.Dtype()), torch.Tensor{}, -100, "mean")
	loss.Backward()

	// fmt.Println(type(net.FC1.Weight.Grad()))
	// fmt.Println(reflect.TypeOf(net.FC1.Weight.Grad()))
	// TODO: Gradients for our layers are computed at this point. Send them.
	model.addGradientsToBuffer()
	// localGrad := MLPGrads{
	// 	model.net.FC1.Weight.Grad(),
	// 	model.net.FC2.Weight.Grad(),
	// 	model.net.FC3.Weight.Grad(),
	// 	model.net.FC1.Bias.Grad(),
	// 	model.net.FC2.Bias.Grad(),
	// 	model.net.FC3.Bias.Grad(),
	// }
	// localGradSlice := []MLPGrads{localGrad}
	// model.UpdateModel(Gradients{GradBuffer: localGradSlice})
	trainLoss := loss.Item().(float32)
	model.ZeroGrad()
	return numSamples, trainLoss
}

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

// func (model *SimpleNN) sumModelWeights(otherModel *SimpleNN) {
// 	model.net.FC1.Weight.SetData(torch.Add(model.net.FC1.Weight, otherModel.net.FC1.Weight, 1.))
// 	model.net.FC2.Weight.SetData(torch.Add(model.net.FC2.Weight, otherModel.net.FC2.Weight, 1.))
// 	model.net.FC3.Weight.SetData(torch.Add(model.net.FC3.Weight, otherModel.net.FC3.Weight, 1.))
// 	model.net.FC1.Bias.SetData(torch.Add(model.net.FC1.Bias, otherModel.net.FC1.Bias, 1.))
// 	model.net.FC2.Bias.SetData(torch.Add(model.net.FC2.Bias, otherModel.net.FC2.Bias, 1.))
// 	model.net.FC3.Bias.SetData(torch.Add(model.net.FC3.Bias, otherModel.net.FC3.Bias, 1.))
// }

// func (model *SimpleNN) divideModelWeights(n float32) {
// 	model.net.FC1.Weight.SetData(torch.Div(model.net.FC1.Weight, otherModel.net.FC1.Weight, 1.))
// 	model.net.FC2.Weight.SetData(torch.Add(model.net.FC2.Weight, otherModel.net.FC2.Weight, 1.))
// 	model.net.FC3.Weight.SetData(torch.Add(model.net.FC3.Weight, otherModel.net.FC3.Weight, 1.))
// 	model.net.FC1.Bias.SetData(torch.Add(model.net.FC1.Bias, otherModel.net.FC1.Bias, 1.))
// 	model.net.FC2.Bias.SetData(torch.Add(model.net.FC2.Bias, otherModel.net.FC2.Bias, 1.))
// 	model.net.FC3.Bias.SetData(torch.Add(model.net.FC3.Bias, otherModel.net.FC3.Bias, 1.))
// }
