package ml

// import (
// 	"encoding/gob"
// 	"flag"
// 	"fmt"
// 	"log"
// 	"os"
// 	"path/filepath"
// 	"strings"
// 	"time"

// 	torch "github.com/wangkuiyi/gotorch"
// 	F "github.com/wangkuiyi/gotorch/nn/functional"
// 	"github.com/wangkuiyi/gotorch/nn/initializer"
// 	"github.com/wangkuiyi/gotorch/vision/imageloader"
// 	"github.com/wangkuiyi/gotorch/vision/models"
// 	"github.com/wangkuiyi/gotorch/vision/transforms"
// 	"gocv.io/x/gocv"
// )

// const LR = .01

// var device torch.Device
// var gw GradientWrapper

// // func listenForGradRequest() {

// // }

// func GetGradients() (ready bool, grads []MLPGrads) {
// 	// flush gradient buffer
// 	gw.lock.Lock()
// 	defer gw.lock.Unlock()

// 	if len(gw.gradBuffer) != 0 {
// 		gradBufferCopy := make([]MLPGrads, len(gw.gradBuffer))
// 		copy(gradBufferCopy, gw.gradBuffer)

// 		gw.gradBuffer = nil
// 		return true, gradBufferCopy
// 	} else {
// 		return false, nil
// 	}
// }

// func addGradients(net *models.MLPModule) {
// 	gw.lock.Lock()
// 	defer gw.lock.Unlock()
// 	mlpgrads := MLPGrads{net.FC1.Weight.Grad(), net.FC2.Weight.Grad(), net.FC3.Weight.Grad(), net.FC1.Bias.Grad(), net.FC2.Bias.Grad(), net.FC3.Bias.Grad()}
// 	gw.gradBuffer = append(gw.gradBuffer, mlpgrads)
// }

// func UpdateModel(net *models.MLPModule, incomingGrads []MLPGrads) {
// 	gw.lock.Lock()
// 	defer gw.lock.Unlock()

// 	for _, mlpgrad := range incomingGrads {
// 		newW1 := torch.Sub(net.FC1.Weight, mlpgrad.W1, LR)
// 		newW2 := torch.Sub(net.FC2.Weight, mlpgrad.W2, LR)
// 		newW3 := torch.Sub(net.FC3.Weight, mlpgrad.W3, LR)
// 		net.FC1.Weight.SetData(newW1)
// 		net.FC2.Weight.SetData(newW2)
// 		net.FC3.Weight.SetData(newW3)

// 		newB1 := torch.Sub(net.FC1.Bias, mlpgrad.B1, LR)
// 		newB2 := torch.Sub(net.FC2.Bias, mlpgrad.B2, LR)
// 		newB3 := torch.Sub(net.FC3.Bias, mlpgrad.B3, LR)
// 		net.FC1.Bias.SetData(newB1)
// 		net.FC2.Bias.SetData(newB2)
// 		net.FC3.Bias.SetData(newB3)
// 	}
// }

// func main() {
// 	if torch.IsCUDAAvailable() {
// 		log.Println("CUDA is valid")
// 		device = torch.NewDevice("cuda")
// 	} else {
// 		log.Println("No CUDA found; CPU only")
// 		device = torch.NewDevice("cpu")
// 	}

// 	initializer.ManualSeed(1)

// 	trainCmd := flag.NewFlagSet("train", flag.ExitOnError)
// 	trainTar := trainCmd.String("data", "./data/mnist_png/mnist_png_training_shuffled.tar.gz", "data tarball")
// 	testTar := trainCmd.String("test", "./data/mnist_png/mnist_png_testing_shuffled.tar.gz", "data tarball")
// 	save := trainCmd.String("save", "./ml/mnist_model.gob", "the model file")
// 	epoch := trainCmd.Int("epoch", 5, "the number of epochs")

// 	predictCmd := flag.NewFlagSet("predict", flag.ExitOnError)
// 	load := predictCmd.String("load", "./ml/mnist_model.gob", "the model file")

// 	if len(os.Args) < 2 {
// 		fmt.Fprintf(os.Stderr, "Usage: %s needs subcomamnd train or predict\n", os.Args[0])
// 		os.Exit(1)
// 	}

// 	switch os.Args[1] {
// 	case "train":
// 		net := models.MLP()
// 		net.To(device)
// 		opt := torch.SGD(LR, 0, 0, 0, false)
// 		opt.AddParameters(net.Parameters())

// 		trainCmd.Parse(os.Args[2:])
// 		train(net, opt, *trainTar, *testTar, *epoch, *save)
// 	case "predict":
// 		predictCmd.Parse(os.Args[2:])
// 		predict(*load, predictCmd.Args())
// 	}
// }

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

// // MNISTLoader returns a ImageLoader with MNIST training or testing tgz file
// func MNISTLoader(fn string, vocab map[string]int) *imageloader.ImageLoader {
// 	trans := transforms.Compose(transforms.ToTensor(), transforms.Normalize([]float32{0.1307}, []float32{0.3081}))
// 	loader, e := imageloader.New(fn, vocab, trans, 64, 64, time.Now().UnixNano(), torch.IsCUDAAvailable(), "gray")
// 	if e != nil {
// 		panic(e)
// 	}
// 	return loader
// }

// func test(model *models.MLPModule, loader *imageloader.ImageLoader) {
// 	testLoss := float32(0)
// 	correct := int64(0)
// 	samples := 0
// 	for loader.Scan() {
// 		data, label := loader.Minibatch()
// 		data = data.To(device, data.Dtype())
// 		label = label.To(device, label.Dtype())
// 		output := model.Forward(data)
// 		loss := F.NllLoss(output, label, torch.Tensor{}, -100, "mean")
// 		pred := output.Argmax(1)
// 		testLoss += loss.Item().(float32)
// 		correct += pred.Eq(label.View(pred.Shape()...)).Sum(map[string]interface{}{"dim": 0, "keepDim": false}).Item().(int64)
// 		samples += int(label.Shape()[0])
// 	}
// 	log.Printf("Test average loss: %.4f, Accuracy: %.2f%%\n",
// 		testLoss/float32(samples), 100.0*float32(correct)/float32(samples))
// }

// func saveModel(model *models.MLPModule, modelFn string) {
// 	log.Println("Saving model to", modelFn)
// 	f, e := os.Create(modelFn)
// 	if e != nil {
// 		log.Fatalf("Cannot create file to save model: %v", e)
// 	}
// 	defer f.Close()

// 	d := torch.NewDevice("cpu")
// 	model.To(d)
// 	if e := gob.NewEncoder(f).Encode(model.StateDict()); e != nil {
// 		log.Fatal(e)
// 	}
// }

// func predict(modelFn string, inputs []string) {
// 	net := loadModel(modelFn)

// 	for _, in := range inputs {
// 		for _, pa := range strings.Split(in, ":") {
// 			fns, e := filepath.Glob(pa)
// 			if e != nil {
// 				log.Fatal(e)
// 			}

// 			for _, fn := range fns {
// 				predictFile(fn, net)
// 			}
// 		}
// 	}
// }

// func loadModel(modelFn string) *models.MLPModule {
// 	f, e := os.Open(modelFn)
// 	if e != nil {
// 		log.Fatal(e)
// 	}
// 	defer f.Close()

// 	states := make(map[string]torch.Tensor)
// 	if e := gob.NewDecoder(f).Decode(&states); e != nil {
// 		log.Fatal(e)
// 	}

// 	net := models.MLP()
// 	net.SetStateDict(states)
// 	return net
// }

// func predictFile(fn string, m *models.MLPModule) {
// 	img := gocv.IMRead(fn, gocv.IMReadGrayScale)
// 	t := transforms.ToTensor().Run(img)
// 	n := transforms.Normalize([]float32{0.1307}, []float32{0.3081}).Run(t)
// 	fmt.Println(m.Forward(n).Argmax().Item())
// }
