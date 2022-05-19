package main

import (
	"final_project/ml"
	"flag"
	"fmt"
	"log"
	"os"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

var device torch.Device

func main() {
	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		device = torch.NewDevice("cuda")
	} else {
		log.Println("No CUDA found; CPU only")
		device = torch.NewDevice("cpu")
	}

	initializer.ManualSeed(1)

	trainCmd := flag.NewFlagSet("train", flag.ExitOnError)
	trainTar := trainCmd.String("data", "./data/mnist_png/mnist_png_training_shuffled.tar.gz", "data tarball")
	testTar := trainCmd.String("test", "./data/mnist_png/mnist_png_testing_shuffled.tar.gz", "data tarball")
	save := trainCmd.String("save", "/tmp/mnist_model.gob", "the model file")

	predictCmd := flag.NewFlagSet("predict", flag.ExitOnError)
	// load := predictCmd.String("load", "/tmp/mnist_model.gob", "the model file")

	lr := trainCmd.Float64("lr", .01, "learning rate")
	epochs := trainCmd.Int("epochs", 10, "number of epochs")

	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s needs subcomamnd train or predict\n", os.Args[0])
		os.Exit(1)
	}

	switch os.Args[1] {
	case "train":
		trainCmd.Parse(os.Args[2:])
		model := ml.MakeSimpleNN(*lr, *epochs, device)
		model.Train(*trainTar, *testTar, *save)
	case "predict":
		predictCmd.Parse(os.Args[2:])
		// predict(*load, predictCmd.Args())
	}
}
