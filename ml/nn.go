package federated

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type ToyNN struct {
}

type NN struct {
	lr float64

	input_dim  int
	hidden_dim int
	output_dim int

	w1 *mat.Dense
	w2 *mat.Dense
}

// https://github.com/aleksficek/Go-Neural-Network/blob/master/neural_net.go
// https://pkg.go.dev/github.com/orktes/go-torch#readme-installing

func MakeGoNetwork(input_dim, hidden_dim, output_dim int, lr float64) *NN {

	nn := NN{
		lr:         lr,
		input_dim:  input_dim,
		hidden_dim: hidden_dim,
		output_dim: output_dim,
		w1:         mat.NewDense(hidden_dim, input_dim, createRandomArray(hidden_dim*input_dim)),
		w2:         mat.NewDense(output_dim, hidden_dim, createRandomArray(output_dim*hidden_dim)),
	}
	fmt.Println("Creating neural net structure: ", nn)
	return &nn
}
