package util

import (
	"fmt"
	"os"
)

var debug bool = true

func Debug[T any](s T) {
	if debug {
		fmt.Println(s)
	}
}

var f, err = os.Create("log.txt")

func Log(s string) {

	d1 := []byte(s)
	f.Write(d1)
}
