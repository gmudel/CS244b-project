package util

import (
	"fmt"
	"io"
	"log"
	"os"
)

var debug bool = true

func Debug[T any](s T) {
	if debug {
		fmt.Println(s)
	}
}

var file, err = os.Create("log.txt")
var mw = io.MultiWriter(file)
var Logger = log.New(mw, "LOG: ", 0)
