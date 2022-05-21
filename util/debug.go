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

var Logger *log.Logger = log.Default()

func InitLogger(nodeId int) {
	file, _ := os.Create("log.txt")
	mw := io.MultiWriter(file)
	prefix := fmt.Sprintf("LOG %d: ", nodeId)
	Logger = log.New(mw, prefix, 0)
}
