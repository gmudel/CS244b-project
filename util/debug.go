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
	fname := fmt.Sprintf("log%d.txt", nodeId)
	file, _ := os.Create(fname)
	mw := io.MultiWriter(file)
	prefix := fmt.Sprintf("LOG %d: ", nodeId)
	Logger = log.New(mw, prefix, 0)
}
