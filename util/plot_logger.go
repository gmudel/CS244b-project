package util

import (
	"fmt"
	"io"
	"log"
	"os"
)

var PlotLogger *log.Logger = log.Default()

func InitPlotLogger(nodeId int, tag string) {
	fname := fmt.Sprintf("plot_logs_%d_%s.txt", nodeId, tag)
	file, _ := os.Create(fname)
	mw := io.MultiWriter(file)
	prefix := fmt.Sprintf("plot_logs_%d_%s: ", nodeId, tag)
	PlotLogger = log.New(mw, prefix, 0)
}
