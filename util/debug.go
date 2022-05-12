package util

import "fmt"

var debug bool = true

func Debug[T any](s T) {
	if debug {
		fmt.Println(s)
	}
}
