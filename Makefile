all:
	nvcc -I. main.cu layer.cu -o CNN.exe -lcuda -lcudart -lcublas -Wno-deprecated-gpu-targets
run:
	./CNN.exe
clean:
	rm CNN
	
