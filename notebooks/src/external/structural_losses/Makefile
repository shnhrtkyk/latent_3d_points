nvcc := /apps/t3/sles12sp2/cuda/9.0.176/bin/nvcc
cudalib := /apps/t3/sles12sp2/cuda/9.0.176/lib64
cudainclude :=/apps/t3/sles12sp2/cuda/9.0.176/include
tensorflow := /apps/t3/sles12sp2/free/tensorflow/1.9.0/gnu/lib/python3.4/site-packages/tensorflow/include

TF_CFLAGS := $$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') 
TF_LFLAGS := $$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') 


all: tf_nndistance_so.so
clean:
	rm -rf *.o *.so
.PHONY : all clean

tf_nndistance_so.so: tf_nndistance_g.o tf_nndistance.cpp
	g++ -std=c++11 -shared tf_nndistance.cpp tf_nndistance_g.o -o tf_nndistance_so.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2

tf_nndistance_g.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

