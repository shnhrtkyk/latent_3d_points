set -e
if [ 'tf_approxmatch_g.cu.o' -ot 'tf_approxmatch_g.cu' ] ; then
	echo 'nvcc'
	/apps/t3/sles12sp2/cuda/9.0.176/bin/nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
fi
if [ 'tf_approxmatch_so.so'  -ot 'tf_approxmatch.cpp' ] || [ 'tf_approxmatch_so.so'  -ot 'tf_approxmatch_g.cu.o' ] ; then
	echo 'g++'
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I /apps/t3/sles12sp2/free/tensorflow/1.9.0/gnu/lib/python3.4/site-packages/tensorflow/include -I /apps/t3/sles12sp2/cuda/9.0.176/include  -L /apps/t3/sles12sp2/cuda/9.0.176/lib64/ -O2
fi

