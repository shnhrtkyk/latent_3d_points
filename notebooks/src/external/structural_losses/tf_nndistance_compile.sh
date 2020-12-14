

/apps/t3/sles12sp2/cuda/8.0.61/bin/nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I /apps/t3/sles12sp2/free/tensorflow/1.9.0/gnu/lib/python3.4/site-packages/tensorflow/include -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 && g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I /apps/t3/sles12sp2/free/tensorflow/1.9.0/gnu/lib/python3.4/site-packages/tensorflow/include -L /apps/t3/sles12sp2/cuda/8.0.61/lib64 -O2


# /apps/t3/sles12sp2/cuda/9.0.176/bin/nvcc  tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I /apps/t3/sles12sp2/free/tensorflow/1.9.0/gnu/lib/python3.4/site-packages/tensorflow/include -I /apps/t3/sles12sp2/cuda/9.0.176/include -I /apps/t3/sles12sp2/free/tensorflow/1.9.0/gnu/lib/python3.4/site-packages/tensorflow/include/include/external/nsync/public -lcudart -L /apps/t3/sles12sp2/cuda/9.0.176/lib64/ -L/apps/t3/sles12sp2/free/tensorflow/1.9.0/gnu/lib/python3.4/site-packages/tensorflow/include -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0



# /apps/t3/sles12sp2/cuda/9.0.176/bin/nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I$TF_INC -I/apps/t3/sles12sp2/cuda/9.0.176/include -lcudart -L/apps/t3/sles12sp2/cuda/9.0.176/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework
