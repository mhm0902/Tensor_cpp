TARGET = demo
CC = nvcc
CPP = g++

INCS = 
LIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand -lcudnn -lopencv_core \
-lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
-lnppicc -lnppc -lnppidei -lnppist -lnppitc -lnppig -L/usr/local/lib/ -L/home/jack/cuda/lib64/
SUBDIRS =

DEFINC = -I/home/jack/cuda/include/ -I/usr/local/cuda/include/ -I/usr/local/include/opencv4/ -I./common -I./layer -I./yolo -I/usr/include

CSRCS = $(wildcard ./common/*.cu ) 
CSRCS += $(wildcard ./layer/*.cu ) 
CSRCS += $(wildcard ./yolo/*.cu ) 

CPPSRCS = $(wildcard ./*.cpp ) 
CPPSRCS += $(wildcard ./common/*.cpp ) 
CPPSRCS += $(wildcard ./layer/*.cpp ) 
CPPSRCS += $(wildcard ./yolo/*.cpp ) 


COBJS := $(CSRCS:.cu=.o)
CPPOBJS := $(CPPSRCS:.cpp=.o)

MAKEDEPEND = gcc -MM -MT

CFLAGS =-DUSE_TENSORRT -DUSE_TENSORRT_INT8 -DUSE_NPP -O3 -Xcompiler -fPIC \
-gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 \
-gencode arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_62,code=sm_62 \
-gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_75,code=sm_75

CPPFLAGS = -Wall -O0 -g -fpermissive -fPIC -std=c++11
CDEF = $(CSRCS:.cu=.d)
CPPDEF = $(CPPSRCS:.cpp=.d)
 
all:$(TARGET)

$(CDEF) : %.d : %.cu
	$(MAKEDEPEND) $(<:.cu=.o) $< $(DEFINC) > $@
$(CPPDEF) : %.d : %.cpp
	$(MAKEDEPEND) $(<:.cpp=.o) $< $(DEFINC) > $@

depend:
	-rm $(CDEF)
	-rm $(CPPDEF)
	$(MAKE) $(CDEF)
	$(MAKE) $(CPPDEF)

$(COBJS) : %.o: %.cu
	$(CC) -c $< -o $@ $(INCS) $(DEFINC) $(CFLAGS)
$(CPPOBJS) : %.o: %.cpp
	$(CPP) -c $< -o $@ $(INCS) $(DEFINC) $(CPPFLAGS)

$(TARGET) : $(COBJS) $(CPPOBJS)
	$(CPP) -fPIC -o $(TARGET) $(COBJS) $(CPPOBJS) $(LIBS)
	#$(CPP) -shared -fPIC -o $(TARGET) $(COBJS) $(CPPOBJS) $(LIBS)
	#-rm $(COBJS) $(CPPOBJS)

.PHONY:clean cleanall

clean:			
	-rm $(TARGET) $(COBJS) $(CPPOBJS)
#rm *.d
cleanall:
	-rm $(TARGET) $(COBJS) $(CPPOBJS)