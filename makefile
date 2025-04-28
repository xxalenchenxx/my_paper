# 編譯器與旗標
CC = nvcc
CFLAGS = -O3 -Wno-deprecated-gpu-targets

# 目標檔與路徑
TARGET = a
LIB_DIRS := ../Lib_cpps/
CPP_LIBS := vVector qQueue tTime FileReader AdjList CSR D1Process
CU_LIBS := AP_Process

# 生成路徑
CPP_OBJS := $(addsuffix .o, $(CPP_LIBS))
CU_OBJS := $(addsuffix .o, $(CU_LIBS))

# 預設目標
all: $(TARGET)

# 最終執行檔
$(TARGET): $(CPP_OBJS) $(CU_OBJS)
	$(CC) $(CFLAGS) BC.cu $^ -o $@

# 編譯 .cpp 為 .o
$(CPP_OBJS): %.o: $(LIB_DIRS)% 
	$(CC) $(CFLAGS) -c $</$*.cpp -o $@

# 編譯 .cu 為 .o
$(CU_OBJS): %.o: $(LIB_DIRS)% 
	$(CC) $(CFLAGS) -c $</$*.cu -o $@

# 清除中間檔與可執行檔
clean:
	rm -f *.o $(TARGET)
