CXX = g++
CXXFLAGS = -O2 -std=c++17
TARGET = generate_graph
SRC = ./graphs/generate_graph.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Usage: make graph V=10 E=20 MINW=1 MAXW=10 OUT=graph.txt
graph: $(TARGET)
	./$(TARGET) $(V) $(E) $(MINW) $(MAXW) $(OUT)

clean:
	rm -f $(TARGET) graph.txt
