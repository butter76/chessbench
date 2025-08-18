PROJECT_NAME := chess_sample
CXX ?= clang++-18
STD ?= c++23

BUILD_DIR := build
SRC_DIR := src
DEPS_DIR := third_party
CHESS_DIR := $(DEPS_DIR)/chess-library
CHESS_HEADER := $(CHESS_DIR)/chess.hpp
CHESS_URL := https://raw.githubusercontent.com/Disservin/chess-library/master/include/chess.hpp

CXXFLAGS ?= -O2 -Wall -Wextra -Wpedantic -std=$(STD) -I$(CHESS_DIR) -stdlib=libc++
LDFLAGS ?= -stdlib=libc++ -lc++abi

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
TARGET := $(BUILD_DIR)/$(PROJECT_NAME)

.PHONY: all deps clean run

all: deps $(TARGET)

deps: $(CHESS_HEADER)

$(CHESS_HEADER):
	mkdir -p $(CHESS_DIR)
	curl -fsSL $(CHESS_URL) -o $(CHESS_HEADER)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(CHESS_HEADER) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

run: all
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)


