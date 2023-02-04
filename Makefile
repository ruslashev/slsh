NAME = slsh
MODE = debug

TARGET_DIR = $(shell pwd)/target
COMP_SHADERS_DIR = $(TARGET_DIR)/shaders
BIN = $(TARGET_DIR)/$(MODE)/$(NAME)
DEP = $(BIN).d

SHADERS = $(wildcard shaders/*)
BUILT_SHADERS = $(SHADERS:shaders/%=$(COMP_SHADERS_DIR)/%.spv)

ifeq ($(MODE), release)
    CFLAGS = --release
else ifneq ($(MODE), debug)
    $(error Unknown build mode "$(MODE)")
endif

GLSLC_FLAGS = -O

run: $(BIN)
	@$(BIN)

$(BIN): $(BUILT_SHADERS)
	cargo build $(CFLAGS)

$(COMP_SHADERS_DIR)/%.spv: shaders/%
	@mkdir -p $(@D)
	glslc $(GLSLC_FLAGS) $^ -o $@

clean:
	cargo clean

.NOTPARALLEL:
.PHONY: run clean

-include $(DEP)
