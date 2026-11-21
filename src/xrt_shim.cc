#include <cstring>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

namespace {

xrt::device &dev() {
  static auto *d = new xrt::device(0);
  return *d;
}

template <typename Fn> auto xrt_try(Fn &&fn) noexcept -> decltype(fn()) {
  try {
    return fn();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "XRT Error: %s\n", e.what());
    std::abort();
  }
}
}

struct Program {
  xrt::hw_context hw_context;
  xrt::kernel kernel;
  xrt::run run;
  int instruction_group_id;
  int data_group_id;
};

struct Instructions {
  xrt::bo buffer;
  uint32_t ninstr;
};

struct Buffer {
  xrt::bo buffer;
  void *ptr;
  size_t bytes;
};

extern "C" {

Program *npu_program_load(const uint8_t *data, size_t len) {
  return xrt_try([&] {
    xrt::xclbin xclbin{
        std::string_view(reinterpret_cast<const char *>(data), len)};
    auto uuid = dev().register_xclbin(xclbin);
    xrt::hw_context hw_context(dev(), uuid);
    xrt::kernel kernel(hw_context, "MLIR_AIE");
    xrt::run run(kernel);
    run.set_arg(0, uint64_t(3));
    int instruction_group_id = kernel.group_id(1);
    int data_group_id = kernel.group_id(3);
    return new Program{std::move(hw_context), std::move(kernel), std::move(run),
                       instruction_group_id, data_group_id};
  });
}

void npu_program_free(Program *p) { delete p; }

Instructions *npu_instructions_load(const Program *p, const uint8_t *data,
                                    size_t len) {
  return xrt_try([&] {
    xrt::bo buffer(dev(), len, xrt::bo::flags::cacheable,
                   p->instruction_group_id);
    std::memcpy(buffer.map<void *>(), data, len);
    buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return new Instructions{std::move(buffer), static_cast<uint32_t>(len / 4)};
  });
}

void npu_instructions_free(Instructions *i) { delete i; }

Buffer *npu_buffer_alloc(const Program *p, size_t bytes) {
  return xrt_try([&] {
    size_t aligned = (bytes + 4095) & ~size_t(4095);
    if (aligned == 0)
      aligned = 4096;
    xrt::bo buffer(dev(), aligned, xrt::bo::flags::host_only, p->data_group_id);
    void *ptr = buffer.map<void *>();
    return new Buffer{std::move(buffer), ptr, aligned};
  });
}

Buffer *npu_buffer_sub(const Buffer *parent, size_t offset, size_t bytes) {
  return xrt_try([&] {
    xrt::bo sub(parent->buffer, bytes, offset);
    void *ptr = static_cast<uint8_t *>(parent->ptr) + offset;
    return new Buffer{std::move(sub), ptr, bytes};
  });
}

void *npu_buffer_ptr(Buffer *b) { return b->ptr; }
size_t npu_buffer_size(const Buffer *b) { return b->bytes; }
void npu_buffer_free(Buffer *b) { delete b; }

void npu_buffer_sync(Buffer *b, int to_device, size_t offset, size_t bytes) {
  xrt_try([&] {
    auto dir =
        to_device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE;
    if (offset == 0 && bytes == 0)
      b->buffer.sync(dir);
    else
      b->buffer.sync(dir, bytes, offset);
  });
}

void npu_program_bind(Program *p, const Instructions *insts,
                      Buffer *const *buffers, size_t n) {
  xrt_try([&] {
    p->run.set_arg(1, insts->buffer);
    p->run.set_arg(2, insts->ninstr);
    for (size_t i = 0; i < n; ++i) {
      p->run.set_arg(3 + static_cast<int>(i), buffers[i]->buffer);
    }
  });
}

void npu_program_set_buffer(Program *p, size_t slot, Buffer *buffer) {
  xrt_try([&] { p->run.set_arg(3 + static_cast<int>(slot), buffer->buffer); });
}

void npu_program_start(Program *p) {
  xrt_try([&] { p->run.start(); });
}

void npu_program_wait(Program *p) {
  xrt_try([&] {
    auto state = p->run.wait(std::chrono::milliseconds(1000));
    if (state != ERT_CMD_STATE_COMPLETED) {
      std::fprintf(stderr, "XRT Timeout: state=%d\n", static_cast<int>(state));
      std::abort();
    }
  });
}
}