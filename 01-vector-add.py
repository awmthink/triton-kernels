import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # program_id 应该是指 BlockIdx
    pid = tl.program_id(axis=0)
    # 当前block处理的起始位置
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # 这时候x和y是在global mem还是shard memory还是在register上是不知道的？
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # grid可以是一个Tuple[int]类型，也可能是一个Callable(metaparameters)->Tuple[int]的类型
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # 传入的x, y, output(torch.Tensor类型)会隐式的转换为其第一个元素的指针
    # 被triton.jit装饰的函数，可以通过grid来进行索引，返回一个可调用的gpu kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def main_test():
    torch.manual_seed(0)
    size = 90432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output_torch = x + y
    output_triton = add(x, y)
    print(
        f"The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - output_triton))}"
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles
        )
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    main_test()
    benchmark.run(print_data=True, show_plots=False)
