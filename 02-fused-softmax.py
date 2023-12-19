import torch
import triton
import triton.language as tl


@torch.jit.script
def naive_softmax(x: torch.Tensor):
    # x.shape: [M, N]

    # x.max会同时返回value和indices
    # x_max.shape: [M,]
    x_max = x.max(dim=1)[0]  # Read MN, Write M
    # z.shape: [M, N]
    z = x - x_max[:, None]  # Read MN + M, Write MN
    numerator = torch.exp(z)  # Read MN, Write MN
    # denominator.shape: [M,]
    denominator = numerator.sum(dim=1)  # Read MN, Write M
    ret = numerator / denominator[:, None]  # Read MN+M, Write MN
    # Total Read: 5MN+2N, Write: 3MN+2M
    return ret


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 处理一行数据
    pid = tl.program_id(axis=0)

    input_block_start = pid * input_row_stride
    input_offset = input_block_start + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offset < input_block_start + n_cols

    # 将超出的元素填充最小值
    x = tl.load(input_ptr + input_offset, mask=input_mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    y = x_exp / tl.sum(x_exp, axis=0)

    output_block_start = pid * output_row_stride
    output_offset = output_block_start + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offset < output_block_start + n_cols

    tl.store(output_ptr + output_offset, y, mask=output_mask)


def softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def main_test():
    torch.manual_seed(0)
    M, N = 32, 512
    x = torch.randn((M, N), device="cuda")
    output_torch = naive_softmax(x)
    output_triton = softmax(x)
    print(
        f"The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - output_triton))}"
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["triton", "torch-native", "torch-jit"],
        line_names=["Triton", "Torch (native)", "Torch (jit)"],
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn((M, N), device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: softmax(x), quantiles=quantiles
        )
    if provider == "torch-native":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, dim=1), quantiles=quantiles
        )
    if provider == "torch-jit":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: naive_softmax(x), quantiles=quantiles
        )
    gbps = lambda ms: 2 * x.nelement() * x.nelement() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# Triton大约可以比torch(jit)快4倍左右
# Triton的实现和torch native实现相比水平相当，在数据量大时，更快
if __name__ == "__main__":
    main_test()
    benchmark.run(show_plots=True, print_data=True)
