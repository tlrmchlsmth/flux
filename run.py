import multiprocessing as mp
import torch

import flux

def test(tp_group, m, n, k, transpose_weight, local_copy, dtype):
    world_size = torch.distributed.get_world_size(tp_group)

    a = (-2 * torch.rand(m, k, dtype=dtype, device="cuda") + 1) / .1
    b = ((-2 * torch.rand(k, n, dtype=dtype, device="cuda") + 1) / .1)

    # Massage b depending on transpose_weight
    b_ag_gemm = b
    if not transpose_weight:
        b_ag_gemm = b.t().contiguous()

    # Run fused AllGather Gemm
    ag_gemm_op = flux.AGKernel(tp_group,
                               1,
                               8192,
                               n,
                               k,
                               dtype,
                               dtype,
                               transpose_weight=transpose_weight,
                               local_copy=local_copy)

    torch.distributed.barrier()
    ag_gemm_output = ag_gemm_op.forward(a, b_ag_gemm)

    # Run a torch AllGather followed by a GEMM

    a_gathered = torch.zeros(m * world_size, k, dtype=dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(a_gathered, a, 0)
    torch_output = torch.mm(a_gathered, b)
    torch.distributed.barrier()

    if not torch.allclose(torch_output, ag_gemm_output, atol=1e-1, rtol=1e-1):
        difference = (torch_output - ag_gemm_output).to(dtype=torch.float32)
        print(f"""Error:  Max diff.
        Process: {torch.distributed.get_rank(tp_group)}.
        Arguments: {m}, {n}, {k}, {local_copy}, {transpose_weight}
        Torch output norm: {torch.norm(torch_output.to(dtype=torch.float32))}.
        AGGemm output norm: {torch.norm(ag_gemm_output.to(dtype=torch.float32))},
        Norm of difference: {torch.norm(difference)}""")


@torch.no_grad()
def initialize_process(rank, world_size):
    # Assign GPU to this process
    torch.cuda.set_device(rank)

    # Create a torch communicator
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:12345',
        world_size=world_size,
        rank=rank
    )
    tp_group = torch.distributed.new_group(ranks=list(range(world_size)), backend="nccl")

    # Initialize pynvshmem using the torch communicator
    flux.init_flux_shm(tp_group)
    torch.cuda.synchronize()

    # These are OK
    test(tp_group, 16, 4096, 4096, False, False, torch.float16);
    test(tp_group, 16, 6144, 4096, False, False, torch.float16);

    # This produces a wrong answer
    test(tp_group, 16, 3072, 4096, False, False, torch.float16);

    # Clean up
    torch.distributed.destroy_process_group()

def main():
    torch.set_printoptions(precision=4)
    torch.set_printoptions(sci_mode=True)

    world_size = 2  # Number of processes to create
    processes = []

    for rank in range(world_size):
        p = mp.Process(target=initialize_process, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
