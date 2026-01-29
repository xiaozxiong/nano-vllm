import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def worker(rank, world_size, batch, in_dim, out_dim):
    # 1.bind this process to one GPU
    torch.cuda.set_device(rank)

    #
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank,
    )

    # create local shard
    out_per_rank = out_dim // world_size
    W_shard = torch.empty(out_per_rank, in_dim, device="cuda")

    # * nccl only accepts CUDA tensor on both send and receive sides
    if rank == 0:
        W_full_cpu = torch.randn(out_dim, in_dim, pin_memory=True)
        W_shard.copy_(W_full_cpu[:out_per_rank])

        for r in range(1, world_size):
            shard = W_full_cpu[r * out_per_rank : (r + 1) * out_per_rank].cuda(
                non_blocking=True
            )
            dist.send(shard, dst=r)
    else:
        dist.recv(W_shard, src=0)
    
    # data X
    if rank == 0:
        X = torch.randn(batch, in_dim, device="cuda")
    else:
        X = torch.empty(batch, in_dim, device="cuda")
    dist.broadcast(X, src=0)

    # start_comp = torch.cuda.Event(True)
    # end_comp = torch.cuda.Event(True)

    # start_comp.record()
    Y_partial = X @ W_shard.t()
    # end_comp.record()

    # gather
    if rank == 0:
        Y_parts = [
            torch.empty(batch, out_per_rank, device="cuda") for _ in range(world_size)
        ]
    else:
        Y_parts = None
    
    # start_comm = torch.cuda.Event(True)
    # end_comm = torch.cuda.Event(True)
    # start_comm.record()
    dist.gather(Y_partial, Y_parts, dst=0)
    # end_comm.record()

    # torch.cuda.synchronize()

    # comp_ms = start_comp.elapsed_time(end_comp)
    # comm_ms = start_comm.elapsed_time(end_comm)
    
    if rank == 0:
        Y = torch.cat(Y_parts, dim=1)
        print("Final output shape:", Y.shape)

    dist.destroy_process_group()

def main():
    world_size = 2

    batch = 8
    in_dim = 2048
    out_dim = 65536  # must be divisible by world_size

    mp.spawn(
        worker,
        args=(world_size, batch, in_dim, out_dim),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
