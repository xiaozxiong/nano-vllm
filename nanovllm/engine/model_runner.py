import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        # select GPU the current process use
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()  #
        torch.set_default_dtype(hf_config.torch_dtype)  # data type of model

        print(
            f"--- default_dtype: {default_dtype}, hf_config.torch_dtype: {hf_config.torch_dtype}"
        )

        # From now, create tensors on the GPU by default
        torch.set_default_device("cuda")
        # * model construction + weight loading
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)

        self.sampler = Sampler()
        # * GPU warm-up pass for the LLM engine
        self.warmup_model()

        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        #! shm is used for communication
        if self.world_size > 1:
            if rank == 0: # controller: scheduler, HTTP server
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier() # sync
            else: # workers
                dist.barrier() # sync, weit for creation of shm
                #* Attaches to existing shared memory
                self.shm = SharedMemory(name="nanovllm")  # = shm in rank0
                # They never leave this function until the application shuts down.
                self.loop() # read content from shm

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        # Wait until rank-0 tells me what function to run, then run it.
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        # This function is only used by worker ranks
        assert self.world_size > 1 and self.rank > 0
        # synchronization primitive, sleep until rank0 call event.set()
        self.event.wait()
        # read message length
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # read method and arguments
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        # reset flag
        self.event.clear()
        # return the command
        return method_name, args
    #!
    def write_shm(self, method_name, *args):
        # Only rank 0 writes commands
        assert self.world_size > 1 and self.rank == 0
        # 
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        # Wake all workers
        for event in self.event:
            event.set()
    
    #! call the specific method across multi-processes
    def call(self, method_name, *args):
        #* broadcast with Shared Memory (SHM)
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        
        #* local execution
        # find a method with specific name
        # method = self.method_name
        method = getattr(self, method_name, None)
        return method(*args) # invoke this method

    def warmup_model(self):
        print("------ warm up model -------")
        torch.cuda.empty_cache()  # Releases cached GPU memory
        torch.cuda.reset_peak_memory_stats()  # Resets max-memory tracking

        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]

        print(
            f"--- max_num_batched_tokens = {max_num_batched_tokens}, max_model_len = {max_model_len}, num_seqs = {num_seqs}"
        )

        self.run(seqs, True)
        print("------ Finish warming up ------")
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config

        free, total = torch.cuda.mem_get_info()
        # Real GPU usage
        used = total - free
        # The maximum PyTorch tensor memory ever allocated
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # Live PyTorch tensors
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # world_size is the number of tensor-parallel GPUs
        # Heads are split across GPUs, KV cache is stored per rank
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        print(f"block_size = {self.block_size}")
        #* this is the size in bytes of a block
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size # tokens per block
            * num_kv_heads
            * head_dim
            * hf_config.torch_dtype.itemsize
        )
        # how many blocks fit in the GPU?
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0
        print(f"--- num_blocks = {config.num_kvcache_blocks}")
        # memory pool
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        #* binding
        # assign kv cache
        layer_id = 0
        # Iterate over all modules
        for module in self.model.modules():
            # print(f"--- module: {module}")
            # print(f"--- module: {dir(module)}")
            # only select attention layer
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        Expand block tables to max len
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = [] # tokens feed into the model
        positions = [] # positions within each seq for elements in input_ids
        # prefix-sum offsets
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]

        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = [] # address book: logical position -> physical position, tell attention where KV blocks live in memory
        block_tables = None
        # for each sequence
        for seq in seqs:
            seqlen = len(seq)
            # Seq: [ prefix cached | new tokens ]
            #* only care about new tokens, prefix cache
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup
                continue
            # build slot mapping
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                # Calculate physical start address for this block
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                # Add these physical addresses to the mapping, locally continuous
                slot_mapping.extend(list(range(start, end)))
        
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  #* prefix cache exists
            block_tables = self.prepare_block_tables(seqs)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        #* metadata for attention computation, global variation
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        #* without cuda graph
        # e.g., model = Qwen3ForCausalLM
        # self.model() -> Qwen3ForCausalLM.forward
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        #* with cuda graph
        else:
            bs = input_ids.size(0)
            context = get_context() # obtain metadata set in preparation
            # Picks the smallest graph that can fit bs
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            
            #* Executes the captured CUDA graph
            graph.replay()
            #! compute
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    #* runs one model step
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        print(f"--- run in rank{self.rank}")
        #* Prepare inputs
        # prefill: input_ids: [batch_size, seq_len], positions: [batch_size, seq_len]
        #          length = sum(len(seq)), length = sum(len(seq))
        # decode: input_ids: [batch_size, 1], positions: [batch_size, 1]
        # shape = (bs, ), shape = (bs, )
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        print(f"--- input_ids shape = {input_ids.shape}, positions shape = {positions.shape}")
        # only rank0, get temperature list for seqs
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        
        # all ranks execute this
        #? perform all-gather to get the final result
        #? This ensures that the final probability distribution (logits) for the next token is assembled 
        #? from the shards and available (at least) to Rank 0.
        # logits: (batch_size, the vocabulary size) e.g., (7, 151936)
        logits = self.run_model(input_ids, positions, is_prefill)
        
        # sample next token, only rank0
        #* Select the token with highest score in the vocabulary
        print(f"--- finish computing logits, select token_id...")
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )

        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
