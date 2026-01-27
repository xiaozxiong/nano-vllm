import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # gather config field
        config_fields = {field.name for field in fields(Config)}
        # collect eligible cofing
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # config class for model
        config = Config(model, **config_kwargs)
        print("Used config: ", config_kwargs)
        #* tensor parallelism
        self.ps = [] # keep track of working process
        self.events = [] # 
        ctx = mp.get_context("spawn") # multiprocessing context
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # create a worker process, child process, (rank > 0)
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            # process executes
            process.start()
            self.ps.append(process)
            self.events.append(event)
        #* rank0
        self.model_runner = ModelRunner(config, 0, self.events) # rank0

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    #* convert text prompt into tokens
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        print(f"--- prompt: {prompt}")
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        print(f"--- token_ids of prompt = {prompt}\n")
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
    #!
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        print(f"--- the number of scheduled seqs: {len(seqs)}")
        print(f"--- schedulted seq id = {[seq.seq_id for seq in seqs]}")
        #! run
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        self.scheduler.postprocess(seqs, token_ids)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        #* add request into waiting queue of scheduler
        print(f"--- length of sampling_params: {len(sampling_params)}")
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        outputs = {} # {1: [], 2: []}
        prefill_throughput = decode_throughput = 0.

        #* ask scheduler if both waiting queue and running queue are empty
        step_count = 0
        print("--------------------------------- start generation -------------------------")
        while not self.is_finished():
            t = perf_counter()
            #! generate one token for each scheduled seq
            #! the number of scheduled seqs depends on memory capacity
            print(f"==== step#{step_count} ====")
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                # store output token to corresponding sequence
                print(f"--- number of tokens of seq#{seq_id} per step: {len(token_ids)}")
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

            step_count += 1
        
        print(f"--- output: length = {len(outputs)}, {outputs}\n")
        # sort sequences (request) by their id, a sequence is a request
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # convert token ids back into text
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
