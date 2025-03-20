# LiteRT Inference with Webcam

## Usage

Tested on Raspberry Pi 5 with USB Webcam.

After cloning,

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements-lite.txt`
4. Copy in your `.tflite` model from [Prelab](https://usafa-ece.github.io/ece386-book/b3-devboard/lab-cat-dog.html#pre-lab)
5. `python litert_continuous.py cat-dog-mnv2.tflite`

Verify your signatures are what you expect, then get to work!

## Discussion Questions

The size of the Keras model is 26.3MB while the size of the LiteRT model is 2.4MB. The Raspberry Pi 5 has four cores with a 512kB per-core L2 cache and a 2MB shared L3 cache. Therefore, the LiteRT model at 2.4MB can barely fit into the L3 cache and one core's L2 cache with 112kB left over. This makes the access time much faster than accessing the SDRAM.
This can be seen quantitatively through the differences seen in the perf benchmark of both models. The execution time of each model is one of the most direct ways to measure performance; the Keras model took over eight times longer to run than the LiteRT model. This increased execution time is the direct result of increased cache use by the Keras model compared to the LiteRT model. Since the Keras model is much larger, it required over twenty-six times more cache references with twenty-six times more L1 references, forty times more L2 references and thirty-six times more L3 references. All of these cache references moved data in and out of the cache constantly, which resulted in the Keras model having thirty times more cache misses. This includes a penalty for the processor to fetch the desired information and bring it into the cache before finally using it. Additionally, the Keras model is much larger than these caches, so most of the model needed to be stored in SDRAM, which resulted in twenty-seven times more memory accesses compared to the LiteRT model. These cache miss and memory access times result in stalled cycles where the CPU is not doing any work but simply waiting for data to arrive from memory. The Keras model required almost fourteen times more stalled cycles which contributed greatly to its increased execution time. 
Since the L2 cache is very small, its data is constantly overwritten when moving data from the L3 cache. This is particularly emphasized in the Keras model which overwrote data in the L2 cache seventy times more than the LiteRT model. This could suggest that thrashing is occuring in which the same data is being moved back and forth in the L2 cache. 

## Documentation

I used https://www.raspberrypi.com/products/raspberry-pi-5/ for information about the Raspberry pi 5's L2 and L3 caches.

### People

None.

### LLMs

I used ChatGPT to help me use black with tensorflow functions. I used the #fmt on / off comments for black to ignore tensorflow functions that black struggles to format.
https://chatgpt.com/share/67dc2e4c-ad34-8000-971c-50a0a0abdb0d
