# Study-by-JP

# How to Calculate GPU Requirement 
in full precision (float32), every parameter of the model is stored in 32 bits or 4 bytes. Hence 4 bytes / parameter * 7 billion parameters = 28 billion bytes = 28 GB of GPU memory required, for inference only. In half precision, each parameter would be stored in 16 bits, or 2 bytes. Hence you would need 14 GB for inference. There are now also 8 bit and 4 bit algorithms, so with 4 bits (or half a byte) per parameter you would need 3.5 GB of memory for inference.

For training, it depends on the optimizer you use.

In case you use regular AdamW, then you need 8 bytes per parameter (as it not only stores the parameters, but also their gradients and second order gradients). Hence, for a 7B model you would need 8 bytes per parameter * 7 billion parameters = 56 GB of GPU memory. If you use AdaFactor, then you need 4 bytes per parameter, or 28 GB of GPU memory. With the optimizers of bitsandbytes (like 8 bit AdamW), you would need 2 bytes per parameter, or 14 GB of GPU memory.

I highly recommend this guide: Efficient Training on a Single GPU 1.9k which goes over all of this in much more detail.
