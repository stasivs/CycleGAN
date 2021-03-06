# CycleGAN

This project is an implementation of CycleGAN architecture to 
train generative models

## Architecture

+ Layers.py - File with layers. The layer is Convolutional + Normalization + Activation function.
+ Blocks.py - File with blocks. Blocks consists of multiple layers.
+ Generators.py - File with Generators.
+ Discriminators.py - File with Discriminators.
+ CycleGAN.py - File with CycleGAN architecture to train Generators and Discriminators.
+ DataPreprocessing.py - File to prepare your data for training.

## Usage

You can use project implementation of Generators and Discriminators or write your own.
There is an example how to train model:

```python
''' 
For example you have a Generator class and Discriminator class.

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        ...
	
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        ...
	
To train them you need to pass them like this:
'''

cg = CycleGAN(gpu_mode=True, const_photo=None, generator=Generator, discriminator=Discriminator)
netG, losses, image_hist = cg.fit(data1, data2, epochs=50) # Where data2 is target and data1 is what we want to interpolate to data2
```
