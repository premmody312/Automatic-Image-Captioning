# Automatic-Image-Captioning
Combine CNN and RNN knowledge to build a network that automatically produces captions, given an input image.


## Dependencies:

nltk pytorch os

## Installing dependencies:

> pip install nltk | pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html

## Screenshots:
![alt text](https://github.com/premmody312/Automatic-Image-Captioning/blob/master/images/Capture1.PNG)
___
![alt text](https://github.com/premmody312/Automatic-Image-Captioning/blob/master/images/Capture2.PNG)
___


## Algorithm:

### Loading and Visualizing the dataset

The Microsoft Common Objects in COntext (MS COCO) dataset is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms. You can read more about the dataset on the website or the research paper.


### Image and Caption Preprocessing

#### Image Pre-Processing
Image pre-processing is relatively straightforward (from the __getitem__ method in the CoCoDataset class):

###### Convert image to tensor and pre-process using transform
```python
image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
image = self.transform(image)
```
After loading the image in the training folder with name path, the image is pre-processed using the same transform (transform_train) that was supplied when instantiating the data loader.


#### Caption Pre-Processing

The captions also need to be pre-processed and prepped for training. In this example, for generating captions, we are aiming to create a model that predicts the next token of a sentence from previous tokens, so we turn the caption associated with any image into a list of tokenized words, before casting it to a PyTorch tensor that we can use to train the network.

To understand in more detail how COCO captions are pre-processed, we'll first need to take a look at the vocab instance variable of the CoCoDataset class. The code snippet below is pulled from the __init__ method of the CoCoDataset class:

```python
def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
       
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
```


### Defining the NN-RNN  Architecture:

The encoder is basically a ResNet with the fully connected layers removed and the decoder is a LSTM (Long short term memory). I have referred to the paper of "Show and Tell: A Neural Image Captioning Generator" . AS using softmax activation function leads to deterioration in efficiency.Other attributes are generally as mentioned in the paper.Xavier Initialization ,hidden_size=512, embed_size=512. For the batch size i used hit and trial method , i tired 25,30,50 and 64.The best result was given by 64.(I just observed from epoch 1 and dint execute each batch size)
We have used Adam optimizer as it is memory and computationally efficient.


### Inference

As the last task in this project, we will loop over the images until you find two image-caption pairs of interest:
Two should include image-caption pairs that show instances when the model performed well.



## Conclusion:
Thus we gain knowledge in defining and using the CNN  and RNN Architecture by referring to many research papers and were able to apply this knowledge to generate captions for various images
