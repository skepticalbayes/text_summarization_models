# text_summarization_models
Abstractive and extractive summarization implemented using tensorflow

# Abstractive Summarization
<h3>Abstractive summarization using encoder and transformer decoder</h3>

I have used a text generation library called Texar , Its a beautiful library with a lot of abstractions, i would say it to be 
scikit learn for text generation problems.

The main idea behind this architecture is to use the transfer learning from pretrained transformer encoder a masked language model ,
I have replaced the Encoder part with Encoder and the deocder is trained from the scratch.

One of the advantages of using Transfomer Networks is training is much faster than LSTM based models as we elimanate sequential behaviour in Transformer models.

Transformer based models generate more gramatically correct  and coherent sentences.

<h1> Code </h1>

<pre>


<h2>download the texar code and install all the python packages specified 
    in requirement.txt of texar_repo</h2>
import sys
!test -d texar_repo || git clone https://github.com/asyml/texar.git texar_repo
if not 'texar_repo' in sys.path:
  sys.path += ['texar_repo']

</h2>download the CNN Stories data set and unzip the file</h2>
https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ
tar -zxf cnn_stories.tgz
