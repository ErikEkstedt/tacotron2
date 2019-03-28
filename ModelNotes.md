# Tacotron Model Notes



## Tacotron

Layers:
- Embedding
  - Normal embedding
  - Characters
- Encoding
  - Normal BiLSTM encoding
- Decoding
  - prenet
  - attention
- Postnet
  - Five 1-d convolution with 512 channels and kernel size 5
  - BatchNorm1d
  - Residual addition to decoder output


### Encoder
Pretty standard network

- Layers
  - Three 1-d convolution banks
    - Init normalization: `torch.nn.init.calculate_gain('relu'))`
    - stride: 1
    - dilation: 1
    - kernel: hparams
    - Batcnormd1d
    - Relu
    - 50% dropout (during training)
  - **Bidirectional** LSTM
- Forward
  - Conv layers with dropout in each layer 
  - RNN
  - Dont save hidden state. Only pass output


### Decoder

The most complicated layer in the model. 

- Layers:
  - **Prenet**
    - Linear stack
    - N layers
    - Relu activation
    - 50% dropout
  - **Attention** RNN (LSTMCell)
  - **Attention** layer
  - **Decoder** RNN (LSTMCell)
  - **Linear projection**
    - Init normalization: `torch.nn.init.calculate_gain('relu'))`
  - **Gate layer**
    - Init normalization: `torch.nn.init.calculate_gain('relu'))`
  - Postnet
- Forward
  - Get start frames (zeros)
  - Parse `decoder_inputs`, teacher forcing = answers
  - concatenate `torch.cat((decoder_input, decoder_inputs), dim=0)`
    - why?
  - Send through **Prenet**
  - Initialize decoder states
    - uses memory = encoder output
    - =~ `get_mask_from_lengths`
    - what does =~ do?
  - Loop over `decoder_inputs`
    - add output of decoder to lists
    - `mel_output`, `gate_output`, `attention_weights` = `self.decode(decoder_input)`
    - parse decoder outputs

### Postnet





