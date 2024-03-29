# Vision Transformers

> You can read more about the Vision Transformers in the [[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale]](https://arxiv.org/abs/1506.02025)

The abstract from the paper is the following:

> *While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.*



The objective is to explain the following classes from [this](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py) implementation block by block:

- Embeddings
- Encoder
- Block
- Attention
- MLP

The vision transformer treats an input image as a sequence of patches. 

<img src="https://github.com/gokul-pv/DawnOfTransformers/blob/main/VisualTransformers/Images/image1.gif" style="zoom: 33%;" />



How the ViT works in a nutshell:

1. Split the image into patches (16x16)
2. Flatten the patches
3. Produce lower-dimensional linear embeddings from the flattened patches
4. Add positional embeddings (so patches can retain their positional information)
5. Feed the sequence as an input to a standard transformer encoder
6. Pre-train the model with image labels (fully supervised on a huge dataset)
7. Finetune on the downstream dataset for image classification. 



## Embedding

- Let us start with a 224x224x3 image. We divide the image into 14 patches of 16x16x3  size. 

- These patches are passed through the same linear layer, a Conv2d  layer is used for this, to get 1x768. This is obtained by using a kernel_size and stride equal to the `patch_size`.  

- Next step is to add the CLS token and the position embedding. cls_tokens is a torch Parameter and is randomly initialized. In the  forward method it will be expanded to match the  batch size B. CLS token is a vector of size 1x768, and nn.Parameter makes it a learnable parameter

- For the model to know the original position of the patches, we need  to pass the spatial information. In ViT we let the model learn it. The  position embedding is just a tensor of shape 1, n_patches + 1(token),  hidden_size that is added to the projected patches. In the forward  function below, position_embeddings is summed up with the patches (x)

  
  
  ![embed](https://github.com/gokul-pv/DawnOfTransformers/blob/main/VisualTransformers/Images/embedding.png)
  
  
  
  ```python
    class Embeddings(nn.Module):
        """Construct the embeddings from patch, position embeddings.
        """
        def __init__(self, config, img_size, in_channels=3):
            super(Embeddings, self).__init__()
            self.hybrid = None
            img_size = _pair(img_size)
  
            if config.patches.get("grid") is not None:
                grid_size = config.patches["grid"]
                patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
                n_patches = (img_size[0] // 16) * (img_size[1] // 16)
                self.hybrid = True
            else:
                patch_size = _pair(config.patches["size"])
                n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
                self.hybrid = False
  
            if self.hybrid:
                self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                             width_factor=config.resnet.width_factor)
                in_channels = self.hybrid_model.width * 16
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=config.hidden_size,
                                           kernel_size=patch_size,
                                           stride=patch_size)
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
  
            self.dropout = Dropout(config.transformer["dropout_rate"])
  
        def forward(self, x):
            B = x.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)
  
            if self.hybrid:
                x = self.hybrid_model(x)
            x = self.patch_embeddings(x)
            x = x.flatten(2)
            x = x.transpose(-1, -2)
            x = torch.cat((cls_tokens, x), dim=1)
  
            embeddings = x + self.position_embeddings
            embeddings = self.dropout(embeddings)
            return embeddings
  ```



## Encoder

The resulting tensor is passeed into a Transformer. In ViT only the  Encoder is used, the Transformer encoder module comprises a Multi-Head  Self Attention ( MSA ) layer and a Multi-Layer Perceptron (MLP) layer.  The encoder combines multiple layers of Transformer Blocks in a  sequential manner. 

<img src="https://github.com/gokul-pv/DawnOfTransformers/blob/main/VisualTransformers/Images/vit-07.png" style="zoom: 25%;" />

```python
    class Encoder(nn.Module):
        def __init__(self, config, vis):
            super(Encoder, self).__init__()
            self.vis = vis
            self.layer = nn.ModuleList()
            self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
            for _ in range(config.transformer["num_layers"]):
                layer = Block(config, vis)
                self.layer.append(copy.deepcopy(layer))

        def forward(self, hidden_states):
            attn_weights = []
            for layer_block in self.layer:
                hidden_states, weights = layer_block(hidden_states)
                if self.vis:
                    attn_weights.append(weights)
            encoded = self.encoder_norm(hidden_states)
            return encoded, attn_weights
```

 

## Block

The Block class combines both the attention module and the MLP module with layer normalization, dropout and residual connections. 

```python

    class Block(nn.Module):
        def __init__(self, config, vis):
            super(Block, self).__init__()
            self.hidden_size = config.hidden_size
            self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn = Mlp(config)
            self.attn = Attention(config, vis)

        def forward(self, x):
            h = x
            x = self.attention_norm(x)
            x, weights = self.attn(x)
            x = x + h

            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + h
            return x, weights
```



## Attention

Attention Module is used to perform self-attention operation allowing the model to attend information from different representation subspaces on an input sequence of embeddings. The sequence of operations is as follows :-

<img src="https://github.com/gokul-pv/DawnOfTransformers/blob/main/VisualTransformers/Images/iz6yaKgHqq.png" style="zoom: 50%;" />



```
Q → 197x768 | Q_LINEAR_LAYER (768x768) | Q-Vector (197x768)

K → 197x768 | K_LINEAR_LAYER (768x768) | K-Vector (197x768)

V → 197x768 | V_LINEAR_LAYER (768x768) | V-Vector (197x768)

SoftMax(Q×KT)= 197×197 = Attention_Matrix

Attention_Matrix(197x197) × V(197×768) → Output(197×768)
```



The attention takes three inputs, the queries, keys, and values,  reshapes and computes the attention matrix using queries and values and  use it to “attend” to the values. In this case, we are using multi-head  attention meaning that the computation is split across n = 12 heads with  smaller input size.



```
QKV → 197x768 | QKV_LINEAR_LAYER (768x768x3) | QKV-Vector (197x2304)

QKV-Vector (197x2304) → 12Head-QKV-Vector(197x3x12x64)

DESTACK

Q-Vector (12x197x64) | K-Vector (12x197x64) | V-Vector (12x197x64)

SoftMax(Q×KT)= 12×197×197 = Attention_Matrix

Attention_Matrix(12x197x197) × V(12x197x64) → Output(12x197×64) → Output(197×768)
```



```python
  class Attention(nn.Module):
      def __init__(self, config, vis):
          super(Attention, self).__init__()
          self.vis = vis
          self.num_attention_heads = config.transformer["num_heads"]
          self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
          self.all_head_size = self.num_attention_heads * self.attention_head_size

          self.query = Linear(config.hidden_size, self.all_head_size)
          self.key = Linear(config.hidden_size, self.all_head_size)
          self.value = Linear(config.hidden_size, self.all_head_size)

          self.out = Linear(config.hidden_size, config.hidden_size)
          self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
          self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

          self.softmax = Softmax(dim=-1)

      def transpose_for_scores(self, x):
          new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
          x = x.view(*new_x_shape)
          return x.permute(0, 2, 1, 3)

      def forward(self, hidden_states):
          mixed_query_layer = self.query(hidden_states)
          mixed_key_layer = self.key(hidden_states)
          mixed_value_layer = self.value(hidden_states)

          query_layer = self.transpose_for_scores(mixed_query_layer)
          key_layer = self.transpose_for_scores(mixed_key_layer)
          value_layer = self.transpose_for_scores(mixed_value_layer)

          attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
          attention_scores = attention_scores / math.sqrt(self.attention_head_size)
          attention_probs = self.softmax(attention_scores)
          weights = attention_probs if self.vis else None
          attention_probs = self.attn_dropout(attention_probs)

          context_layer = torch.matmul(attention_probs, value_layer)
          context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
          new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
          context_layer = context_layer.view(*new_context_layer_shape)
          attention_output = self.out(context_layer)
          attention_output = self.proj_dropout(attention_output)
          return attention_output, weights
```



## MLP

The attension output is passed to MLP,  which has two sequential  linear layers with GELU activation function. 

Gaussian Error Linear Unit (GELu), an activation function used in the most recent Transformers – Google's BERT and OpenAI's GPT-2. The paper  is from 2016, but is only catching attention up until recently. 

This activation function takes the form of this equation:

![](https://github.com/gokul-pv/DawnOfTransformers/blob/main/VisualTransformers/Images/gelu.PNG)

Below is the graph for the gaussian  error linear unit:

![](https://github.com/gokul-pv/DawnOfTransformers/blob/main/VisualTransformers/Images/gelu_graph.PNG)



```python
    class Mlp(nn.Module):
        def __init__(self, config):
            super(Mlp, self).__init__()
            self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
            self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
            self.act_fn = ACT2FN["gelu"]
            self.dropout = Dropout(config.transformer["dropout_rate"])

            self._init_weights()

        def _init_weights(self):
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.normal_(self.fc1.bias, std=1e-6)
            nn.init.normal_(self.fc2.bias, std=1e-6)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act_fn(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x
```



## References

- [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- [https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py)

