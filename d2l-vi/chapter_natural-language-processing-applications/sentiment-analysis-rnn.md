# Phân tích cảm xúc: Sử dụng mạng nơ-ron hồi tiếp
<a id="sec_sentiment_rnn"></a> 


Tương tự các tác vụ độ tương đồng từ và phép loại suy,
chúng ta cũng có thể áp dụng các vector từ đã tiền huấn luyện
cho phân tích cảm xúc.
Vì bộ dữ liệu đánh giá IMDb
trong [sec_sentiment](#sec_sentiment)
không quá lớn,
việc dùng các biểu diễn văn bản
đã được tiền huấn luyện
trên các kho ngữ liệu quy mô lớn
có thể giảm overfitting của mô hình.
Như một ví dụ cụ thể
được minh họa trong [fig_nlp-map-sa-rnn](#fig_nlp-map-sa-rnn),
chúng ta sẽ biểu diễn mỗi token
bằng mô hình GloVe đã tiền huấn luyện,
và đưa các biểu diễn token này
vào một RNN hai chiều nhiều tầng
để thu được biểu diễn của chuỗi văn bản,
biểu diễn này sau đó sẽ
được biến đổi thành
đầu ra phân tích cảm xúc [Maas.Daly.Pham.ea.2011].
Với cùng ứng dụng downstream này,
về sau chúng ta sẽ xét một lựa chọn
kiến trúc khác.

![Phần này đưa GloVe đã tiền huấn luyện vào một kiến trúc dựa trên RNN cho phân tích cảm xúc.](../img/nlp-map-sa-rnn.svg)
<a id="fig_nlp-map-sa-rnn"></a>

```python
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```python
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## Biểu diễn một văn bản đơn lẻ bằng RNN

Trong các tác vụ phân loại văn bản,
chẳng hạn như phân tích cảm xúc,
một chuỗi văn bản có độ dài thay đổi
sẽ được biến đổi thành các hạng mục có độ dài cố định.
Trong lớp `BiRNN` sau đây,
trong khi mỗi token của một chuỗi văn bản
nhận được biểu diễn GloVe
đã tiền huấn luyện
riêng của nó thông qua tầng embedding
(`self.embedding`),
toàn bộ chuỗi
được mã hóa bởi một RNN hai chiều (`self.encoder`).
Cụ thể hơn,
các trạng thái ẩn (ở tầng cuối cùng)
của LSTM hai chiều
tại cả bước thời gian ban đầu và cuối cùng
được nối lại
làm biểu diễn của chuỗi văn bản.
Biểu diễn văn bản đơn lẻ này
sau đó được biến đổi thành các hạng mục đầu ra
bởi một tầng kết nối đầy đủ (`self.decoder`)
với hai đầu ra ("positive" và "negative").

```python
#@tab mxnet
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs = self.encoder(embeddings)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

```python
#@tab pytorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs, _ = self.encoder(embeddings)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1) 
        outs = self.decoder(encoding)
        return outs
```

Hãy xây dựng một RNN hai chiều với hai tầng ẩn để biểu diễn một văn bản đơn lẻ cho phân tích cảm xúc.

```python
#@tab all
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
```

```python
#@tab mxnet
net.initialize(init.Xavier(), ctx=devices)
```

```python
#@tab pytorch
def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
net.apply(init_weights);
```

## Nạp các vector từ đã tiền huấn luyện

Dưới đây chúng ta nạp các embedding GloVe đã tiền huấn luyện 100 chiều (cần nhất quán với `embed_size`) cho các token trong vocabulary.

```python
#@tab all
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

In hình dạng của các vector
cho tất cả token trong vocabulary.

```python
#@tab all
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

Chúng ta dùng các
vector từ đã tiền huấn luyện này
để biểu diễn các token trong các đánh giá
và sẽ không cập nhật
các vector này trong quá trình huấn luyện.

```python
#@tab mxnet
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

```python
#@tab pytorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

## Huấn luyện và đánh giá mô hình

Bây giờ chúng ta có thể huấn luyện RNN hai chiều cho phân tích cảm xúc.

```python
#@tab mxnet
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```python
#@tab pytorch
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

Chúng ta định nghĩa hàm sau để dự đoán cảm xúc của một chuỗi văn bản bằng mô hình đã huấn luyện `net`.

```python
#@tab mxnet
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = np.array(vocab[sequence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'
```

```python
#@tab pytorch
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

Cuối cùng, hãy dùng mô hình đã huấn luyện để dự đoán cảm xúc cho hai câu đơn giản.

```python
#@tab all
predict_sentiment(net, vocab, 'this movie is so great')
```

```python
#@tab all
predict_sentiment(net, vocab, 'this movie is so bad')
```

## Tóm tắt

* Các vector từ đã tiền huấn luyện có thể biểu diễn từng token riêng lẻ trong một chuỗi văn bản.
* RNN hai chiều có thể biểu diễn một chuỗi văn bản, chẳng hạn thông qua việc nối các trạng thái ẩn tại bước thời gian ban đầu và cuối cùng. Biểu diễn văn bản đơn lẻ này có thể được biến đổi thành các hạng mục bằng một tầng kết nối đầy đủ.


## Bài tập

1. Tăng số epoch. Bạn có thể cải thiện độ chính xác huấn luyện và kiểm tra không? Còn việc điều chỉnh các siêu tham số khác thì sao?
1. Dùng các vector từ đã tiền huấn luyện lớn hơn, chẳng hạn embedding GloVe 300 chiều. Điều này có cải thiện độ chính xác phân loại không?
1. Chúng ta có thể cải thiện độ chính xác phân loại bằng cách dùng tokenization của spaCy không? Bạn cần cài đặt spaCy (`pip install spacy`) và cài đặt gói tiếng Anh (`python -m spacy download en`). Trong code, trước tiên hãy import spaCy (`import spacy`). Sau đó, nạp gói tiếng Anh của spaCy (`spacy_en = spacy.load('en')`). Cuối cùng, định nghĩa hàm `def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]` và thay thế hàm `tokenizer` ban đầu. Lưu ý các dạng khác nhau của token cụm từ trong GloVe và spaCy. Ví dụ, token cụm từ "new york" có dạng "new-york" trong GloVe và dạng "new york" sau khi tokenization bằng spaCy.


[Thảo luận](https://discuss.d2l.ai/t/1424)
