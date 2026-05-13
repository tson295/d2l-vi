# Tiền Huấn Luyện word2vec
<a id="sec_word2vec_pretraining"></a>


Chúng ta tiếp tục hiện thực mô hình
skip-gram đã định nghĩa trong
[sec_word2vec](#sec_word2vec).
Sau đó,
chúng ta sẽ tiền huấn luyện word2vec bằng negative sampling
trên tập dữ liệu PTB.
Trước hết,
hãy lấy iterator dữ liệu
và từ vựng cho tập dữ liệu này
bằng cách gọi hàm `d2l.load_data_ptb`,
đã được mô tả trong [sec_word2vec_data](#sec_word2vec_data).

```python
#@tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```python
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## Mô Hình Skip-Gram

Chúng ta hiện thực mô hình skip-gram
bằng cách dùng các tầng embedding và phép nhân ma trận theo batch.
Trước hết, hãy ôn lại
cách các tầng embedding hoạt động.


### Tầng Embedding

Như đã mô tả trong [sec_seq2seq](#sec_seq2seq),
một tầng embedding
ánh xạ chỉ số của token sang vector đặc trưng của nó.
Trọng số của tầng này
là một ma trận có số hàng bằng
kích thước từ điển (`input_dim`) và
số cột bằng
chiều vector của mỗi token (`output_dim`).
Sau khi một mô hình word embedding được huấn luyện,
trọng số này là thứ chúng ta cần.

```python
#@tab mxnet
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```python
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

Đầu vào của một tầng embedding là
chỉ số của một token (từ).
Với bất kỳ chỉ số token $i$ nào,
biểu diễn vector của nó
có thể được lấy từ
hàng thứ $i$ của ma trận trọng số
trong tầng embedding.
Vì chiều vector (`output_dim`)
được đặt là 4,
tầng embedding
trả về các vector có hình dạng (2, 3, 4)
cho một minibatch chỉ số token có hình dạng
(2, 3).

```python
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### Định Nghĩa Lan Truyền Xuôi

Trong lan truyền xuôi,
đầu vào của mô hình skip-gram
bao gồm
các chỉ số từ trung tâm `center`
có hình dạng (kích thước batch, 1)
và
các chỉ số từ ngữ cảnh và từ nhiễu đã nối `contexts_and_negatives`
có hình dạng (kích thước batch, `max_len`),
trong đó `max_len`
được định nghĩa
trong [subsec_word2vec-minibatch-loading](#subsec_word2vec-minibatch-loading).
Hai biến này trước hết được biến đổi từ
chỉ số token thành vector thông qua tầng embedding,
sau đó phép nhân ma trận theo batch của chúng
(đã mô tả trong [subsec_batch_dot](#subsec_batch_dot))
trả về
một đầu ra có hình dạng (kích thước batch, 1, `max_len`).
Mỗi phần tử trong đầu ra là tích vô hướng của
một vector từ trung tâm và một vector từ ngữ cảnh hoặc từ nhiễu.

```python
#@tab mxnet
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```python
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

Hãy in hình dạng đầu ra của hàm `skip_gram` này cho một số đầu vào ví dụ.

```python
#@tab mxnet
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```python
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## Huấn Luyện

Trước khi huấn luyện mô hình skip-gram với negative sampling,
trước hết hãy định nghĩa hàm mất mát của nó.


### Mất Mát Binary Cross-Entropy

Theo định nghĩa của hàm mất mát
cho negative sampling trong [subsec_negative-sampling](#subsec_negative-sampling),
chúng ta sẽ dùng
mất mát binary cross-entropy.

```python
#@tab mxnet
loss = gluon.loss.SigmoidBCELoss()
```

```python
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

Nhắc lại các mô tả của chúng ta
về biến mask
và biến nhãn trong
[subsec_word2vec-minibatch-loading](#subsec_word2vec-minibatch-loading).
Đoạn sau
tính mất mát
binary cross-entropy
cho các biến đã cho.

```python
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

Bên dưới cho thấy
cách các kết quả trên được tính
(theo cách kém hiệu quả hơn)
bằng hàm kích hoạt sigmoid
trong mất mát binary cross-entropy.
Chúng ta có thể xem
hai đầu ra này như
hai mất mát đã chuẩn hóa
được lấy trung bình trên các dự đoán không bị mask.

```python
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### Khởi Tạo Tham Số Mô Hình

Chúng ta định nghĩa hai tầng embedding
cho tất cả các từ trong từ vựng
khi chúng được dùng lần lượt làm từ trung tâm
và từ ngữ cảnh.
Chiều vector từ
`embed_size` được đặt là 100.

```python
#@tab mxnet
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```python
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### Định Nghĩa Vòng Lặp Huấn Luyện

Vòng lặp huấn luyện được định nghĩa bên dưới. Do có phần đệm, phép tính hàm mất mát hơi khác so với các hàm huấn luyện trước đây.

```python
#@tab mxnet
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```python
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(module):
        if type(module) == nn.Embedding:
            nn.init.xavier_uniform_(module.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

Bây giờ chúng ta có thể huấn luyện một mô hình skip-gram bằng negative sampling.

```python
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## Áp Dụng Word Embedding
<a id="subsec_apply-word-embed"></a>


Sau khi huấn luyện mô hình word2vec,
chúng ta có thể dùng độ tương đồng cosine
của các vector từ từ mô hình đã huấn luyện
để
tìm các từ trong từ điển
có ngữ nghĩa giống nhất
với một từ đầu vào.

```python
#@tab mxnet
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```python
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## Tóm Tắt

* Chúng ta có thể huấn luyện mô hình skip-gram với negative sampling bằng các tầng embedding và mất mát binary cross-entropy.
* Các ứng dụng của word embedding bao gồm tìm các từ tương tự về ngữ nghĩa với một từ cho trước dựa trên độ tương đồng cosine của vector từ.


## Bài Tập

1. Dùng mô hình đã huấn luyện, tìm các từ tương tự về ngữ nghĩa cho các từ đầu vào khác. Bạn có thể cải thiện kết quả bằng cách tinh chỉnh siêu tham số không?
1. Khi kho ngữ liệu huấn luyện rất lớn, chúng ta thường lấy mẫu các từ ngữ cảnh và từ nhiễu cho các từ trung tâm trong minibatch hiện tại *khi cập nhật tham số mô hình*. Nói cách khác, cùng một từ trung tâm có thể có các từ ngữ cảnh hoặc từ nhiễu khác nhau trong các epoch huấn luyện khác nhau. Lợi ích của phương pháp này là gì? Hãy thử hiện thực phương pháp huấn luyện này.


[Thảo luận](https://discuss.d2l.ai/t/1335)
