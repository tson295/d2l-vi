# Tiền huấn luyện BERT
<a id="sec_bert-pretraining"></a>

Với mô hình BERT đã được triển khai trong [sec_bert](#sec_bert)
và các ví dụ tiền huấn luyện được tạo từ bộ dữ liệu WikiText-2 trong [sec_bert-dataset](#sec_bert-dataset),
trong phần này chúng ta sẽ tiền huấn luyện BERT trên bộ dữ liệu WikiText-2.

```python
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx

npx.set_np()
```

```python
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

Để bắt đầu, chúng ta nạp bộ dữ liệu WikiText-2 thành các minibatch
gồm các ví dụ tiền huấn luyện cho masked language modeling và next sentence prediction.
Kích thước batch là 512 và độ dài tối đa của một chuỗi đầu vào BERT là 64.
Lưu ý rằng trong mô hình BERT gốc, độ dài tối đa là 512.

```python
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

## Tiền huấn luyện BERT

BERT gốc có hai phiên bản với kích thước mô hình khác nhau [Devlin.Chang.Lee.ea.2018].
Mô hình cơ sở ($\textrm{BERT}_{\textrm{BASE}}$) dùng 12 tầng (các khối Transformer encoder)
với 768 đơn vị ẩn (kích thước ẩn) và 12 head self-attention.
Mô hình lớn ($\textrm{BERT}_{\textrm{LARGE}}$) dùng 24 tầng
với 1024 đơn vị ẩn và 16 head self-attention.
Đáng chú ý, mô hình trước có 110 triệu tham số trong khi mô hình sau có 340 triệu tham số.
Để minh họa cho thuận tiện,
chúng ta định nghĩa [**một BERT nhỏ, dùng 2 tầng, 128 đơn vị ẩn và 2 head self-attention**].

```python
#@tab mxnet
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```

```python
#@tab pytorch
net = d2l.BERTModel(len(vocab), num_hiddens=128, 
                    ffn_num_hiddens=256, num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()
```

Trước khi định nghĩa vòng lặp huấn luyện,
chúng ta định nghĩa một hàm trợ giúp `_get_batch_loss_bert`.
Với một shard của các ví dụ huấn luyện,
hàm này [**tính loss cho cả hai tác vụ masked language modeling và next sentence prediction**].
Lưu ý rằng loss cuối cùng của quá trình tiền huấn luyện BERT
chỉ là tổng của loss masked language modeling
và loss next sentence prediction.

```python
#@tab mxnet
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X_shards,
                         segments_X_shards, valid_lens_x_shards,
                         pred_positions_X_shards, mlm_weights_X_shards,
                         mlm_Y_shards, nsp_y_shards):
    mlm_ls, nsp_ls, ls = [], [], []
    for (tokens_X_shard, segments_X_shard, valid_lens_x_shard,
         pred_positions_X_shard, mlm_weights_X_shard, mlm_Y_shard,
         nsp_y_shard) in zip(
        tokens_X_shards, segments_X_shards, valid_lens_x_shards,
        pred_positions_X_shards, mlm_weights_X_shards, mlm_Y_shards,
        nsp_y_shards):
        # Forward pass
        _, mlm_Y_hat, nsp_Y_hat = net(
            tokens_X_shard, segments_X_shard, valid_lens_x_shard.reshape(-1),
            pred_positions_X_shard)
        # Compute masked language model loss
        mlm_l = loss(
            mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y_shard.reshape(-1),
            mlm_weights_X_shard.reshape((-1, 1)))
        mlm_l = mlm_l.sum() / (mlm_weights_X_shard.sum() + 1e-8)
        # Compute next sentence prediction loss
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        nsp_l = nsp_l.mean()
        mlm_ls.append(mlm_l)
        nsp_ls.append(nsp_l)
        ls.append(mlm_l + nsp_l)
        npx.waitall()
    return mlm_ls, nsp_ls, ls
```

```python
#@tab pytorch
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

Gọi hai hàm trợ giúp vừa nêu,
hàm `train_bert` sau đây
định nghĩa quy trình để [**tiền huấn luyện BERT (`net`) trên bộ dữ liệu WikiText-2 (`train_iter`)**].
Huấn luyện BERT có thể mất rất nhiều thời gian.
Thay vì chỉ định số epoch để huấn luyện
như trong hàm `train_ch13` (xem [sec_image_augmentation](#sec_image_augmentation)),
đầu vào `num_steps` của hàm sau đây
chỉ định số bước lặp để huấn luyện.

```python
#@tab mxnet
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in train_iter:
            (tokens_X_shards, segments_X_shards, valid_lens_x_shards,
             pred_positions_X_shards, mlm_weights_X_shards,
             mlm_Y_shards, nsp_y_shards) = [gluon.utils.split_and_load(
                elem, devices, even_split=False) for elem in batch]
            timer.start()
            with autograd.record():
                mlm_ls, nsp_ls, ls = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X_shards, segments_X_shards,
                    valid_lens_x_shards, pred_positions_X_shards,
                    mlm_weights_X_shards, mlm_Y_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            mlm_l_mean = sum([float(l) for l in mlm_ls]) / len(mlm_ls)
            nsp_l_mean = sum([float(l) for l in nsp_ls]) / len(nsp_ls)
            metric.add(mlm_l_mean, nsp_l_mean, batch[0].shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

```python
#@tab pytorch
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net(*next(iter(train_iter))[:4])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

Chúng ta có thể vẽ cả loss masked language modeling và loss next sentence prediction
trong quá trình tiền huấn luyện BERT.

```python
#@tab all
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

## [**Biểu diễn văn bản bằng BERT**]

Sau khi tiền huấn luyện BERT,
chúng ta có thể dùng nó để biểu diễn một văn bản đơn lẻ, các cặp văn bản, hoặc bất kỳ token nào trong đó.
Hàm sau đây trả về các biểu diễn BERT (`net`) cho tất cả token
trong `tokens_a` và `tokens_b`.

```python
#@tab mxnet
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = np.expand_dims(np.array(vocab[tokens], ctx=devices[0]),
                               axis=0)
    segments = np.expand_dims(np.array(segments, ctx=devices[0]), axis=0)
    valid_len = np.expand_dims(np.array(len(tokens), ctx=devices[0]), axis=0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

```python
#@tab pytorch
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

[**Xét câu "a crane is flying".**]
Nhắc lại biểu diễn đầu vào của BERT đã thảo luận trong [subsec_bert_input_rep](#subsec_bert_input_rep).
Sau khi chèn các token đặc biệt “&lt;cls&gt;” (dùng cho phân loại)
và “&lt;sep&gt;” (dùng để phân tách),
chuỗi đầu vào BERT có độ dài là sáu.
Vì số không là chỉ số của token “&lt;cls&gt;”,
`encoded_text[:, 0, :]` là biểu diễn BERT của toàn bộ câu đầu vào.
Để đánh giá token đa nghĩa "crane",
chúng ta cũng in ra ba phần tử đầu tiên của biểu diễn BERT của token này.

```python
#@tab all
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

[**Bây giờ xét một cặp câu
"a crane driver came" và "he just left".**]
Tương tự, `encoded_pair[:, 0, :]` là kết quả mã hóa của toàn bộ cặp câu từ BERT đã tiền huấn luyện.
Lưu ý rằng ba phần tử đầu tiên của token đa nghĩa "crane" khác với khi ngữ cảnh thay đổi.
Điều này ủng hộ nhận định rằng các biểu diễn BERT nhạy với ngữ cảnh.

```python
#@tab all
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Tokens: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

Trong [chap_nlp_app](#chap_nlp_app), chúng ta sẽ fine-tune một mô hình BERT đã tiền huấn luyện
cho các ứng dụng xử lý ngôn ngữ tự nhiên downstream.

## Tóm tắt

* BERT gốc có hai phiên bản, trong đó mô hình cơ sở có 110 triệu tham số và mô hình lớn có 340 triệu tham số.
* Sau khi tiền huấn luyện BERT, chúng ta có thể dùng nó để biểu diễn một văn bản đơn lẻ, các cặp văn bản, hoặc bất kỳ token nào trong đó.
* Trong thí nghiệm, cùng một token có biểu diễn BERT khác nhau khi ngữ cảnh của nó khác nhau. Điều này ủng hộ nhận định rằng các biểu diễn BERT nhạy với ngữ cảnh.


## Bài tập

1. Trong thí nghiệm, chúng ta có thể thấy loss masked language modeling cao hơn đáng kể so với loss next sentence prediction. Vì sao?
2. Đặt độ dài tối đa của một chuỗi đầu vào BERT là 512 (giống mô hình BERT gốc). Sử dụng các cấu hình của mô hình BERT gốc như $\textrm{BERT}_{\textrm{LARGE}}$. Bạn có gặp lỗi nào khi chạy phần này không? Vì sao?


[Thảo luận](https://discuss.d2l.ai/t/1497)
