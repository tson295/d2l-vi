# Phân tích cảm xúc và bộ dữ liệu
<a id="sec_sentiment"></a>


Với sự phát triển mạnh của mạng xã hội trực tuyến
và các nền tảng đánh giá,
một lượng lớn
dữ liệu mang ý kiến
đã được ghi lại,
có tiềm năng lớn trong việc
hỗ trợ các quá trình ra quyết định.
*Phân tích cảm xúc*
nghiên cứu cảm xúc của con người
trong văn bản họ tạo ra,
chẳng hạn như đánh giá sản phẩm,
bình luận blog,
và
thảo luận trên diễn đàn.
Nó có ứng dụng rộng rãi
trong nhiều lĩnh vực đa dạng như
chính trị (ví dụ, phân tích cảm xúc của công chúng đối với chính sách),
tài chính (ví dụ, phân tích cảm xúc của thị trường),
và
marketing (ví dụ, nghiên cứu sản phẩm và quản lý thương hiệu).

Vì cảm xúc
có thể được phân loại
thành các cực hoặc thang đo rời rạc (ví dụ, tích cực và tiêu cực),
chúng ta có thể xem
phân tích cảm xúc
như một tác vụ phân loại văn bản,
biến đổi một chuỗi văn bản có độ dài thay đổi
thành một hạng mục văn bản có độ dài cố định.
Trong chương này,
chúng ta sẽ dùng [bộ dữ liệu đánh giá phim lớn](https://ai.stanford.edu/%7Eamaas/data/sentiment/) của Stanford
cho phân tích cảm xúc.
Bộ dữ liệu này gồm một tập huấn luyện và một tập kiểm tra,
mỗi tập chứa 25000 đánh giá phim được tải xuống từ IMDb.
Trong cả hai bộ dữ liệu,
số lượng nhãn
"positive" và "negative" là bằng nhau,
biểu thị các cực cảm xúc khác nhau.

```python
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```python
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

## Đọc bộ dữ liệu

Trước tiên, tải xuống và giải nén bộ dữ liệu đánh giá IMDb này
vào đường dẫn `../data/aclImdb`.

```python
#@tab all
d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz', 
                          '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

Tiếp theo, đọc các bộ dữ liệu huấn luyện và kiểm tra. Mỗi ví dụ là một đánh giá và nhãn của nó: 1 cho "positive" và 0 cho "negative".

```python
#@tab all
def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[:60])
```

## Tiền xử lý bộ dữ liệu

Xem mỗi từ là một token
và lọc bỏ các từ xuất hiện ít hơn 5 lần,
chúng ta tạo một vocabulary từ bộ dữ liệu huấn luyện.

```python
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

Sau khi token hóa,
hãy vẽ histogram của
độ dài đánh giá tính theo token.

```python
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

Như mong đợi,
các đánh giá có độ dài khác nhau.
Để xử lý
một minibatch các đánh giá như vậy ở mỗi thời điểm,
chúng ta đặt độ dài của mỗi đánh giá là 500 bằng cách cắt ngắn và đệm,
tương tự như
bước tiền xử lý
cho bộ dữ liệu dịch máy
trong [sec_machine_translation](#sec_machine_translation).

```python
#@tab all
num_steps = 500  # sequence length
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## Tạo các iterator dữ liệu

Bây giờ chúng ta có thể tạo các iterator dữ liệu.
Ở mỗi lần lặp, một minibatch các ví dụ được trả về.

```python
#@tab mxnet
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```python
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## Ghép mọi thứ lại với nhau

Cuối cùng, chúng ta gói các bước trên vào hàm `load_data_imdb`.
Hàm này trả về các iterator dữ liệu huấn luyện và kiểm tra cùng vocabulary của bộ dữ liệu đánh giá IMDb.

```python
#@tab mxnet
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```python
#@tab pytorch
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## Tóm tắt

* Phân tích cảm xúc nghiên cứu cảm xúc của con người trong văn bản họ tạo ra, và được xem là một bài toán phân loại văn bản biến đổi một chuỗi văn bản có độ dài thay đổi
thành một hạng mục văn bản có độ dài cố định.
* Sau khi tiền xử lý, chúng ta có thể nạp bộ dữ liệu đánh giá phim lớn của Stanford (bộ dữ liệu đánh giá IMDb) vào các iterator dữ liệu cùng một vocabulary.


## Bài tập


1. Chúng ta có thể sửa những siêu tham số nào trong phần này để tăng tốc huấn luyện các mô hình phân tích cảm xúc?
1. Bạn có thể triển khai một hàm để nạp bộ dữ liệu [đánh giá Amazon](https://snap.stanford.edu/data/web-Amazon.html) vào các iterator dữ liệu và nhãn cho phân tích cảm xúc không?


[Thảo luận](https://discuss.d2l.ai/t/1387)
