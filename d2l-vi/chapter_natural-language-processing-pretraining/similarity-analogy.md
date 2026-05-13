# Độ Tương Đồng và Loại Suy Từ
<a id="sec_synonyms"></a>

Trong [sec_word2vec_pretraining](#sec_word2vec_pretraining),
chúng ta đã huấn luyện một mô hình word2vec trên một tập dữ liệu nhỏ,
và áp dụng nó
để tìm các từ tương tự về ngữ nghĩa
cho một từ đầu vào.
Trong thực tế,
các vector từ được tiền huấn luyện
trên các kho ngữ liệu lớn có thể được
áp dụng cho các tác vụ xử lý ngôn ngữ tự nhiên
downstream,
sẽ được trình bày sau
trong [chap_nlp_app](#chap_nlp_app).
Để minh họa
ngữ nghĩa của các vector từ tiền huấn luyện
từ các kho ngữ liệu lớn một cách trực quan,
hãy áp dụng chúng
trong các tác vụ độ tương đồng từ và loại suy từ.

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

## Tải Vector Từ Đã Tiền Huấn Luyện

Bên dưới liệt kê các embedding GloVe đã tiền huấn luyện với chiều 50, 100, và 300,
có thể được tải xuống từ [trang web GloVe](https://nlp.stanford.edu/projects/glove/).
Các embedding fastText đã tiền huấn luyện có sẵn trong nhiều ngôn ngữ.
Ở đây chúng ta xét một phiên bản tiếng Anh (300 chiều "wiki.en") có thể tải xuống từ
[trang web fastText](https://fasttext.cc/).

```python
#@tab all
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

Để tải các embedding GloVe và fastText đã tiền huấn luyện này, chúng ta định nghĩa lớp `TokenEmbedding` sau.

```python
#@tab all
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

Bên dưới, chúng ta tải
embedding GloVe 50 chiều
(đã tiền huấn luyện trên một tập con Wikipedia).
Khi tạo thực thể `TokenEmbedding`,
tệp embedding được chỉ định sẽ được tải xuống nếu nó
chưa có.

```python
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

In kích thước từ vựng. Từ vựng chứa 400000 từ (token) và một token không biết đặc biệt.

```python
#@tab all
len(glove_6b50d)
```

Chúng ta có thể lấy chỉ số của một từ trong từ vựng, và ngược lại.

```python
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Áp Dụng Vector Từ Đã Tiền Huấn Luyện

Dùng các vector GloVe đã tải,
chúng ta sẽ minh họa ngữ nghĩa của chúng
bằng cách áp dụng chúng
trong các tác vụ độ tương đồng từ và loại suy từ sau.


### Độ Tương Đồng Từ

Tương tự [subsec_apply-word-embed](#subsec_apply-word-embed),
để tìm các từ tương tự về ngữ nghĩa
cho một từ đầu vào
dựa trên độ tương đồng cosine giữa
các vector từ,
chúng ta hiện thực hàm `knn`
($k$ láng giềng gần nhất) sau.

```python
#@tab mxnet
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```python
#@tab pytorch
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

Sau đó, chúng ta
tìm kiếm các từ tương tự
bằng các vector từ đã tiền huấn luyện
từ thực thể `TokenEmbedding` `embed`.

```python
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

Từ vựng của các vector từ đã tiền huấn luyện
trong `glove_6b50d` chứa 400000 từ và một token không biết đặc biệt.
Loại trừ từ đầu vào và token không biết,
trong từ vựng này
hãy tìm
ba từ tương tự nhất về ngữ nghĩa
với từ "chip".

```python
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

Bên dưới xuất ra các từ tương tự
với "baby" và "beautiful".

```python
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```python
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Loại Suy Từ

Bên cạnh việc tìm các từ tương tự,
chúng ta cũng có thể áp dụng vector từ
cho các tác vụ loại suy từ.
Ví dụ,
“man”:“woman”::“son”:“daughter”
là dạng của một phép loại suy từ:
“man” đối với “woman” giống như “son” đối với “daughter”.
Cụ thể,
tác vụ hoàn thành loại suy từ
có thể được định nghĩa như sau:
với một loại suy từ
$a : b :: c : d$, cho ba từ đầu $a$, $b$ và $c$, hãy tìm $d$.
Ký hiệu vector của từ $w$ là $\textrm{vec}(w)$.
Để hoàn thành phép loại suy,
chúng ta sẽ tìm từ
có vector giống nhất
với kết quả của $\textrm{vec}(c)+\textrm{vec}(b)-\textrm{vec}(a)$.

```python
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

Hãy kiểm chứng phép loại suy "nam-nữ" bằng các vector từ đã tải.

```python
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

Bên dưới hoàn thành một phép loại suy
“thủ đô-quốc gia”:
“beijing”:“china”::“tokyo”:“japan”.
Điều này minh họa
ngữ nghĩa trong các vector từ đã tiền huấn luyện.

```python
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

Với phép loại suy
“tính từ-tính từ so sánh nhất”
như
“bad”:“worst”::“big”:“biggest”,
ta có thể thấy rằng các vector từ đã tiền huấn luyện
có thể nắm bắt thông tin cú pháp.

```python
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

Để cho thấy khái niệm
thì quá khứ được nắm bắt trong các vector từ đã tiền huấn luyện,
chúng ta có thể kiểm tra cú pháp bằng
phép loại suy "thì hiện tại-thì quá khứ": “do”:“did”::“go”:“went”.

```python
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## Tóm Tắt

* Trong thực tế, các vector từ được tiền huấn luyện trên các kho ngữ liệu lớn có thể được áp dụng cho các tác vụ xử lý ngôn ngữ tự nhiên downstream.
* Các vector từ đã tiền huấn luyện có thể được áp dụng cho các tác vụ độ tương đồng từ và loại suy từ.


## Bài Tập

1. Kiểm tra kết quả fastText bằng `TokenEmbedding('wiki.en')`.
1. Khi từ vựng cực kỳ lớn, làm thế nào để tìm các từ tương tự hoặc hoàn thành một phép loại suy từ nhanh hơn?


[Thảo luận](https://discuss.d2l.ai/t/1336)
