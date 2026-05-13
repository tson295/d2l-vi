# Transformer cho Thị Giác
<a id="sec_vision-transformer"></a>

Kiến trúc Transformer ban đầu được đề xuất
cho học chuỗi-sang-chuỗi,
với trọng tâm là dịch máy.
Sau đó, Transformer trở thành mô hình được chọn
trong nhiều tác vụ xử lý ngôn ngữ tự nhiên [Radford.Narasimhan.Salimans.ea.2018, Radford.Wu.Child.ea.2019, brown2020language, Devlin.Chang.Lee.ea.2018, raffel2020exploring].
Tuy nhiên, trong lĩnh vực thị giác máy tính
kiến trúc chiếm ưu thế vẫn là
CNN ([chap_modern_cnn](#chap_modern_cnn)).
Một cách tự nhiên, các nhà nghiên cứu bắt đầu tự hỏi
liệu có thể làm tốt hơn không
bằng cách điều chỉnh các mô hình Transformer cho dữ liệu ảnh.
Câu hỏi này đã khơi dậy sự quan tâm rất lớn
trong cộng đồng thị giác máy tính.
Gần đây, ramachandran2019stand đã đề xuất
một sơ đồ để thay thế tích chập bằng self-attention.
Tuy nhiên, việc sử dụng các mẫu chuyên biệt trong attention
làm cho nó khó mở rộng mô hình trên các bộ tăng tốc phần cứng.
Sau đó, cordonnier2020relationship đã chứng minh về mặt lý thuyết
rằng self-attention có thể học để hành xử tương tự như tích chập.
Thực nghiệm, các patch $2 \times 2$ được lấy từ ảnh làm đầu vào,
nhưng kích thước patch nhỏ làm cho mô hình
chỉ áp dụng được cho dữ liệu ảnh có độ phân giải thấp.

Không có các ràng buộc cụ thể về kích thước patch,
*vision Transformer* (ViT)
trích xuất các patch từ ảnh
và đưa chúng vào một bộ mã hóa Transformer
để có được biểu diễn toàn cục,
cuối cùng sẽ được biến đổi để phân loại [Dosovitskiy.Beyer.Kolesnikov.ea.2021].
Đáng chú ý, Transformer cho thấy khả năng mở rộng tốt hơn CNN:
và khi huấn luyện các mô hình lớn hơn trên các tập dữ liệu lớn hơn,
vision Transformer vượt trội hơn ResNet với biên độ đáng kể.
Tương tự như bối cảnh thiết kế kiến trúc mạng trong xử lý ngôn ngữ tự nhiên,
Transformer cũng đã trở thành một yếu tố thay đổi cuộc chơi trong thị giác máy tính.

```python
from d2l import torch as d2l
import torch
from torch import nn
```


## Mô Hình

[fig_vit](#fig_vit) mô tả
kiến trúc mô hình của vision Transformer.
Kiến trúc này bao gồm một gốc
phân tách ảnh thành các patch,
một thân dựa trên bộ mã hóa Transformer nhiều lớp,
và một đầu biến đổi biểu diễn toàn cục
thành nhãn đầu ra.

![Kiến trúc vision Transformer. Trong ví dụ này, một ảnh được chia thành chín patch. Một token đặc biệt "&lt;cls&gt;" và chín patch ảnh được làm phẳng được biến đổi qua patch embedding và $\mathit{n}$ khối bộ mã hóa Transformer thành mười biểu diễn, tương ứng. Biểu diễn "&lt;cls&gt;" được biến đổi thêm thành nhãn đầu ra.](../img/vit.svg)
<a id="fig_vit"></a>

Xét một ảnh đầu vào có chiều cao $h$, chiều rộng $w$,
và $c$ kênh.
Chỉ định chiều cao và chiều rộng patch đều là $p$,
ảnh được chia thành một chuỗi gồm $m = hw/p^2$ patch,
trong đó mỗi patch được làm phẳng thành một vector có độ dài $cp^2$.
Bằng cách này, các patch ảnh có thể được xử lý tương tự như các token trong chuỗi văn bản bởi các bộ mã hóa Transformer.
Một token đặc biệt "&lt;cls&gt;" (lớp) và
$m$ patch ảnh được làm phẳng được chiếu tuyến tính
thành một chuỗi gồm $m+1$ vector,
được cộng với các embedding vị trí có thể học.
Bộ mã hóa Transformer nhiều lớp
biến đổi $m+1$ vector đầu vào
thành cùng số lượng biểu diễn vector đầu ra có cùng độ dài.
Nó hoạt động giống hệt với bộ mã hóa Transformer gốc trong [fig_transformer](#fig_transformer),
chỉ khác ở vị trí của chuẩn hóa.
Vì token "&lt;cls&gt;" chú ý đến tất cả các patch ảnh
thông qua self-attention (xem [fig_cnn-rnn-self-attention](#fig_cnn-rnn-self-attention)),
biểu diễn của nó từ đầu ra bộ mã hóa Transformer
sẽ được biến đổi thêm thành nhãn đầu ra.

## Patch Embedding

Để triển khai vision Transformer, hãy bắt đầu
với patch embedding trong [fig_vit](#fig_vit).
Chia ảnh thành các patch
và chiếu tuyến tính các patch được làm phẳng này
có thể được đơn giản hóa thành một phép tích chập đơn lẻ,
trong đó cả kích thước kernel và kích thước bước đều được đặt bằng kích thước patch.

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
```


Trong ví dụ sau, nhận ảnh có chiều cao và chiều rộng `img_size` làm đầu vào,
patch embedding xuất ra `(img_size//patch_size)**2` patch
được chiếu tuyến tính thành các vector có độ dài `num_hiddens`.

```python
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros(batch_size, 3, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```


## Bộ Mã Hóa Vision Transformer
<a id="subsec_vit-encoder"></a>

MLP của bộ mã hóa vision Transformer hơi khác
so với FFN theo vị trí của bộ mã hóa Transformer gốc
(xem [subsec_positionwise-ffn](#subsec_positionwise-ffn)).
Thứ nhất, ở đây hàm kích hoạt sử dụng đơn vị tuyến tính lỗi Gaussian (GELU),
có thể được coi là một phiên bản mượt mà hơn của ReLU [Hendrycks.Gimpel.2016].
Thứ hai, dropout được áp dụng cho đầu ra của mỗi lớp kết nối đầy đủ trong MLP để điều chuẩn.

```python
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
```


Triển khai khối bộ mã hóa vision Transformer
chỉ tuân theo thiết kế pre-normalization trong [fig_vit](#fig_vit),
trong đó chuẩn hóa được áp dụng ngay *trước* attention đa đầu hoặc MLP.
Ngược lại với post-normalization ("cộng & chuẩn hóa" trong [fig_transformer](#fig_transformer)),
trong đó chuẩn hóa được đặt ngay *sau* kết nối phần dư,
pre-normalization dẫn đến huấn luyện hiệu quả hơn cho Transformer [baevski2018adaptive, wang2019learning, xiong2020layer].

```python
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))
```


Giống như trong [subsec_transformer-encoder](#subsec_transformer-encoder),
không có khối bộ mã hóa vision Transformer nào thay đổi hình dạng đầu vào của nó.

```python
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```


## Kết Hợp Tất Cả Lại

Lượt truyền thuận của vision Transformer dưới đây rất đơn giản.
Đầu tiên, ảnh đầu vào được đưa vào một thể hiện `PatchEmbedding`,
đầu ra của nó được nối với embedding token "&lt;cls&gt;".
Chúng được cộng với các embedding vị trí có thể học trước dropout.
Sau đó đầu ra được đưa vào bộ mã hóa Transformer xếp chồng `num_blks` thể hiện của lớp `ViTBlock`.
Cuối cùng, biểu diễn của token "&lt;cls&gt;" được chiếu bởi đầu mạng.

```python
class ViT(d2l.Classifier):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(d2l.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
```


## Huấn Luyện

Huấn luyện vision Transformer trên tập dữ liệu Fashion-MNIST giống như cách CNN được huấn luyện trong [chap_modern_cnn](#chap_modern_cnn).

```python
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
trainer.fit(model, data)
```

## Tóm Tắt và Thảo Luận

Bạn có thể nhận thấy rằng với các tập dữ liệu nhỏ như Fashion-MNIST,
vision Transformer đã triển khai của chúng ta
không vượt trội hơn ResNet trong [sec_resnet](#sec_resnet).
Có thể quan sát tương tự ngay cả trên tập dữ liệu ImageNet (1,2 triệu ảnh).
Điều này là vì Transformer *thiếu* những nguyên tắc hữu ích trong tích chập,
chẳng hạn như tính bất biến dịch chuyển và tính cục bộ ([sec_why-conv](#sec_why-conv)).
Tuy nhiên, bức tranh thay đổi khi huấn luyện các mô hình lớn hơn trên các tập dữ liệu lớn hơn (ví dụ: 300 triệu ảnh),
nơi vision Transformer vượt trội hơn ResNet với biên độ lớn trong phân loại ảnh, chứng minh
tính vượt trội nội tại của Transformer về khả năng mở rộng [Dosovitskiy.Beyer.Kolesnikov.ea.2021].
Sự ra đời của vision Transformer
đã thay đổi bối cảnh thiết kế mạng để mô hình hóa dữ liệu ảnh.
Chúng sớm được chứng minh là hiệu quả trên tập dữ liệu ImageNet
với các chiến lược huấn luyện tiết kiệm dữ liệu của DeiT [touvron2021training].
Tuy nhiên, độ phức tạp bậc hai của self-attention
([sec_self-attention-and-positional-encoding](#sec_self-attention-and-positional-encoding))
làm cho kiến trúc Transformer
ít phù hợp hơn cho các ảnh có độ phân giải cao hơn.
Hướng tới một mạng xương sống đa mục đích trong thị giác máy tính,
Swin Transformer đã giải quyết độ phức tạp tính toán bậc hai
theo kích thước ảnh ([subsec_cnn-rnn-self-attention](#subsec_cnn-rnn-self-attention))
và khôi phục lại các tiên nghiệm giống tích chập,
mở rộng khả năng ứng dụng của Transformer cho một loạt các tác vụ thị giác máy tính
ngoài phân loại ảnh với kết quả tốt nhất [liu2021swin].

## Bài Tập

1. Giá trị `img_size` ảnh hưởng như thế nào đến thời gian huấn luyện?
1. Thay vì chiếu biểu diễn token "&lt;cls&gt;" thành đầu ra, bạn sẽ chiếu các biểu diễn patch trung bình như thế nào? Hãy triển khai thay đổi này và xem nó ảnh hưởng như thế nào đến độ chính xác.
1. Bạn có thể sửa đổi các siêu tham số để cải thiện độ chính xác của vision Transformer không?

[Discussions](https://discuss.d2l.ai/t/8943)
