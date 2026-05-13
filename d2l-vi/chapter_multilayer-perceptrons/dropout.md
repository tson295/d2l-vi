# Dropout
<a id="sec_dropout"></a>


Hãy suy nghĩ ngắn gọn về những gì chúng ta
mong đợi từ một mô hình dự đoán tốt.
Chúng ta muốn nó hoạt động tốt trên dữ liệu chưa thấy.
Lý thuyết tổng quát hóa cổ điển
gợi ý rằng để thu hẹp khoảng cách giữa
hiệu suất huấn luyện và kiểm tra,
chúng ta nên hướng tới một mô hình đơn giản.
Sự đơn giản có thể đến dưới dạng
một số chiều nhỏ.
Chúng ta đã khám phá điều này khi thảo luận về
các hàm cơ sở đơn thức của mô hình tuyến tính
trong [sec_generalization_basics](#sec_generalization_basics).
Ngoài ra, như chúng ta đã thấy khi thảo luận về suy giảm trọng số
(chuẩn hóa $\ell_2$) trong [sec_weight_decay](#sec_weight_decay),
(nghịch đảo của) chuẩn của các tham số cũng
đại diện cho một thước đo hữu ích về sự đơn giản.
Một khái niệm hữu ích khác về sự đơn giản là độ mượt,
tức là hàm không nên nhạy cảm
với những thay đổi nhỏ đối với đầu vào của nó.
Ví dụ, khi chúng ta phân loại hình ảnh,
chúng ta mong đợi rằng thêm một số nhiễu ngẫu nhiên
vào các pixel sẽ hầu như vô hại.

Bishop.1995 đã chính thức hóa
ý tưởng này khi ông chứng minh rằng huấn luyện với nhiễu đầu vào
tương đương với chuẩn hóa Tikhonov.
Công trình này đã vẽ một mối liên hệ toán học rõ ràng
giữa yêu cầu rằng một hàm phải mượt (và do đó đơn giản),
và yêu cầu rằng nó phải đàn hồi
trước các nhiễu loạn trong đầu vào.

Sau đó, Srivastava.Hinton.Krizhevsky.ea.2014
đã phát triển một ý tưởng thông minh về cách áp dụng ý tưởng của Bishop
cho các lớp bên trong của mạng.
Ý tưởng của họ, được gọi là *dropout*, liên quan đến
việc đưa nhiễu vào trong khi tính toán
mỗi lớp bên trong trong quá trình lan truyền xuôi,
và nó đã trở thành một kỹ thuật tiêu chuẩn
để huấn luyện mạng nơ-ron.
Phương pháp này được gọi là *dropout* vì chúng ta thực sự
*loại bỏ* một số nơ-ron trong quá trình huấn luyện.
Trong suốt quá trình huấn luyện, ở mỗi lần lặp,
dropout tiêu chuẩn bao gồm việc đưa về không
một phần các nút trong mỗi lớp
trước khi tính toán lớp tiếp theo.

Để rõ ràng, chúng ta đang áp đặt
cách giải thích của riêng mình với liên kết đến Bishop.
Bài báo gốc về dropout
cung cấp trực giác thông qua một
sự tương tự đáng ngạc nhiên với sinh sản hữu tính.
Các tác giả lập luận rằng quá khớp mạng nơ-ron
được đặc trưng bởi một trạng thái trong đó
mỗi lớp dựa vào một
mẫu kích hoạt cụ thể trong lớp trước,
gọi điều kiện này là *đồng thích nghi*.
Dropout, họ khẳng định, phá vỡ đồng thích nghi
giống như sinh sản hữu tính được lập luận là
phá vỡ các gen đồng thích nghi.
Mặc dù sự biện hộ như vậy cho lý thuyết này chắc chắn còn tranh luận,
bản thân kỹ thuật dropout đã chứng tỏ sức bền,
và các dạng dropout khác nhau được cài đặt
trong hầu hết các thư viện deep learning.


Thách thức chính là cách đưa nhiễu này vào.
Một ý tưởng là đưa nó vào theo cách *không thiên lệch*
để giá trị kỳ vọng của mỗi lớp---trong khi cố định
các lớp khác---bằng với giá trị mà nó sẽ nhận được khi không có nhiễu.
Trong công trình của Bishop, ông đã thêm nhiễu Gaussian
vào đầu vào của mô hình tuyến tính.
Ở mỗi lần lặp huấn luyện, ông đã thêm nhiễu
được lấy mẫu từ một phân phối có giá trị trung bình bằng không
$\epsilon \sim \mathcal{N}(0,\sigma^2)$ vào đầu vào $\mathbf{x}$,
tạo ra điểm nhiễu loạn $\mathbf{x}' = \mathbf{x} + \epsilon$.
Theo kỳ vọng, $E[\mathbf{x}'] = \mathbf{x}$.

Trong chuẩn hóa dropout tiêu chuẩn,
người ta đưa về không một phần các nút trong mỗi lớp
rồi *khử thiên lệch* mỗi lớp bằng cách chuẩn hóa
theo tỷ lệ các nút được giữ lại (không bị dropout).
Nói cách khác,
với *xác suất dropout* $p$,
mỗi kích hoạt trung gian $h$ được thay thế bởi
một biến ngẫu nhiên $h'$ như sau:

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \textrm{ với xác suất } p \\
    \frac{h}{1-p} & \textrm{ ngược lại}
\end{cases}
\end{aligned}
$$

Theo thiết kế, kỳ vọng vẫn không đổi, tức là $E[h'] = h$.


```python
from d2l import torch as d2l
import torch
from torch import nn
```


## Dropout trong Thực tế

Hãy nhớ lại MLP với một lớp ẩn và năm đơn vị ẩn
từ [fig_mlp](#fig_mlp).
Khi chúng ta áp dụng dropout cho một lớp ẩn,
đưa về không mỗi đơn vị ẩn với xác suất $p$,
kết quả có thể được xem như một mạng
chỉ chứa một tập con của các nơ-ron ban đầu.
Trong [fig_dropout2](#fig_dropout2), $h_2$ và $h_5$ bị loại bỏ.
Do đó, tính toán đầu ra
không còn phụ thuộc vào $h_2$ hoặc $h_5$
và gradient tương ứng của chúng cũng tiêu biến
khi thực hiện lan truyền ngược.
Theo cách này, tính toán lớp đầu ra
không thể phụ thuộc quá nhiều vào bất kỳ
một phần tử nào trong $h_1, \ldots, h_5$.

![MLP trước và sau dropout.](../img/dropout2.svg)
<a id="fig_dropout2"></a>

Thông thường, chúng ta tắt dropout trong thời gian kiểm tra.
Cho một mô hình đã được huấn luyện và một ví dụ mới,
chúng ta không loại bỏ bất kỳ nút nào
và do đó không cần chuẩn hóa.
Tuy nhiên, có một số ngoại lệ:
một số nhà nghiên cứu sử dụng dropout trong thời gian kiểm tra như một heuristic
để ước tính *sự không chắc chắn* của các dự đoán mạng nơ-ron:
nếu các dự đoán đồng thuận trên nhiều đầu ra dropout khác nhau,
thì chúng ta có thể nói rằng mạng tự tin hơn.

## Cài đặt từ Đầu

Để cài đặt hàm dropout cho một lớp duy nhất,
chúng ta phải lấy mẫu bao nhiêu mẫu
từ một biến ngẫu nhiên Bernoulli (nhị phân)
bằng số chiều của lớp,
trong đó biến ngẫu nhiên nhận giá trị $1$ (giữ)
với xác suất $1-p$ và $0$ (loại) với xác suất $p$.
Một cách dễ dàng để cài đặt điều này là lấy mẫu trước
từ phân phối đều $U[0, 1]$.
Sau đó chúng ta có thể giữ lại những nút mà mẫu tương ứng
lớn hơn $p$, loại bỏ phần còn lại.

Trong đoạn code sau, chúng ta (**cài đặt hàm `dropout_layer`
loại bỏ các phần tử trong tensor đầu vào `X`
với xác suất `dropout`**),
và tái tỉ lệ phần còn lại như mô tả ở trên:
chia những phần tử sống sót cho `1.0-dropout`.


```python
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```


Chúng ta có thể [**kiểm tra hàm `dropout_layer` trên một vài ví dụ**].
Trong các dòng code sau,
chúng ta truyền đầu vào `X` qua phép toán dropout,
với xác suất lần lượt là 0, 0.5, và 1.

```python
if tab.selected('mxnet'):
    X = np.arange(16).reshape(2, 8)
if tab.selected('pytorch'):
    X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
if tab.selected('tensorflow'):
    X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
if tab.selected('jax'):
    X = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
```

### Định nghĩa Mô hình

Mô hình bên dưới áp dụng dropout cho đầu ra
của mỗi lớp ẩn (sau hàm kích hoạt).
Chúng ta có thể đặt xác suất dropout cho mỗi lớp riêng biệt.
Một lựa chọn phổ biến là đặt
xác suất dropout thấp hơn gần lớp đầu vào.
Chúng ta đảm bảo rằng dropout chỉ hoạt động trong quá trình huấn luyện.


```python
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:  
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```


### [**Huấn luyện**]

Phần sau tương tự như việc huấn luyện MLP được mô tả trước đó.

```python
hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## [**Cài đặt Gọn**]

Với các API cấp cao, tất cả những gì chúng ta cần làm là thêm một lớp `Dropout`
sau mỗi lớp kết nối đầy đủ,
truyền vào xác suất dropout
như đối số duy nhất cho hàm khởi tạo của nó.
Trong quá trình huấn luyện, lớp `Dropout` sẽ ngẫu nhiên
loại bỏ đầu ra của lớp trước
(hoặc tương đương, đầu vào của lớp tiếp theo)
theo xác suất dropout đã chỉ định.
Khi không ở chế độ huấn luyện,
lớp `Dropout` đơn giản truyền dữ liệu qua trong quá trình kiểm tra.


```python
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(), 
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(), 
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))
```


Tiếp theo, chúng ta [**huấn luyện mô hình**].

```python
model = DropoutMLP(**hparams)
trainer.fit(model, data)
```

## Tóm tắt

Ngoài việc kiểm soát số chiều và kích thước vectơ trọng số, dropout là một công cụ khác để tránh quá khớp. Các công cụ thường được sử dụng kết hợp.
Lưu ý rằng dropout chỉ
được sử dụng trong quá trình huấn luyện:
nó thay thế một kích hoạt $h$ bằng một biến ngẫu nhiên có giá trị kỳ vọng $h$.


## Bài tập

1. Điều gì xảy ra nếu bạn thay đổi xác suất dropout cho lớp đầu tiên và thứ hai? Cụ thể, điều gì xảy ra nếu bạn hoán đổi xác suất của cả hai lớp? Thiết kế một thí nghiệm để trả lời những câu hỏi này, mô tả kết quả của bạn theo cách định lượng, và tóm tắt những rút ra định tính.
1. Tăng số epoch và so sánh kết quả thu được khi sử dụng dropout với kết quả khi không sử dụng nó.
1. Phương sai của các kích hoạt trong mỗi lớp ẩn khi dropout được và không được áp dụng là bao nhiêu? Vẽ một biểu đồ để cho thấy đại lượng này tiến hóa theo thời gian như thế nào đối với cả hai mô hình.
1. Tại sao dropout thường không được sử dụng trong thời gian kiểm tra?
1. Sử dụng mô hình trong phần này làm ví dụ, so sánh tác động của việc sử dụng dropout và suy giảm trọng số. Điều gì xảy ra khi dropout và suy giảm trọng số được sử dụng cùng một lúc? Các kết quả có cộng hưởng không? Có lợi nhuận giảm dần (hoặc tệ hơn) không? Chúng có triệt tiêu nhau không?
1. Điều gì xảy ra nếu chúng ta áp dụng dropout cho các trọng số riêng lẻ của ma trận trọng số thay vì các kích hoạt?
1. Phát minh một kỹ thuật khác để đưa nhiễu ngẫu nhiên vào mỗi lớp khác với kỹ thuật dropout tiêu chuẩn. Bạn có thể phát triển một phương pháp vượt trội hơn dropout trên tập dữ liệu Fashion-MNIST không (với kiến trúc cố định)?


[Discussions](https://discuss.d2l.ai/t/101)
