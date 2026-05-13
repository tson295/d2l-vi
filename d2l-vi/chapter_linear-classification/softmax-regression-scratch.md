# Cài đặt Hồi quy Softmax từ Đầu
<a id="sec_softmax_scratch"></a>

Vì hồi quy softmax là nền tảng căn bản,
chúng ta tin rằng bạn nên biết
cách tự cài đặt nó.
Ở đây, chúng ta giới hạn bản thân trong việc định nghĩa các
khía cạnh đặc thù của softmax trong mô hình
và tái sử dụng các thành phần khác
từ phần hồi quy tuyến tính,
bao gồm cả vòng lặp huấn luyện.


```python
from d2l import torch as d2l
import torch
```


## Hàm Softmax

Hãy bắt đầu với phần quan trọng nhất:
ánh xạ từ số vô hướng đến xác suất.
Để nhắc lại, hãy nhớ lại phép toán tổng
theo các chiều cụ thể trong tensor,
như đã thảo luận trong [subsec_lin-alg-reduction](#subsec_lin-alg-reduction)
và [subsec_lin-alg-non-reduction](#subsec_lin-alg-non-reduction).
[**Cho một ma trận `X`, chúng ta có thể tính tổng tất cả phần tử (mặc định) hoặc chỉ
các phần tử trong cùng một trục.**]
Biến `axis` cho phép chúng ta tính tổng theo hàng và cột:

```python
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

Tính toán softmax yêu cầu ba bước:
(i) lũy thừa mỗi phần tử;
(ii) tổng theo mỗi hàng để tính hằng số chuẩn hóa cho mỗi mẫu;
(iii) chia mỗi hàng cho hằng số chuẩn hóa của nó,
đảm bảo rằng kết quả có tổng bằng 1:

(**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**)

(Logarit của) mẫu số
được gọi là (log) *hàm phân hoạch*.
Nó được giới thiệu trong [vật lý thống kê](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))
để tính tổng trên tất cả các trạng thái có thể trong một tập hợp nhiệt động học.
Cài đặt rất đơn giản:

```python
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

Với bất kỳ đầu vào `X` nào, [**chúng ta biến mỗi phần tử
thành một số không âm.
Mỗi hàng có tổng bằng 1,**]
như yêu cầu cho xác suất. Lưu ý: code ở trên *không* bền vững với các đối số rất lớn hoặc rất nhỏ. Mặc dù đủ để minh họa những gì đang xảy ra, bạn *không nên* sử dụng code này nguyên văn cho bất kỳ mục đích nghiêm túc nào. Các framework deep learning có sẵn các biện pháp bảo vệ như vậy và chúng ta sẽ sử dụng softmax tích hợp từ đây trở đi.


## Mô hình

Bây giờ chúng ta có đầy đủ mọi thứ cần thiết
để cài đặt [**mô hình hồi quy softmax.**]
Như trong ví dụ hồi quy tuyến tính,
mỗi mẫu sẽ được biểu diễn
bằng một vector có độ dài cố định.
Vì dữ liệu thô ở đây bao gồm
các ảnh pixel $28 \times 28$,
[**chúng ta làm phẳng mỗi ảnh,
coi chúng là vector có độ dài 784.**]
Trong các chương sau, chúng ta sẽ giới thiệu
mạng nơ-ron tích chập,
khai thác cấu trúc không gian
theo cách thỏa mãn hơn.


Trong hồi quy softmax,
số đầu ra từ mạng của chúng ta
phải bằng số lớp.
(**Vì bộ dữ liệu của chúng ta có 10 lớp,
mạng của chúng ta có chiều đầu ra là 10.**)
Do đó, trọng số của chúng ta tạo thành ma trận $784 \times 10$
cộng với vector hàng $1 \times 10$ cho các hệ số chặn.
Như với hồi quy tuyến tính,
chúng ta khởi tạo trọng số `W`
với nhiễu Gaussian.
Các hệ số chặn được khởi tạo bằng không.


```python
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
```


Code bên dưới định nghĩa cách mạng
ánh xạ mỗi đầu vào sang đầu ra.
Lưu ý rằng chúng ta làm phẳng mỗi ảnh pixel $28 \times 28$ trong batch
thành vector bằng `reshape`
trước khi truyền dữ liệu qua mô hình.

```python
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.W.shape[0]))
    return softmax(d2l.matmul(X, self.W) + self.b)
```

## Hàm Mất mát Cross-Entropy

Tiếp theo chúng ta cần cài đặt hàm mất mát cross-entropy
(được giới thiệu trong [subsec_softmax-regression-loss-func](#subsec_softmax-regression-loss-func)).
Đây có thể là hàm mất mát phổ biến nhất
trong tất cả deep learning.
Hiện tại, các ứng dụng deep learning
dễ dàng được đặt thành bài toán phân loại
vượt xa số lượng những bài tốt hơn nên xử lý như bài toán hồi quy.

Nhớ lại rằng cross-entropy lấy log-likelihood âm
của xác suất dự đoán được gán cho nhãn thực.
Để hiệu quả, chúng ta tránh vòng lặp for của Python và sử dụng chỉ số thay thế.
Đặc biệt, mã hóa one-hot trong $\mathbf{y}$
cho phép chúng ta chọn các phần tử tương ứng trong $\hat{\mathbf{y}}$.

Để thấy điều này trong thực tế, chúng ta [**tạo dữ liệu mẫu `y_hat`
với 2 mẫu xác suất dự đoán trên 3 lớp và nhãn tương ứng `y`.**]
Nhãn đúng lần lượt là $0$ và $2$ (tức là lớp thứ nhất và thứ ba).
[**Sử dụng `y` làm chỉ số của xác suất trong `y_hat`,**]
chúng ta có thể chọn các phần tử một cách hiệu quả.


## Huấn luyện

Chúng ta tái sử dụng phương thức `fit` đã định nghĩa trong [sec_linear_scratch](#sec_linear_scratch) để [**huấn luyện mô hình với 10 epoch.**]
Lưu ý rằng số epoch (`max_epochs`),
kích thước minibatch (`batch_size`),
và tốc độ học (`lr`)
là các siêu tham số có thể điều chỉnh.
Điều đó có nghĩa là trong khi các giá trị này không được
học trong vòng lặp huấn luyện chính,
chúng vẫn ảnh hưởng đến hiệu suất
của mô hình, cả về hiệu suất huấn luyện
lẫn hiệu suất tổng quát hóa.
Trong thực tế, bạn sẽ muốn chọn các giá trị này
dựa trên phần *kiểm định* của dữ liệu
và sau đó, cuối cùng, đánh giá mô hình cuối cùng của bạn
trên phần *kiểm tra*.
Như đã thảo luận trong [subsec_generalization-model-selection](#subsec_generalization-model-selection),
chúng ta sẽ coi dữ liệu kiểm tra của Fashion-MNIST
là tập kiểm định, do đó
báo cáo mất mát kiểm định và độ chính xác kiểm định
trên phần này.

```python
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Dự đoán

Bây giờ huấn luyện đã hoàn tất,
mô hình của chúng ta đã sẵn sàng để [**phân loại một số ảnh.**]

```python
X, y = next(iter(data.val_dataloader()))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = d2l.argmax(model(X), axis=1)
if tab.selected('jax'):
    preds = d2l.argmax(model.apply({'params': trainer.state.params}, X), axis=1)
preds.shape
```

Chúng ta quan tâm hơn đến các ảnh mà chúng ta gán nhãn *sai*. Chúng ta trực quan hóa chúng bằng cách
so sánh nhãn thực của chúng
(dòng văn bản đầu tiên)
với dự đoán từ mô hình
(dòng văn bản thứ hai).

```python
wrong = d2l.astype(preds, y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
```

## Tóm tắt

Đến đây chúng ta đang bắt đầu có được một số kinh nghiệm
trong việc giải quyết các bài toán hồi quy tuyến tính
và phân loại.
Với đó, chúng ta đã đạt đến điều có thể được coi là
trình độ nghệ thuật của mô hình thống kê thập niên 1960--1970.
Trong phần tiếp theo, chúng ta sẽ chỉ cho bạn cách tận dụng
các framework deep learning để cài đặt mô hình này
hiệu quả hơn nhiều.

## Bài tập

1. Trong phần này, chúng ta đã trực tiếp cài đặt hàm softmax dựa trên định nghĩa toán học của phép toán softmax. Như đã thảo luận trong [sec_softmax](#sec_softmax) điều này có thể gây ra bất ổn số.
    1. Kiểm tra xem `softmax` có còn hoạt động đúng không nếu đầu vào có giá trị $100$.
    1. Kiểm tra xem `softmax` có còn hoạt động đúng không nếu giá trị lớn nhất của tất cả đầu vào nhỏ hơn $-100$.
    1. Cài đặt một bản sửa lỗi bằng cách xem xét giá trị so với phần tử lớn nhất trong đối số.
1. Cài đặt hàm `cross_entropy` theo định nghĩa của hàm mất mát cross-entropy $\sum_i y_i \log \hat{y}_i$.
    1. Thử trong ví dụ code của phần này.
    1. Tại sao bạn nghĩ nó chạy chậm hơn?
    1. Bạn có nên sử dụng nó không? Khi nào thì có ý nghĩa?
    1. Bạn cần cẩn thận điều gì? Gợi ý: hãy xem miền của logarit.
1. Việc luôn trả về nhãn có khả năng cao nhất có phải là ý tưởng tốt không? Ví dụ, bạn có làm điều này cho chẩn đoán y tế không? Bạn sẽ cố gắng giải quyết điều này như thế nào?
1. Giả sử rằng chúng ta muốn sử dụng hồi quy softmax để dự đoán từ tiếp theo dựa trên một số đặc trưng. Những vấn đề nào có thể phát sinh từ từ vựng lớn?
1. Thử nghiệm với các siêu tham số của code trong phần này. Cụ thể:
    1. Vẽ đồ thị mất mát kiểm định thay đổi như thế nào khi bạn thay đổi tốc độ học.
    1. Mất mát kiểm định và huấn luyện có thay đổi khi bạn thay đổi kích thước minibatch không? Bạn cần tăng hoặc giảm bao nhiêu trước khi thấy hiệu ứng?


[Discussions](https://discuss.d2l.ai/t/51)
