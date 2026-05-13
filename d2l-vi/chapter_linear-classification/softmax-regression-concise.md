# Cài đặt Gọn của Hồi quy Softmax
<a id="sec_softmax_concise"></a>


Cũng như các framework deep learning cấp cao
đã giúp cài đặt hồi quy tuyến tính dễ dàng hơn
(xem [sec_linear_concise](#sec_linear_concise)),
chúng cũng tiện lợi tương tự ở đây.


```python
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```


## Định nghĩa Mô hình

Như trong [sec_linear_concise](#sec_linear_concise),
chúng ta xây dựng lớp kết nối đầy đủ
bằng lớp tích hợp sẵn.
Phương thức `__call__` tích hợp sau đó gọi `forward`
mỗi khi chúng ta cần áp dụng mạng vào một đầu vào nào đó.


Chúng ta dùng lớp `Flatten` để chuyển tensor bậc bốn `X` sang bậc hai
bằng cách giữ nguyên chiều theo trục đầu tiên.


```python
class SoftmaxRegression(d2l.Classifier):  
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
```


## Xem Lại Softmax
<a id="subsec_softmax-implementation-revisited"></a>

Trong [sec_softmax_scratch](#sec_softmax_scratch) chúng ta đã tính đầu ra của mô hình
và áp dụng hàm mất mát cross-entropy. Mặc dù điều này hoàn toàn
hợp lý về mặt toán học, nhưng nó có rủi ro về mặt tính toán, do
tràn số dưới và tràn số trên trong phép lũy thừa.

Nhớ lại rằng hàm softmax tính xác suất qua
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$.
Nếu một số $o_k$ rất lớn, tức là rất dương,
thì $\exp(o_k)$ có thể lớn hơn số lớn nhất
mà chúng ta có thể có với một số kiểu dữ liệu nhất định. Đây gọi là *tràn số trên*. Tương tự,
nếu mọi đối số là số âm rất lớn, chúng ta sẽ gặp *tràn số dưới*.
Ví dụ, số thực dấu phẩy động độ chính xác đơn xấp xỉ
bao gồm phạm vi từ $10^{-38}$ đến $10^{38}$. Do đó, nếu phần tử lớn nhất trong $\mathbf{o}$
nằm ngoài khoảng $[-90, 90]$, kết quả sẽ không ổn định.
Một cách giải quyết vấn đề này là trừ $\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$ từ
tất cả các phần tử:

$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

Theo cấu trúc, chúng ta biết rằng $o_j - \bar{o} \leq 0$ với mọi $j$. Vì vậy, với bài toán phân loại $q$ lớp, mẫu số nằm trong khoảng $[1, q]$. Hơn nữa, tử số không bao giờ vượt quá $1$, do đó ngăn ngừa tràn số trên. Tràn số dưới chỉ xảy ra khi $\exp(o_j - \bar{o})$ được tính số học cho ra $0$. Tuy vậy, một vài bước tiếp theo chúng ta có thể gặp rắc rối khi muốn tính $\log \hat{y}_j$ thành $\log 0$.
Đặc biệt, trong lan truyền ngược,
chúng ta có thể đối mặt với đầy màn hình
kết quả `NaN` (Not a Number) đáng sợ.

May mắn thay, chúng ta được cứu bởi thực tế là
mặc dù chúng ta đang tính các hàm lũy thừa,
nhưng cuối cùng chúng ta có ý định lấy logarit của chúng
(khi tính hàm mất mát cross-entropy).
Bằng cách kết hợp softmax và cross-entropy,
chúng ta có thể thoát khỏi các vấn đề ổn định số hoàn toàn. Chúng ta có:

$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

Điều này tránh cả tràn số trên và tràn số dưới.
Chúng ta sẽ muốn giữ hàm softmax thông thường tiện dụng
trong trường hợp chúng ta muốn đánh giá xác suất đầu ra của mô hình.
Nhưng thay vì truyền xác suất softmax vào hàm mất mát mới,
chúng ta chỉ
[**truyền logit và tính softmax cùng logarit của nó
tất cả cùng một lúc bên trong hàm mất mát cross-entropy,**]
thực hiện những điều thông minh như ["thủ thuật LogSumExp"](https://en.wikipedia.org/wiki/LogSumExp).


## Huấn luyện

Tiếp theo chúng ta huấn luyện mô hình. Chúng ta sử dụng ảnh Fashion-MNIST, làm phẳng thành vector đặc trưng 784 chiều.

```python
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

Như trước đây, thuật toán hội tụ về một giải pháp
khá chính xác,
mặc dù lần này với ít dòng code hơn trước.


## Tóm tắt

API cấp cao rất tiện lợi khi ẩn đi những khía cạnh nguy hiểm tiềm ẩn khỏi người dùng, chẳng hạn như ổn định số. Hơn nữa, chúng cho phép người dùng thiết kế mô hình gọn gàng chỉ với rất ít dòng code. Điều này vừa là lợi ích vừa là bất lợi. Lợi ích rõ ràng là làm cho mọi thứ dễ tiếp cận hơn, ngay cả với các kỹ sư chưa từng học một lớp thống kê nào trong cuộc sống (thực ra, họ là một phần của đối tượng mục tiêu của cuốn sách). Nhưng việc che giấu các phần sắc bén cũng đi kèm với một cái giá: không có động lực để tự thêm các thành phần mới và khác nhau, vì có ít cơ bắp ghi nhớ để làm điều đó. Hơn nữa, nó khiến việc *sửa chữa* mọi thứ trở nên khó khăn hơn khi lớp đệm bảo vệ của một framework không bao phủ hoàn toàn tất cả các trường hợp cạnh. Một lần nữa, điều này là do thiếu quen thuộc.

Do đó, chúng tôi thúc giục bạn xem xét *cả* phiên bản thô sơ lẫn phiên bản thanh lịch của nhiều cài đặt tiếp theo. Trong khi chúng tôi nhấn mạnh sự dễ hiểu, các cài đặt vẫn thường khá hiệu quả (tích chập là ngoại lệ lớn ở đây). Chúng tôi có ý định cho phép bạn xây dựng dựa trên những điều này khi bạn phát minh ra điều gì đó mới mà không có framework nào có thể cung cấp cho bạn.


## Bài tập

1. Deep learning sử dụng nhiều định dạng số khác nhau, bao gồm FP64 độ chính xác kép (hiếm khi dùng),
FP32 độ chính xác đơn, BFLOAT16 (tốt cho biểu diễn nén), FP16 (rất không ổn định), TF32 (định dạng mới từ NVIDIA), và INT8. Tính đối số nhỏ nhất và lớn nhất của hàm lũy thừa mà kết quả không dẫn đến tràn số dưới hoặc tràn số trên.
1. INT8 là định dạng rất hạn chế bao gồm các số khác không từ $1$ đến $255$. Làm thế nào bạn có thể mở rộng phạm vi động của nó mà không dùng thêm bit? Phép nhân và cộng chuẩn có vẫn hoạt động không?
1. Tăng số epoch để huấn luyện. Tại sao độ chính xác kiểm định có thể giảm sau một thời gian? Chúng ta có thể khắc phục điều này như thế nào?
1. Điều gì xảy ra khi bạn tăng tốc độ học? So sánh đường cong mất mát với nhiều tốc độ học khác nhau. Cái nào hoạt động tốt hơn? Khi nào?


[Discussions](https://discuss.d2l.ai/t/53)
