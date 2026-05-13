# Cài đặt Gọn của Hồi quy Tuyến tính
<a id="sec_linear_concise"></a>

Deep learning đã chứng kiến một dạng bùng nổ Cambri
trong thập kỷ vừa qua.
Số lượng kỹ thuật, ứng dụng và thuật toán vượt xa
tiến bộ của các thập kỷ trước.
Điều này là do sự kết hợp may mắn của nhiều yếu tố,
một trong số đó là các công cụ miễn phí mạnh mẽ
được cung cấp bởi một số framework deep learning mã nguồn mở.
Theano [Bergstra.Breuleux.Bastien.ea.2010],
DistBelief [Dean.Corrado.Monga.ea.2012],
và Caffe [Jia.Shelhamer.Donahue.ea.2014]
được coi là đại diện cho
thế hệ đầu tiên của các mô hình như vậy
được áp dụng rộng rãi.
Trái ngược với các công trình trước đó (tiên phong) như
SN2 (Simulateur Neuristique) [Bottou.Le-Cun.1988],
cung cấp trải nghiệm lập trình giống Lisp,
các framework hiện đại cung cấp vi phân tự động
và sự tiện lợi của Python.
Các framework này cho phép ta tự động hóa và mô-đun hóa
công việc lặp đi lặp lại khi cài đặt các thuật toán học dựa trên gradient.

Trong [sec_linear_scratch](#sec_linear_scratch), ta chỉ dựa vào
(i) tensor để lưu trữ dữ liệu và đại số tuyến tính;
và (ii) vi phân tự động để tính gradient.
Trong thực tế, vì các bộ lặp dữ liệu, hàm mất mát, bộ tối ưu hóa,
và các lớp mạng nơ-ron
rất phổ biến, các thư viện hiện đại cũng cài đặt các thành phần này cho ta.
Trong phần này, (**ta sẽ chỉ cho bạn cách cài đặt
mô hình hồi quy tuyến tính**) từ [sec_linear_scratch](#sec_linear_scratch)
(**một cách gọn gàng bằng cách sử dụng API cấp cao**) của các framework deep learning.


```python
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```


## Định nghĩa Mô hình

Khi ta cài đặt hồi quy tuyến tính từ đầu
trong [sec_linear_scratch](#sec_linear_scratch),
ta đã định nghĩa rõ ràng các tham số mô hình
và viết code tính toán để tạo ra đầu ra
bằng các phép toán đại số tuyến tính cơ bản.
Bạn *nên* biết cách làm điều này.
Nhưng khi mô hình của bạn trở nên phức tạp hơn,
và khi bạn phải làm điều này hầu như mỗi ngày,
bạn sẽ vui mừng vì có sự hỗ trợ.
Tình huống này giống như việc tự viết blog của riêng mình từ đầu.
Làm một hoặc hai lần thì bổ ích và giáo dục,
nhưng bạn sẽ là một nhà phát triển web tệ
nếu bạn dành cả tháng để tái phát minh bánh xe.

Với các phép toán tiêu chuẩn,
ta có thể [**dùng các lớp được định nghĩa sẵn của framework,**]
cho phép ta tập trung
vào các lớp được dùng để xây dựng mô hình
thay vì lo lắng về cài đặt của chúng.
Hãy nhớ lại kiến trúc của mạng một lớp
như được mô tả trong [fig_single_neuron](#fig_single_neuron).
Lớp được gọi là *kết nối đầy đủ*,
vì mỗi đầu vào của nó được kết nối
với mỗi đầu ra của nó
thông qua phép nhân ma trận--vector.


Trong PyTorch, lớp kết nối đầy đủ được định nghĩa trong các lớp `Linear` và `LazyLinear` (có sẵn từ phiên bản 1.8.0).
Lớp sau
cho phép người dùng chỉ định *chỉ*
chiều đầu ra,
trong khi lớp trước
ngoài ra còn yêu cầu
bao nhiêu đầu vào đi vào lớp này.
Chỉ định shape đầu vào thật bất tiện và có thể đòi hỏi tính toán không đơn giản
(chẳng hạn như trong các lớp tích chập).
Do đó, để đơn giản, ta sẽ sử dụng các lớp "lười" như vậy
bất cứ khi nào có thể.


Trong phương thức `forward` ta chỉ gọi phương thức `__call__` tích hợp của các lớp được định nghĩa sẵn để tính đầu ra.


## Định nghĩa Hàm Mất mát


[**Lớp `MSELoss` tính sai số bình phương trung bình (không có hệ số $1/2$ trong :eqref:`eq_mse`).**]
Theo mặc định, `MSELoss` trả về mất mát trung bình trên các mẫu.
Nó nhanh hơn (và dễ dùng hơn) so với tự cài đặt.


## Định nghĩa Thuật toán Tối ưu hóa


Minibatch SGD là công cụ tiêu chuẩn
để tối ưu hóa các mạng nơ-ron
và do đó PyTorch hỗ trợ nó cùng với một số
biến thể của thuật toán này trong module `optim`.
Khi ta (**khởi tạo một thực thể `SGD`,**)
ta chỉ định các tham số cần tối ưu hóa,
có thể lấy từ mô hình qua `self.parameters()`,
và tốc độ học (`self.lr`)
được yêu cầu bởi thuật toán tối ưu hóa.


```python
@d2l.add_to_class(LinearRegression)  
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
        return tf.keras.optimizers.SGD(self.lr)
    if tab.selected('jax'):
        return optax.sgd(self.lr)
```

## Huấn luyện

Bạn có thể đã nhận thấy rằng việc biểu diễn mô hình thông qua
API cấp cao của framework deep learning
đòi hỏi ít dòng code hơn.
Ta không phải phân bổ tham số riêng lẻ,
định nghĩa hàm mất mát, hay cài đặt minibatch SGD.
Khi ta bắt đầu làm việc với các mô hình phức tạp hơn nhiều,
ưu điểm của API cấp cao sẽ tăng lên đáng kể.

Bây giờ ta đã có tất cả các thành phần cơ bản,
[**vòng lặp huấn luyện giống hệt
vòng lặp ta đã cài đặt từ đầu.**]
Vì vậy ta chỉ gọi phương thức `fit` (giới thiệu trong [oo-design-training](#oo-design-training)),
dựa vào cài đặt của phương thức `fit_epoch`
trong [sec_linear_scratch](#sec_linear_scratch),
để huấn luyện mô hình.

```python
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Dưới đây, ta
[**so sánh các tham số mô hình được học
bằng cách huấn luyện trên dữ liệu hữu hạn
với các tham số thực tế**]
đã tạo ra tập dữ liệu.
Để truy cập tham số,
ta truy cập trọng số và hệ số chặn
của lớp mà ta cần.
Như trong cài đặt từ đầu của ta,
lưu ý rằng các tham số ước lượng của ta
gần với các giá trị thực tương ứng.


```python
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## Tóm tắt

Phần này chứa cài đặt đầu tiên
của một mạng sâu (trong cuốn sách này)
để tận dụng những tiện lợi
được cung cấp bởi các framework deep learning hiện đại,
chẳng hạn như MXNet [Chen.Li.Li.ea.2015],
JAX [Frostig.Johnson.Leary.2018],
PyTorch [Paszke.Gross.Massa.ea.2019],
và Tensorflow [Abadi.Barham.Chen.ea.2016].
Ta đã dùng các giá trị mặc định của framework để tải dữ liệu, định nghĩa một lớp,
một hàm mất mát, một bộ tối ưu hóa và một vòng lặp huấn luyện.
Bất cứ khi nào framework cung cấp tất cả các tính năng cần thiết,
thường là ý tưởng hay khi dùng chúng,
vì các cài đặt thư viện của các thành phần này
thường được tối ưu hóa cao cho hiệu suất
và được kiểm tra đúng cách về độ tin cậy.
Đồng thời, cố gắng đừng quên
rằng các module này *có thể* được cài đặt trực tiếp.
Điều này đặc biệt quan trọng cho các nhà nghiên cứu có tham vọng
muốn sống ở tuyến đầu của phát triển mô hình,
nơi bạn sẽ phát minh ra các thành phần mới
mà không thể tồn tại trong bất kỳ thư viện hiện tại nào.


Trong PyTorch, module `data` cung cấp công cụ xử lý dữ liệu,
module `nn` định nghĩa nhiều lớp mạng nơ-ron và hàm mất mát phổ biến.
Ta có thể khởi tạo các tham số bằng cách thay thế các giá trị của chúng
bằng các phương thức kết thúc bằng `_`.
Lưu ý rằng ta cần chỉ định chiều đầu vào của mạng.
Mặc dù điều này đơn giản bây giờ, nó có thể có tác động đáng kể
khi ta muốn thiết kế các mạng phức tạp với nhiều lớp.
Cần có sự xem xét cẩn thận về cách tham số hóa các mạng này
để cho phép tính di động.


## Bài tập

1. Bạn cần thay đổi tốc độ học như thế nào nếu bạn thay thế mất mát tổng hợp trên minibatch
   bằng trung bình mất mát trên minibatch?
1. Xem lại tài liệu framework để xem các hàm mất mát nào được cung cấp. Cụ thể,
   thay thế mất mát bình phương bằng hàm mất mát mạnh mẽ Huber. Tức là, dùng hàm mất mát
   $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \textrm{ nếu } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \textrm{ ngược lại}\end{cases}$$
1. Làm thế nào để truy cập gradient của trọng số mô hình?
1. Ảnh hưởng đến nghiệm là gì nếu bạn thay đổi tốc độ học và số epoch? Nó có tiếp tục cải thiện không?
1. Nghiệm thay đổi như thế nào khi bạn thay đổi lượng dữ liệu được tạo ra?
    1. Vẽ sai số ước lượng cho $\hat{\mathbf{w}} - \mathbf{w}$ và $\hat{b} - b$ như là hàm của lượng dữ liệu. Gợi ý: tăng lượng dữ liệu theo logarit thay vì tuyến tính, tức là 5, 10, 20, 50, ..., 10.000 thay vì 1000, 2000, ..., 10.000.
    2. Tại sao gợi ý trong hint lại phù hợp?


[Thảo luận](https://discuss.d2l.ai/t/45)
