# Lớp Tùy chỉnh

Một yếu tố đằng sau thành công của deep learning
là sự sẵn có của nhiều loại lớp
có thể được kết hợp theo những cách sáng tạo
để thiết kế các kiến trúc phù hợp
với nhiều loại tác vụ.
Ví dụ, các nhà nghiên cứu đã phát minh các lớp
đặc biệt để xử lý hình ảnh, văn bản,
lặp qua dữ liệu tuần tự,
và
thực hiện lập trình động.
Sớm hay muộn, bạn sẽ cần
một lớp chưa tồn tại trong framework deep learning.
Trong những trường hợp này, bạn phải xây dựng một lớp tùy chỉnh.
Trong phần này, chúng ta chỉ cho bạn cách làm điều đó.


```python
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```


## (**Lớp không có Tham số**)

Để bắt đầu, chúng ta xây dựng một lớp tùy chỉnh
không có bất kỳ tham số riêng nào.
Điều này trông quen thuộc nếu bạn nhớ lại
phần giới thiệu về mô-đun của chúng ta trong [sec_model_construction](#sec_model_construction).
Lớp `CenteredLayer` sau đây chỉ đơn giản
trừ giá trị trung bình khỏi đầu vào của nó.
Để xây dựng nó, chúng ta chỉ cần kế thừa
từ lớp cơ sở và cài đặt hàm lan truyền xuôi.


```python
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```


Hãy xác minh rằng lớp của chúng ta hoạt động như dự kiến bằng cách truyền một số dữ liệu qua nó.

```python
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

Bây giờ chúng ta có thể [**kết hợp lớp của chúng ta như một thành phần
trong việc xây dựng các mô hình phức tạp hơn.**]


```python
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```


Như một kiểm tra sự tỉnh táo thêm, chúng ta có thể gửi dữ liệu ngẫu nhiên
qua mạng và kiểm tra rằng giá trị trung bình thực sự là 0.
Vì chúng ta đang xử lý các số dấu phẩy động,
chúng ta vẫn có thể thấy một số rất nhỏ khác không
do lượng tử hóa.


## [**Lớp có Tham số**]

Bây giờ khi chúng ta biết cách định nghĩa các lớp đơn giản,
hãy chuyển sang định nghĩa các lớp có tham số
có thể được điều chỉnh thông qua huấn luyện.
Chúng ta có thể sử dụng các hàm tích hợp để tạo tham số, cung cấp
một số chức năng dọn dẹp cơ bản.
Cụ thể, chúng quản lý quyền truy cập, khởi tạo,
chia sẻ, lưu, và tải các tham số mô hình.
Theo cách này, trong số những lợi ích khác, chúng ta sẽ không cần viết
các quy trình tuần tự hóa tùy chỉnh cho mỗi lớp tùy chỉnh.

Bây giờ hãy cài đặt phiên bản riêng của lớp kết nối đầy đủ.
Hãy nhớ lại rằng lớp này yêu cầu hai tham số,
một để đại diện cho trọng số và cái kia cho hệ số chặn.
Trong cài đặt này, chúng ta tích hợp kích hoạt ReLU như mặc định.
Lớp này yêu cầu hai đối số đầu vào: `in_units` và `units`, tương ứng
biểu thị số đầu vào và đầu ra.


```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```


Tiếp theo, chúng ta khởi tạo lớp `MyLinear`
và truy cập các tham số mô hình của nó.


```python
linear = MyLinear(5, 3)
linear.weight
```


Chúng ta có thể [**trực tiếp thực hiện các tính toán lan truyền xuôi bằng các lớp tùy chỉnh.**]


```python
linear(torch.rand(2, 5))
```


Chúng ta cũng có thể (**xây dựng mô hình bằng các lớp tùy chỉnh.**)
Một khi có, chúng ta có thể sử dụng nó giống như lớp kết nối đầy đủ tích hợp sẵn.


```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```


## Tóm tắt

Chúng ta có thể thiết kế các lớp tùy chỉnh thông qua lớp cơ sở. Điều này cho phép chúng ta định nghĩa các lớp mới linh hoạt hoạt động khác với bất kỳ lớp hiện có nào trong thư viện.
Một khi được định nghĩa, các lớp tùy chỉnh có thể được gọi trong các bối cảnh và kiến trúc tùy ý.
Các lớp có thể có các tham số cục bộ, có thể được tạo thông qua các hàm tích hợp sẵn.


## Bài tập

1. Thiết kế một lớp nhận đầu vào và tính toán phép rút gọn tensor,
   tức là nó trả về $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Thiết kế một lớp trả về nửa đầu các hệ số Fourier của dữ liệu.


[Discussions](https://discuss.d2l.ai/t/59)
