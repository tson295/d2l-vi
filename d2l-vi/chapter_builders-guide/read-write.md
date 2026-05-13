# Nhập/Xuất File

Cho đến nay chúng ta đã thảo luận về cách xử lý dữ liệu và cách
xây dựng, huấn luyện, và kiểm tra các mô hình deep learning.
Tuy nhiên, ở một thời điểm nào đó, chúng ta hy vọng sẽ đủ hài lòng
với các mô hình đã học để muốn
lưu kết quả để sử dụng sau trong các bối cảnh khác nhau
(thậm chí có thể để thực hiện dự đoán trong triển khai).
Ngoài ra, khi chạy một quá trình huấn luyện dài,
thực hành tốt nhất là định kỳ lưu các kết quả trung gian (checkpointing)
để đảm bảo chúng ta không mất vài ngày tính toán
nếu chúng ta vấp phải dây nguồn của máy chủ.
Do đó đã đến lúc học cách tải và lưu
cả các vectơ trọng số riêng lẻ và toàn bộ mô hình.
Phần này giải quyết cả hai vấn đề.


```python
import torch
from torch import nn
from torch.nn import functional as F
```


## (**Tải và Lưu Tensor**)

Đối với các tensor riêng lẻ, chúng ta có thể trực tiếp
gọi các hàm `load` và `save`
để đọc và ghi chúng tương ứng.
Cả hai hàm đều yêu cầu chúng ta cung cấp một tên,
và `save` yêu cầu biến cần lưu như đầu vào.


```python
x = torch.arange(4)
torch.save(x, 'x-file')
```


Bây giờ chúng ta có thể đọc dữ liệu từ file đã lưu trở lại vào bộ nhớ.


```python
x2 = torch.load('x-file')
x2
```


Chúng ta có thể [**lưu một danh sách tensor và đọc chúng trở lại vào bộ nhớ.**]


```python
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```


Chúng ta thậm chí có thể [**ghi và đọc một từ điển ánh xạ
từ chuỗi đến tensor.**]
Điều này tiện lợi khi chúng ta muốn
đọc hoặc ghi tất cả trọng số trong một mô hình.


```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```


## [**Tải và Lưu Tham số Mô hình**]

Lưu các vectơ trọng số riêng lẻ (hoặc các tensor khác) hữu ích,
nhưng nó trở nên rất tẻ nhạt nếu chúng ta muốn lưu
(và sau đó tải) một mô hình toàn bộ.
Xét cho cùng, chúng ta có thể có hàng trăm
nhóm tham số rải rác khắp nơi.
Vì lý do này, framework deep learning cung cấp các chức năng tích hợp sẵn
để tải và lưu toàn bộ mạng.
Một chi tiết quan trọng cần lưu ý là điều này
lưu *tham số* mô hình chứ không phải toàn bộ mô hình.
Ví dụ, nếu chúng ta có một MLP 3 lớp,
chúng ta cần chỉ định kiến trúc riêng biệt.
Lý do là các mô hình tự thân có thể chứa code tùy ý,
do đó chúng không thể được tuần tự hóa một cách tự nhiên.
Do đó, để khôi phục một mô hình, chúng ta cần
tạo kiến trúc trong code
và sau đó tải các tham số từ đĩa.
(**Hãy bắt đầu với MLP quen thuộc của chúng ta.**)


```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```


Tiếp theo, chúng ta [**lưu các tham số của mô hình như một file**] với tên "mlp.params".


```python
torch.save(net.state_dict(), 'mlp.params')
```


Để khôi phục mô hình, chúng ta khởi tạo một bản sao
của mô hình MLP ban đầu.
Thay vì khởi tạo ngẫu nhiên các tham số mô hình,
chúng ta [**đọc trực tiếp các tham số được lưu trong file**].


```python
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```


Vì cả hai thực thể có cùng tham số mô hình,
kết quả tính toán của cùng đầu vào `X` phải giống nhau.
Hãy xác minh điều này.


## Tóm tắt

Các hàm `save` và `load` có thể được sử dụng để thực hiện nhập/xuất file cho các đối tượng tensor.
Chúng ta có thể lưu và tải toàn bộ tập tham số cho một mạng thông qua từ điển tham số.
Việc lưu kiến trúc phải được thực hiện trong code thay vì trong tham số.

## Bài tập

1. Ngay cả khi không cần triển khai mô hình đã huấn luyện lên một thiết bị khác, những lợi ích thực tế của việc lưu trữ tham số mô hình là gì?
1. Giả sử chúng ta muốn tái sử dụng chỉ một phần của mạng để kết hợp vào một mạng có kiến trúc khác. Bạn sẽ làm thế nào để sử dụng, ví dụ, hai lớp đầu tiên từ mạng trước trong một mạng mới?
1. Bạn sẽ làm thế nào để lưu kiến trúc mạng và các tham số? Bạn sẽ áp đặt những hạn chế nào lên kiến trúc?


[Discussions](https://discuss.d2l.ai/t/61)
