# Tài liệu

Mặc dù chúng tôi không thể giới thiệu từng hàm và lớp PyTorch
(và thông tin có thể nhanh chóng lỗi thời),
[tài liệu API](https://pytorch.org/docs/stable/index.html) và các [hướng dẫn](https://pytorch.org/tutorials/beginner/basics/intro.html) và ví dụ bổ sung
cung cấp tài liệu như vậy.
Phần này cung cấp một số hướng dẫn về cách khám phá API PyTorch.


```python
import torch
```


## Hàm và Lớp trong Module

Để biết hàm và lớp nào có thể được gọi trong một module,
chúng ta sử dụng hàm `dir`. Ví dụ, chúng ta có thể
(**truy vấn tất cả các thuộc tính trong module để tạo số ngẫu nhiên**):


```python
print(dir(torch.distributions))
```


Thông thường, chúng ta có thể bỏ qua các hàm bắt đầu và kết thúc bằng `__` (các đối tượng đặc biệt trong Python)
hoặc các hàm bắt đầu bằng một `_` duy nhất (thường là các hàm nội bộ).
Dựa trên các tên hàm hoặc thuộc tính còn lại,
chúng ta có thể đoán rằng module này cung cấp
các phương thức khác nhau để tạo số ngẫu nhiên,
bao gồm lấy mẫu từ phân phối đều (`uniform`),
phân phối chuẩn (`normal`) và phân phối đa thức (`multinomial`).

## Hàm và Lớp cụ thể

Để có hướng dẫn cụ thể về cách sử dụng một hàm hoặc lớp nhất định,
chúng ta có thể sử dụng hàm `help`. Ví dụ, hãy
[**khám phá hướng dẫn sử dụng hàm `ones` của tensor**].


```python
help(torch.ones)
```


Từ tài liệu, chúng ta có thể thấy rằng hàm `ones`
tạo một tensor mới với hình dạng được chỉ định
và đặt tất cả các phần tử thành giá trị 1.
Bất cứ khi nào có thể, bạn nên (**chạy thử nhanh**)
để xác nhận cách hiểu của mình:


```python
torch.ones(4)
```


Trong Jupyter notebook, chúng ta có thể sử dụng `?` để hiển thị tài liệu trong một
cửa sổ khác. Ví dụ, `list?` sẽ tạo nội dung
gần như giống với `help(list)`,
hiển thị trong cửa sổ trình duyệt mới.
Ngoài ra, nếu chúng ta sử dụng hai dấu chấm hỏi, như `list??`,
code Python triển khai hàm cũng sẽ được hiển thị.

Tài liệu chính thức cung cấp nhiều mô tả và ví dụ vượt ra ngoài cuốn sách này.
Chúng tôi nhấn mạnh các trường hợp sử dụng quan trọng
sẽ giúp bạn bắt đầu nhanh chóng với các bài toán thực tế,
thay vì tính đầy đủ của nội dung.
Chúng tôi cũng khuyến khích bạn nghiên cứu mã nguồn của các thư viện
để xem các ví dụ về triển khai code sản xuất chất lượng cao.
Bằng cách này bạn sẽ trở thành một kỹ sư giỏi hơn
ngoài việc trở thành một nhà khoa học giỏi hơn.


[Thảo luận](https://discuss.d2l.ai/t/39)
