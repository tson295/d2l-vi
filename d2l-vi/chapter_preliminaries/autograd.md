# Vi phân Tự động
<a id="sec_autograd"></a>

Nhớ lại từ [sec_calculus](#sec_calculus)
rằng tính đạo hàm là bước quan trọng
trong tất cả các thuật toán tối ưu hóa
mà ta sẽ dùng để huấn luyện mạng sâu.
Mặc dù các phép tính này khá đơn giản,
nhưng làm thủ công rất tẻ nhạt và dễ sai,
và những vấn đề này chỉ ngày càng tệ hơn
khi mô hình của ta trở nên phức tạp hơn.

May mắn thay, tất cả các framework deep learning hiện đại
đều giải phóng ta khỏi công việc này
bằng cách cung cấp *vi phân tự động*
(thường viết tắt là *autograd*).
Khi ta truyền dữ liệu qua từng hàm liên tiếp,
framework xây dựng một *đồ thị tính toán*
theo dõi cách mỗi giá trị phụ thuộc vào các giá trị khác.
Để tính đạo hàm,
vi phân tự động
đi ngược qua đồ thị này
áp dụng quy tắc dây chuyền.
Thuật toán tính toán để áp dụng quy tắc dây chuyền
theo cách này được gọi là *lan truyền ngược*.

Mặc dù các thư viện autograd đã trở thành
mối quan tâm nóng trong thập kỷ qua,
chúng có một lịch sử lâu dài.
Trên thực tế, các tài liệu tham khảo sớm nhất về autograd
có từ hơn nửa thế kỷ trước [Wengert.1964].
Các ý tưởng cốt lõi đằng sau lan truyền ngược hiện đại
có từ một luận văn tiến sĩ năm 1980 [Speelpenning.1980]
và được phát triển thêm vào cuối những năm 1980 [Griewank.1989].
Mặc dù lan truyền ngược đã trở thành phương pháp mặc định
để tính gradient, đây không phải là lựa chọn duy nhất.
Ví dụ, ngôn ngữ lập trình Julia sử dụng
lan truyền xuôi [Revels.Lubin.Papamarkou.2016].
Trước khi khám phá các phương pháp,
hãy làm chủ gói autograd trước.


```python
import torch
```


## Một Hàm Đơn giản

Giả sử ta quan tâm đến
(**vi phân hàm
$y = 2\mathbf{x}^{\top}\mathbf{x}$
theo vector cột $\mathbf{x}$.**)
Để bắt đầu, ta gán giá trị ban đầu cho `x`.


```python
x = torch.arange(4.0)
x
```


```python
# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
x.grad  # The gradient is None by default
```


(**Ta bây giờ tính hàm của `x` và gán kết quả cho `y`.**)


```python
y = 2 * torch.dot(x, x)
y
```


[**Ta bây giờ có thể lấy gradient của `y`
theo `x`**] bằng cách gọi
phương thức `backward` của nó.
Tiếp theo, ta có thể truy cập gradient
qua thuộc tính `grad` của `x`.


```python
y.backward()
x.grad
```


(**Ta đã biết rằng gradient của hàm $y = 2\mathbf{x}^{\top}\mathbf{x}$
theo $\mathbf{x}$ phải là $4\mathbf{x}$.**)
Ta bây giờ có thể xác nhận rằng kết quả tính gradient tự động
và kết quả kỳ vọng là giống hệt nhau.


```python
x.grad == 4 * x
```


[**Bây giờ hãy tính
một hàm khác của `x`
và lấy gradient của nó.**]
Lưu ý rằng PyTorch không tự động
đặt lại bộ đệm gradient
khi ta ghi lại gradient mới.
Thay vào đó, gradient mới
được cộng vào gradient đã lưu.
Hành vi này hữu ích
khi ta muốn tối ưu hóa tổng
của nhiều hàm mục tiêu.
Để đặt lại bộ đệm gradient,
ta có thể gọi `x.grad.zero_()` như sau:


```python
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
x.grad
```


## Lan truyền Ngược cho Biến Không Vô hướng

Khi `y` là một vector,
biểu diễn tự nhiên nhất
của đạo hàm của `y`
theo vector `x`
là một ma trận gọi là *Jacobian*
chứa các đạo hàm riêng
của mỗi thành phần của `y`
theo mỗi thành phần của `x`.
Tương tự, với `y` và `x` bậc cao hơn,
kết quả vi phân có thể là tensor bậc cao hơn nữa.

Mặc dù Jacobian xuất hiện trong một số
kỹ thuật machine learning nâng cao,
thông thường hơn ta muốn tổng hợp
gradient của mỗi thành phần của `y`
theo vector đầy đủ `x`,
tạo ra vector có cùng shape với `x`.
Ví dụ, ta thường có vector
biểu diễn giá trị hàm mất mát
được tính riêng lẻ cho mỗi mẫu trong số
một *batch* các mẫu huấn luyện.
Ở đây, ta chỉ muốn (**tổng hợp các gradient
được tính riêng lẻ cho mỗi mẫu**).


Vì các framework deep learning khác nhau
trong cách diễn giải gradient của
các tensor không vô hướng,
PyTorch thực hiện một số bước để tránh nhầm lẫn.
Gọi `backward` trên đối tượng không vô hướng sẽ báo lỗi
trừ khi ta cho PyTorch biết cách rút gọn đối tượng thành vô hướng.
Chính thức hơn, ta cần cung cấp vector $\mathbf{v}$ nào đó
sao cho `backward` sẽ tính
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$
thay vì $\partial_{\mathbf{x}} \mathbf{y}$.
Phần tiếp theo có thể gây nhầm lẫn,
nhưng vì lý do sẽ trở nên rõ ràng sau,
đối số này (biểu diễn $\mathbf{v}$) được đặt tên là `gradient`.
Để mô tả chi tiết hơn, xem bài đăng
[Medium của Yang Zhang](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29).


```python
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
x.grad
```


## Tách rời Tính toán

Đôi khi, ta muốn [**di chuyển một số phép tính
ra ngoài đồ thị tính toán được ghi lại.**]
Ví dụ, giả sử ta dùng đầu vào
để tạo ra một số hạng trung gian phụ
mà ta không muốn tính gradient.
Trong trường hợp này, ta cần *tách rời*
đồ thị tính toán tương ứng
khỏi kết quả cuối cùng.
Ví dụ đơn giản sau làm rõ điều này:
giả sử ta có `z = x * y` và `y = x * x`
nhưng ta muốn tập trung vào ảnh hưởng *trực tiếp* của `x` lên `z`
thay vì ảnh hưởng thông qua `y`.
Trong trường hợp này, ta có thể tạo biến mới `u`
nhận cùng giá trị với `y`
nhưng *nguồn gốc* của nó (cách nó được tạo ra)
đã bị xóa.
Như vậy `u` không có tổ tiên trong đồ thị
và gradient không chảy qua `u` đến `x`.
Ví dụ, lấy gradient của `z = x * u`
sẽ cho kết quả `u`,
(không phải `3 * x * x` như bạn có thể
đã kỳ vọng vì `z = x * x * x`).


```python
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```


Lưu ý rằng mặc dù thủ tục này
tách rời các tổ tiên của `y`
khỏi đồ thị dẫn đến `z`,
đồ thị tính toán dẫn đến `y`
vẫn tồn tại và do đó ta có thể tính
gradient của `y` theo `x`.


```python
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```


## Gradient và Luồng Điều khiển Python

Cho đến nay ta đã xét các trường hợp mà đường đi từ đầu vào đến đầu ra
được xác định rõ ràng qua hàm như `z = x * x * x`.
Lập trình cho ta nhiều tự do hơn trong cách tính kết quả.
Ví dụ, ta có thể làm chúng phụ thuộc vào các biến phụ
hoặc điều kiện hóa các lựa chọn dựa trên kết quả trung gian.
Một lợi ích của việc sử dụng vi phân tự động
là [**ngay cả khi**] xây dựng đồ thị tính toán của
(**hàm yêu cầu đi qua mê cung của luồng điều khiển Python**)
(ví dụ: điều kiện, vòng lặp, và các lời gọi hàm tùy ý),
(**ta vẫn có thể tính gradient của biến kết quả.**)
Để minh họa điều này, hãy xem đoạn code sau, trong đó
số lần lặp của vòng lặp `while`
và kết quả đánh giá của câu lệnh `if`
đều phụ thuộc vào giá trị của đầu vào `a`.


```python
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```


Dưới đây, ta gọi hàm này, truyền vào một giá trị ngẫu nhiên làm đầu vào.
Vì đầu vào là biến ngẫu nhiên,
ta không biết đồ thị tính toán
sẽ có dạng gì.
Tuy nhiên, bất cứ khi nào ta thực thi `f(a)`
trên một đầu vào cụ thể, ta hiện thực hóa
một đồ thị tính toán cụ thể
và sau đó có thể chạy `backward`.


```python
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```


Mặc dù hàm `f` của ta, vì mục đích minh họa, hơi phức tạp một chút,
sự phụ thuộc vào đầu vào của nó khá đơn giản:
đây là *hàm tuyến tính* của `a`
với tỉ lệ được định nghĩa theo từng đoạn.
Vì vậy, `f(a) / a` là vector gồm các phần tử hằng số
và hơn nữa, `f(a) / a` phải khớp
với gradient của `f(a)` theo `a`.


```python
a.grad == d / a
```


Luồng điều khiển động rất phổ biến trong deep learning.
Ví dụ, khi xử lý văn bản, đồ thị tính toán
phụ thuộc vào độ dài của đầu vào.
Trong những trường hợp này, vi phân tự động
trở nên thiết yếu cho mô hình hóa thống kê
vì không thể tính gradient *a priori*.

## Thảo luận

Bạn đã được nếm trải sức mạnh của vi phân tự động.
Việc phát triển các thư viện tính đạo hàm
cả tự động và hiệu quả
đã là một bước tăng năng suất khổng lồ
cho những người thực hành deep learning,
giải phóng họ để tập trung vào những việc ít tẻ nhạt hơn.
Hơn nữa, autograd cho phép ta thiết kế các mô hình khổng lồ
mà các phép tính gradient bằng bút và giấy
sẽ tốn quá nhiều thời gian.
Thú vị thay, trong khi ta dùng autograd để *tối ưu hóa* mô hình
(theo nghĩa thống kê),
chính việc *tối ưu hóa* các thư viện autograd
(theo nghĩa tính toán)
là một chủ đề phong phú
rất quan trọng đối với các nhà thiết kế framework.
Ở đây, các công cụ từ trình biên dịch và thao tác đồ thị
được tận dụng để tính toán kết quả
theo cách nhanh nhất và hiệu quả nhất về bộ nhớ.

Hiện tại, hãy cố nhớ những điều cơ bản sau: (i) gắn gradient vào các biến mà ta muốn lấy đạo hàm theo; (ii) ghi lại quá trình tính giá trị mục tiêu; (iii) thực thi hàm lan truyền ngược; và (iv) truy cập gradient kết quả.


## Bài tập

1. Tại sao đạo hàm bậc hai tốn kém hơn nhiều để tính so với đạo hàm bậc nhất?
1. Sau khi chạy hàm cho lan truyền ngược, chạy lại ngay và xem điều gì xảy ra. Hãy điều tra.
1. Trong ví dụ luồng điều khiển nơi ta tính đạo hàm của `d` theo `a`, điều gì sẽ xảy ra nếu ta thay đổi biến `a` thành vector hoặc ma trận ngẫu nhiên? Lúc này, kết quả của phép tính `f(a)` không còn là vô hướng nữa. Điều gì xảy ra với kết quả? Ta phân tích điều này như thế nào?
1. Cho $f(x) = \sin(x)$. Vẽ đồ thị của $f$ và đạo hàm $f'$ của nó. Đừng khai thác việc $f'(x) = \cos(x)$ mà thay vào đó dùng vi phân tự động để thu kết quả.
1. Cho $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$. Viết ra đồ thị phụ thuộc truy vết kết quả từ $x$ đến $f(x)$.
1. Dùng quy tắc dây chuyền để tính đạo hàm $\frac{df}{dx}$ của hàm đã đề cập ở trên, đặt mỗi số hạng trên đồ thị phụ thuộc mà bạn đã xây dựng trước đó.
1. Với đồ thị và kết quả đạo hàm trung gian, bạn có một số lựa chọn khi tính gradient. Đánh giá kết quả một lần bắt đầu từ $x$ đến $f$ và một lần từ $f$ truy ngược về $x$. Con đường từ $x$ đến $f$ thường được gọi là *vi phân xuôi*, trong khi con đường từ $f$ về $x$ được gọi là vi phân ngược.
1. Khi nào bạn muốn dùng vi phân xuôi, và khi nào dùng vi phân ngược? Gợi ý: hãy xem xét lượng dữ liệu trung gian cần thiết, khả năng song song hóa các bước, và kích thước của các ma trận và vector liên quan.


[Thảo luận](https://discuss.d2l.ai/t/35)
