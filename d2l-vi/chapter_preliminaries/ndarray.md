# Thao tác với Dữ liệu
<a id="sec_ndarray"></a>

Để làm được bất cứ điều gì,
chúng ta cần một cách để lưu trữ và thao tác dữ liệu.
Nhìn chung, có hai việc quan trọng
chúng ta cần làm với dữ liệu:
(i) thu thập dữ liệu;
và (ii) xử lý dữ liệu sau khi đã đưa vào máy tính.
Không có ích gì khi thu thập dữ liệu
mà không có cách nào lưu trữ,
vì vậy để bắt đầu, hãy cùng thực hành
với mảng $n$ chiều,
còn gọi là *tensor*.
Nếu bạn đã quen với gói tính toán khoa học NumPy,
phần này sẽ rất dễ dàng.
Đối với tất cả các framework học sâu hiện đại,
*lớp tensor* (`ndarray` trong MXNet,
`Tensor` trong PyTorch và TensorFlow)
tương tự như `ndarray` của NumPy,
nhưng được bổ sung thêm một số tính năng mạnh mẽ.
Đầu tiên, lớp tensor
hỗ trợ vi phân tự động.
Thứ hai, nó tận dụng GPU
để tăng tốc tính toán số học,
trong khi NumPy chỉ chạy trên CPU.
Những tính chất này làm cho mạng nơ-ron
vừa dễ lập trình vừa chạy nhanh.


## Bắt đầu


(**Để bắt đầu, chúng ta nhập thư viện PyTorch.
Lưu ý rằng tên gói là `torch`.**)


```python
import torch
```


[**Một tensor biểu diễn một mảng (có thể là đa chiều) các giá trị số.**]
Trong trường hợp một chiều, tức là khi chỉ cần một trục cho dữ liệu,
tensor được gọi là *vector*.
Với hai trục, tensor được gọi là *ma trận*.
Với $k > 2$ trục, chúng ta bỏ các tên đặc biệt
và chỉ gọi đối tượng là tensor *bậc $k$*.


PyTorch cung cấp nhiều hàm
để tạo tensor mới
được khởi tạo sẵn với các giá trị.
Ví dụ, bằng cách gọi `arange(n)`,
chúng ta có thể tạo một vector các giá trị cách đều nhau,
bắt đầu từ 0 (bao gồm)
và kết thúc tại `n` (không bao gồm).
Mặc định, khoảng cách giữa các giá trị là $1$.
Trừ khi có chỉ định khác,
các tensor mới được lưu trong bộ nhớ chính
và được chỉ định cho tính toán trên CPU.


```python
x = torch.arange(12, dtype=torch.float32)
x
```


Mỗi giá trị này được gọi là
một *phần tử* của tensor.
Tensor `x` chứa 12 phần tử.
Chúng ta có thể kiểm tra tổng số phần tử
trong một tensor thông qua phương thức `numel` của nó.


```python
x.numel()
```


(**Chúng ta có thể truy cập *shape* của tensor**)
(độ dài dọc theo mỗi trục)
bằng cách kiểm tra thuộc tính `shape` của nó.
Vì chúng ta đang xử lý một vector ở đây,
`shape` chỉ chứa một phần tử duy nhất
và giống hệt kích thước.

```python
x.shape
```

Chúng ta có thể [**thay đổi shape của tensor
mà không làm thay đổi kích thước hay giá trị của nó**],
bằng cách gọi `reshape`.
Ví dụ, chúng ta có thể biến đổi
vector `x` có shape (12,)
thành ma trận `X` có shape (3, 4).
Tensor mới này giữ lại tất cả các phần tử
nhưng sắp xếp lại chúng thành dạng ma trận.
Lưu ý rằng các phần tử của vector
được sắp xếp từng hàng một, do đó
`x[3] == X[0, 3]`.


Lưu ý rằng việc chỉ định tất cả các thành phần shape
cho `reshape` là thừa.
Vì chúng ta đã biết kích thước của tensor,
chúng ta có thể tính ra một thành phần của shape từ các thành phần còn lại.
Ví dụ, với tensor có kích thước $n$
và shape mục tiêu ($h$, $w$),
chúng ta biết rằng $w = n/h$.
Để tự động suy ra một thành phần của shape,
chúng ta có thể đặt `-1` cho thành phần shape
cần được suy ra tự động.
Trong trường hợp này, thay vì gọi `x.reshape(3, 4)`,
chúng ta có thể gọi tương đương `x.reshape(-1, 4)` hoặc `x.reshape(3, -1)`.

Người dùng thường cần làm việc với tensor
được khởi tạo chứa toàn số 0 hoặc 1.
[**Chúng ta có thể tạo tensor với tất cả phần tử bằng 0**] (~~hoặc một~~)
và shape (2, 3, 4) thông qua hàm `zeros`.


```python
torch.zeros((2, 3, 4))
```


Tương tự, chúng ta có thể tạo tensor
với toàn số 1 bằng cách gọi `ones`.


```python
torch.ones((2, 3, 4))
```


Chúng ta thường muốn
[**lấy mẫu ngẫu nhiên (và độc lập) từng phần tử**]
từ một phân phối xác suất cho trước.
Ví dụ, các tham số của mạng nơ-ron
thường được khởi tạo ngẫu nhiên.
Đoạn code sau tạo một tensor
với các phần tử được lấy từ
phân phối Gauss chuẩn (phân phối chuẩn)
với trung bình 0 và độ lệch chuẩn 1.


```python
torch.randn(3, 4)
```


Cuối cùng, chúng ta có thể xây dựng tensor bằng cách
[**cung cấp giá trị chính xác cho từng phần tử**]
thông qua danh sách Python (có thể lồng nhau)
chứa các giá trị số.
Ở đây, chúng ta xây dựng một ma trận với danh sách của các danh sách,
trong đó danh sách ngoài cùng tương ứng với trục 0,
và danh sách bên trong tương ứng với trục 1.


```python
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```


## Chỉ mục và Cắt lát

Giống như danh sách Python,
chúng ta có thể truy cập các phần tử tensor
bằng cách lập chỉ mục (bắt đầu từ 0).
Để truy cập một phần tử dựa trên vị trí của nó
tính từ cuối danh sách,
chúng ta có thể sử dụng chỉ mục âm.
Cuối cùng, chúng ta có thể truy cập toàn bộ dải chỉ mục
thông qua cắt lát (ví dụ: `X[start:stop]`),
trong đó giá trị trả về bao gồm
chỉ mục đầu tiên (`start`) *nhưng không bao gồm* chỉ mục cuối (`stop`).
Khi chỉ một chỉ mục (hoặc lát cắt)
được chỉ định cho tensor bậc $k$,
nó được áp dụng dọc theo trục 0.
Do đó, trong đoạn code sau,
[**`[-1]` chọn hàng cuối cùng và `[1:3]`
chọn hàng thứ hai và thứ ba**].

```python
X[-1], X[1:3]
```


Nếu chúng ta muốn [**gán cùng một giá trị cho nhiều phần tử,
chúng ta áp dụng chỉ mục ở vế trái
của phép gán.**]
Ví dụ, `[:2, :]` truy cập
hàng thứ nhất và thứ hai,
trong đó `:` lấy tất cả các phần tử dọc theo trục 1 (cột).
Mặc dù chúng ta đã thảo luận về chỉ mục cho ma trận,
cách này cũng hoạt động với vector
và với tensor có nhiều hơn hai chiều.


## Các Phép Toán

Bây giờ chúng ta đã biết cách tạo tensor
và cách đọc, ghi các phần tử của chúng,
chúng ta có thể bắt đầu thao tác với chúng
bằng các phép toán toán học khác nhau.
Trong số những phép toán hữu ích nhất
là các phép toán *theo từng phần tử* (elementwise).
Chúng áp dụng một phép toán vô hướng tiêu chuẩn
lên từng phần tử của tensor.
Đối với các hàm nhận hai tensor làm đầu vào,
các phép toán theo từng phần tử áp dụng toán tử nhị phân tiêu chuẩn
lên từng cặp phần tử tương ứng.
Chúng ta có thể tạo hàm theo từng phần tử
từ bất kỳ hàm nào ánh xạ
từ vô hướng sang vô hướng.

Trong ký hiệu toán học, chúng ta ký hiệu
các toán tử vô hướng *đơn ngôi* (nhận một đầu vào)
bằng chữ ký
$f: \mathbb{R} \rightarrow \mathbb{R}$.
Điều này có nghĩa là hàm ánh xạ
từ bất kỳ số thực nào sang một số thực khác.
Hầu hết các toán tử tiêu chuẩn, bao gồm cả toán tử đơn ngôi như $e^x$, có thể được áp dụng theo từng phần tử.


```python
torch.exp(x)
```


Tương tự, chúng ta ký hiệu các toán tử vô hướng *nhị phân*,
ánh xạ các cặp số thực
sang một số thực (duy nhất)
bằng chữ ký
$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Cho hai vector bất kỳ $\mathbf{u}$
và $\mathbf{v}$ *có cùng shape*,
và toán tử nhị phân $f$, chúng ta có thể tạo ra vector
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$
bằng cách đặt $c_i \gets f(u_i, v_i)$ với mọi $i$,
trong đó $c_i, u_i$, và $v_i$ là các phần tử thứ $i$
của vector $\mathbf{c}, \mathbf{u}$, và $\mathbf{v}$.
Ở đây, chúng ta tạo ra
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ có giá trị vector
bằng cách *nâng* hàm vô hướng
lên thành phép toán vector theo từng phần tử.
Các toán tử số học tiêu chuẩn phổ biến
cho phép cộng (`+`), trừ (`-`),
nhân (`*`), chia (`/`),
và lũy thừa (`**`)
đều đã được *nâng* lên thành phép toán theo từng phần tử
cho các tensor có shape giống hệt nhau và hình dạng tùy ý.


```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```


Ngoài các phép tính theo từng phần tử,
chúng ta cũng có thể thực hiện các phép toán đại số tuyến tính,
chẳng hạn như tích vô hướng và phép nhân ma trận.
Chúng ta sẽ giải thích chi tiết về điều này
trong [sec_linear-algebra](#sec_linear-algebra).

Chúng ta cũng có thể [***nối* nhiều tensor,**]
xếp chúng đầu cuối vào nhau để tạo thành một tensor lớn hơn.
Chúng ta chỉ cần cung cấp danh sách các tensor
và cho hệ thống biết nên nối dọc theo trục nào.
Ví dụ dưới đây cho thấy điều gì xảy ra khi chúng ta nối
hai ma trận dọc theo hàng (trục 0)
thay vì cột (trục 1).
Chúng ta thấy rằng độ dài trục-0 của đầu ra đầu tiên ($6$)
là tổng độ dài trục-0 của hai tensor đầu vào ($3 + 3$);
trong khi độ dài trục-1 của đầu ra thứ hai ($8$)
là tổng độ dài trục-1 của hai tensor đầu vào ($4 + 4$).


```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```


Đôi khi, chúng ta muốn
[**xây dựng tensor nhị phân thông qua *các phát biểu logic*.**]
Lấy `X == Y` làm ví dụ.
Với mỗi vị trí `i, j`, nếu `X[i, j]` và `Y[i, j]` bằng nhau,
thì phần tử tương ứng trong kết quả nhận giá trị `1`,
ngược lại nhận giá trị `0`.

```python
X == Y
```

[**Tính tổng tất cả các phần tử trong tensor**] tạo ra tensor chỉ có một phần tử.


## Broadcasting
<a id="subsec_broadcasting"></a>

Đến đây, bạn đã biết cách thực hiện
các phép toán nhị phân theo từng phần tử
trên hai tensor có cùng shape.
Trong một số điều kiện nhất định,
ngay cả khi shape khác nhau,
chúng ta vẫn có thể [**thực hiện các phép toán nhị phân theo từng phần tử
bằng cách gọi *cơ chế broadcasting*.**]
Broadcasting hoạt động theo
quy trình hai bước sau:
(i) mở rộng một hoặc cả hai mảng
bằng cách sao chép các phần tử dọc theo các trục có độ dài 1
sao cho sau khi biến đổi này,
hai tensor có cùng shape;
(ii) thực hiện phép toán theo từng phần tử
trên các mảng kết quả.


```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```


Vì `a` và `b` lần lượt là ma trận $3\times1$
và $1\times2$,
shape của chúng không khớp nhau.
Broadcasting tạo ra ma trận $3\times2$ lớn hơn
bằng cách sao chép ma trận `a` dọc theo các cột
và ma trận `b` dọc theo các hàng
trước khi cộng chúng theo từng phần tử.

```python
a + b
```

## Tiết kiệm Bộ nhớ

[**Thực hiện các phép toán có thể khiến bộ nhớ mới được
cấp phát để lưu kết quả.**]
Ví dụ, nếu chúng ta viết `Y = X + Y`,
chúng ta hủy tham chiếu đến tensor mà `Y` trỏ đến trước đó
và thay vào đó trỏ `Y` đến bộ nhớ mới được cấp phát.
Chúng ta có thể minh họa vấn đề này với hàm `id()` của Python,
hàm này cho chúng ta địa chỉ chính xác
của đối tượng được tham chiếu trong bộ nhớ.
Lưu ý rằng sau khi chạy `Y = Y + X`,
`id(Y)` trỏ đến một vị trí khác.
Đó là vì Python trước tiên đánh giá `Y + X`,
cấp phát bộ nhớ mới cho kết quả
và sau đó trỏ `Y` đến vị trí mới này trong bộ nhớ.

```python
before = id(Y)
Y = Y + X
id(Y) == before
```

Điều này có thể không mong muốn vì hai lý do.
Đầu tiên, chúng ta không muốn liên tục
cấp phát bộ nhớ không cần thiết.
Trong học máy, chúng ta thường có
hàng trăm megabyte tham số
và cập nhật tất cả chúng nhiều lần mỗi giây.
Bất cứ khi nào có thể, chúng ta muốn thực hiện các cập nhật này *tại chỗ*.
Thứ hai, chúng ta có thể trỏ đến
cùng tham số từ nhiều biến.
Nếu chúng ta không cập nhật tại chỗ,
chúng ta phải cẩn thận cập nhật tất cả các tham chiếu này,
kẻo gây ra rò rỉ bộ nhớ
hoặc vô tình tham chiếu đến tham số lỗi thời.


```python
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```


## Chuyển đổi sang Đối tượng Python Khác


[**Chuyển đổi sang tensor NumPy (`ndarray`)**], hoặc ngược lại, rất dễ dàng.
Tensor torch và mảng NumPy
sẽ chia sẻ bộ nhớ cơ bản của chúng,
và thay đổi một trong số chúng thông qua phép toán tại chỗ
cũng sẽ thay đổi cái kia.


```python
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```


Để (**chuyển đổi tensor kích thước 1 sang số vô hướng Python**),
chúng ta có thể gọi hàm `item` hoặc các hàm tích hợp sẵn của Python.


```python
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```


## Tóm tắt

Lớp tensor là giao diện chính để lưu trữ và thao tác dữ liệu trong các thư viện học sâu.
Tensor cung cấp nhiều chức năng bao gồm các hàm tạo; chỉ mục và cắt lát; các phép toán toán học cơ bản; broadcasting; gán hiệu quả bộ nhớ; và chuyển đổi sang và từ các đối tượng Python khác.


## Bài tập

1. Chạy code trong phần này. Thay đổi câu lệnh điều kiện `X == Y` thành `X < Y` hoặc `X > Y`, và xem loại tensor nào bạn có thể nhận được.
1. Thay thế hai tensor hoạt động theo từng phần tử trong cơ chế broadcasting bằng các shape khác, ví dụ: tensor 3 chiều. Kết quả có như mong đợi không?


[Thảo luận](https://discuss.d2l.ai/t/27)
