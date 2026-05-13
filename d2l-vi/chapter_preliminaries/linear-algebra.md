# Đại số Tuyến tính
<a id="sec_linear-algebra"></a>

Đến đây, ta có thể tải tập dữ liệu vào tensor
và thao tác với các tensor này
bằng các phép toán toán học cơ bản.
Để bắt đầu xây dựng các mô hình phức tạp,
ta cũng cần một số công cụ từ đại số tuyến tính.
Phần này cung cấp phần giới thiệu nhẹ nhàng
về các khái niệm thiết yếu nhất,
bắt đầu từ số học vô hướng
và nâng dần lên đến phép nhân ma trận.


```python
import torch
```


## Vô hướng

Hầu hết các phép toán hàng ngày
đều bao gồm thao tác
với từng số một.
Chính thức, ta gọi các giá trị này là *vô hướng*.
Ví dụ, nhiệt độ ở Palo Alto
là $72$ độ Fahrenheit dễ chịu.
Nếu bạn muốn chuyển đổi nhiệt độ sang độ Celsius,
bạn sẽ tính biểu thức
$c = \frac{5}{9}(f - 32)$, đặt $f$ bằng $72$.
Trong phương trình này, các giá trị
$5$, $9$, và $32$ là các vô hướng hằng.
Các biến $c$ và $f$
nói chung biểu diễn các vô hướng chưa biết.

Ta ký hiệu vô hướng
bằng các chữ cái thường thông thường
(ví dụ: $x$, $y$, và $z$)
và không gian của tất cả (liên tục)
các vô hướng *có giá trị thực* bằng $\mathbb{R}$.
Để tiện lợi, ta sẽ bỏ qua
các định nghĩa chặt chẽ về *không gian*:
chỉ cần nhớ rằng biểu thức $x \in \mathbb{R}$
là cách chính thức để nói rằng $x$ là vô hướng có giá trị thực.
Ký hiệu $\in$ (đọc là "thuộc")
biểu thị tư cách thành viên trong một tập hợp.
Ví dụ, $x, y \in \{0, 1\}$
chỉ ra rằng $x$ và $y$ là các biến
chỉ có thể nhận giá trị $0$ hoặc $1$.

(**Vô hướng được triển khai dưới dạng tensor
chỉ chứa một phần tử.**)
Dưới đây, ta gán hai vô hướng
và thực hiện các phép cộng, nhân,
chia và lũy thừa quen thuộc.


```python
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```


## Vector

Để phục vụ hiện tại, [**bạn có thể nghĩ một vector là mảng vô hướng có độ dài cố định.**]
Cũng như các đối tác trong code,
ta gọi các vô hướng này là *phần tử* của vector
(đồng nghĩa bao gồm *mục* và *thành phần*).
Khi vector biểu diễn các mẫu từ tập dữ liệu thực tế,
các giá trị của chúng mang ý nghĩa thực tế.
Ví dụ, nếu ta đang huấn luyện mô hình để dự đoán
rủi ro vỡ nợ của khoản vay,
ta có thể liên kết mỗi người xin vay với vector
có các thành phần tương ứng với các đại lượng
như thu nhập, thâm niên làm việc,
hoặc số lần vỡ nợ trước đây.
Nếu ta đang nghiên cứu rủi ro đau tim,
mỗi vector có thể biểu diễn một bệnh nhân
và các thành phần của nó có thể tương ứng với
các dấu hiệu sinh tồn gần nhất, mức cholesterol,
số phút tập thể dục mỗi ngày, v.v.
Ta ký hiệu vector bằng chữ thường in đậm,
(ví dụ: $\mathbf{x}$, $\mathbf{y}$, và $\mathbf{z}$).

Vector được triển khai dưới dạng tensor bậc $1$.
Nói chung, các tensor như vậy có thể có độ dài tùy ý,
tùy thuộc vào giới hạn bộ nhớ. Lưu ý: trong Python, cũng như trong hầu hết các ngôn ngữ lập trình, chỉ số vector bắt đầu từ $0$, còn gọi là *chỉ mục gốc không*, trong khi trong đại số tuyến tính chỉ số bắt đầu từ $1$ (chỉ mục gốc một).


```python
x = torch.arange(3)
x
```


Ta có thể tham chiếu đến một phần tử của vector bằng chỉ số dưới.
Ví dụ, $x_2$ biểu thị phần tử thứ hai của $\mathbf{x}$.
Vì $x_2$ là vô hướng, ta không in đậm nó.
Mặc định, ta hình dung vector
bằng cách xếp chồng các phần tử theo chiều dọc:

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix}.$$

Ở đây $x_1, \ldots, x_n$ là các phần tử của vector.
Sau này, ta sẽ phân biệt giữa *vector cột* như vậy
và *vector hàng* có các phần tử xếp nằm ngang.
Nhớ lại [**ta truy cập phần tử tensor thông qua chỉ mục.**]

```python
x[2]
```

Để chỉ ra rằng một vector chứa $n$ phần tử,
ta viết $\mathbf{x} \in \mathbb{R}^n$.
Chính thức, ta gọi $n$ là *số chiều* của vector.
[**Trong code, điều này tương ứng với độ dài của tensor**],
có thể truy cập qua hàm `len` tích hợp của Python.

```python
len(x)
```

Ta cũng có thể truy cập độ dài qua thuộc tính `shape`.
Shape là bộ dữ liệu cho biết độ dài tensor dọc theo mỗi trục.
(**Tensor chỉ có một trục có shape chỉ với một phần tử.**)

```python
x.shape
```

Thường thì từ "chiều" được dùng lẫn lộn
để chỉ cả số trục
và độ dài dọc theo một trục cụ thể.
Để tránh nhầm lẫn này,
ta dùng *bậc* để chỉ số trục
và *số chiều* chuyên dùng để chỉ
số lượng thành phần.


## Ma trận

Cũng như vô hướng là tensor bậc $0$
và vector là tensor bậc $1$,
ma trận là tensor bậc $2$.
Ta ký hiệu ma trận bằng chữ hoa in đậm
(ví dụ: $\mathbf{X}$, $\mathbf{Y}$, và $\mathbf{Z}$),
và biểu diễn chúng trong code bằng tensor với hai trục.
Biểu thức $\mathbf{A} \in \mathbb{R}^{m \times n}$
chỉ ra rằng ma trận $\mathbf{A}$
chứa $m \times n$ vô hướng có giá trị thực,
được sắp xếp thành $m$ hàng và $n$ cột.
Khi $m = n$, ta nói ma trận là *vuông*.
Về mặt trực quan, ta có thể minh họa bất kỳ ma trận nào dưới dạng bảng.
Để tham chiếu đến một phần tử riêng lẻ,
ta lấy chỉ số dưới cả hàng và cột, ví dụ:
$a_{ij}$ là giá trị thuộc
hàng thứ $i$ và cột thứ $j$ của $\mathbf{A}$:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$

Trong code, ta biểu diễn ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$
bằng tensor bậc $2$ với shape ($m$, $n$).
[**Ta có thể chuyển đổi bất kỳ tensor $m \times n$ có kích thước phù hợp
thành ma trận $m \times n$**]
bằng cách truyền shape mong muốn vào `reshape`:


```python
A = torch.arange(6).reshape(3, 2)
A
```


Đôi khi ta muốn lật các trục.
Khi ta hoán đổi hàng và cột của ma trận,
kết quả được gọi là *chuyển vị* của nó.
Chính thức, ta ký hiệu chuyển vị của ma trận $\mathbf{A}$
bằng $\mathbf{A}^\top$ và nếu $\mathbf{B} = \mathbf{A}^\top$,
thì $b_{ij} = a_{ji}$ với mọi $i$ và $j$.
Như vậy, chuyển vị của ma trận $m \times n$
là ma trận $n \times m$:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

Trong code, ta có thể truy cập (**chuyển vị của bất kỳ ma trận nào**) như sau:


[**Ma trận đối xứng là tập con của ma trận vuông
bằng với chuyển vị của chính nó:
$\mathbf{A} = \mathbf{A}^\top$.**]
Ma trận sau đây là ma trận đối xứng:


```python
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```


Ma trận rất hữu ích để biểu diễn tập dữ liệu.
Thường thì hàng tương ứng với các bản ghi riêng lẻ
và cột tương ứng với các thuộc tính phân biệt.


## Tensor

Mặc dù bạn có thể đi xa trong hành trình machine learning
chỉ với vô hướng, vector và ma trận,
cuối cùng bạn có thể cần làm việc với
[**tensor**] bậc cao hơn.
Tensor (**cho ta cách tổng quát để mô tả
các phần mở rộng thành mảng $n$ chiều.**)
Ta gọi các đối tượng phần mềm của *lớp tensor* là "tensor"
chính xác vì chúng cũng có thể có số trục tùy ý.
Mặc dù có thể gây nhầm lẫn khi dùng từ
*tensor* cho cả đối tượng toán học
lẫn hiện thực hóa của nó trong code,
ý nghĩa của ta thường rõ ràng từ ngữ cảnh.
Ta ký hiệu tensor tổng quát bằng chữ hoa
với kiểu chữ đặc biệt
(ví dụ: $\mathsf{X}$, $\mathsf{Y}$, và $\mathsf{Z}$)
và cơ chế chỉ mục của chúng
(ví dụ: $x_{ijk}$ và $[\mathsf{X}]_{1, 2i-1, 3}$)
tuân theo tự nhiên từ cơ chế của ma trận.

Tensor sẽ trở nên quan trọng hơn
khi ta bắt đầu làm việc với ảnh.
Mỗi ảnh đến dưới dạng tensor bậc $3$
với các trục tương ứng với chiều cao, chiều rộng và *kênh*.
Tại mỗi vị trí không gian, cường độ
của mỗi màu (đỏ, xanh lá, và xanh dương)
được xếp chồng dọc theo kênh.
Hơn nữa, một tập hợp ảnh được biểu diễn
trong code bằng tensor bậc $4$,
trong đó các ảnh riêng biệt được lập chỉ mục
dọc theo trục đầu tiên.
Tensor bậc cao hơn được xây dựng, như vector và ma trận,
bằng cách tăng số thành phần shape.


```python
torch.arange(24).reshape(2, 3, 4)
```


## Tính chất Cơ bản của Số học Tensor

Vô hướng, vector, ma trận,
và tensor bậc cao hơn
đều có một số tính chất tiện dụng.
Ví dụ, các phép toán theo từng phần tử
tạo ra các đầu ra có cùng
shape với các toán hạng của chúng.


```python
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B
```


[**Tích theo từng phần tử của hai ma trận
được gọi là *tích Hadamard***] (ký hiệu $\odot$).
Ta có thể liệt kê các phần tử
của tích Hadamard của hai ma trận
$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$:

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```python
A * B
```

[**Cộng hoặc nhân một vô hướng và một tensor**] tạo ra kết quả
có cùng shape với tensor ban đầu.
Ở đây, mỗi phần tử của tensor được cộng vào (hoặc nhân với) vô hướng.


```python
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```


## Rút gọn
<a id="subsec_lin-alg-reduction"></a>

Thường thì ta muốn tính [**tổng các phần tử của tensor.**]
Để biểu diễn tổng các phần tử trong vector $\mathbf{x}$ có độ dài $n$,
ta viết $\sum_{i=1}^n x_i$. Có một hàm đơn giản cho điều này:


```python
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```


Để biểu diễn [**tổng qua các phần tử của tensor có shape tùy ý**],
ta chỉ cần tổng qua tất cả các trục của nó.
Ví dụ, tổng các phần tử
của ma trận $m \times n$ $\mathbf{A}$
có thể được viết là $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.


Mặc định, gọi hàm sum
*rút gọn* tensor dọc theo tất cả các trục của nó,
cuối cùng tạo ra một vô hướng.
Các thư viện của ta cũng cho phép ta [**chỉ định các trục
dọc theo đó tensor nên được rút gọn.**]
Để tính tổng qua tất cả phần tử dọc theo hàng (trục 0),
ta chỉ định `axis=0` trong `sum`.
Vì ma trận đầu vào được rút gọn dọc theo trục 0
để tạo ra vector đầu ra,
trục này bị thiếu trong shape của đầu ra.


Chỉ định `axis=1` sẽ rút gọn chiều cột (trục 1) bằng cách tính tổng tất cả các phần tử của tất cả các cột.


Rút gọn ma trận dọc theo cả hàng và cột bằng phép tính tổng
tương đương với tính tổng tất cả các phần tử của ma trận.


[**Một đại lượng liên quan là *trung bình*, còn gọi là *giá trị trung bình*.**]
Ta tính trung bình bằng cách chia tổng
cho tổng số phần tử.
Vì tính trung bình rất phổ biến,
nó được cung cấp hàm thư viện riêng
hoạt động tương tự như `sum`.


```python
A.mean(), A.sum() / A.numel()
```


Tương tự, hàm tính trung bình
cũng có thể rút gọn tensor dọc theo các trục cụ thể.


## Tổng Không Rút gọn
<a id="subsec_lin-alg-non-reduction"></a>

Đôi khi hữu ích khi [**giữ nguyên số trục**]
khi gọi hàm tính tổng hoặc trung bình.
Điều này quan trọng khi ta muốn sử dụng cơ chế broadcasting.


Ví dụ, vì `sum_A` giữ hai trục sau khi tính tổng mỗi hàng,
ta có thể (**chia `A` cho `sum_A` với broadcasting**)
để tạo ra ma trận trong đó mỗi hàng có tổng bằng $1$.

```python
A / sum_A
```

Nếu ta muốn tính [**tổng lũy tích của các phần tử của `A` dọc theo một trục nào đó**],
giả sử `axis=0` (hàng theo hàng), ta có thể gọi hàm `cumsum`.
Theo thiết kế, hàm này không rút gọn tensor đầu vào dọc theo bất kỳ trục nào.


## Tích vô hướng

Cho đến nay, ta chỉ thực hiện các phép toán theo từng phần tử, tổng và trung bình.
Và nếu đây là tất cả những gì ta có thể làm, đại số tuyến tính
sẽ không xứng đáng có phần riêng của nó.
May mắn thay, đây là nơi mọi thứ trở nên thú vị hơn.
Một trong những phép toán cơ bản nhất là tích vô hướng.
Cho hai vector $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$,
*tích vô hướng* của chúng $\mathbf{x}^\top \mathbf{y}$ (còn gọi là *tích trong*, $\langle \mathbf{x}, \mathbf{y}  \rangle$)
là tổng tích của các phần tử cùng vị trí:
$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

[~~*Tích vô hướng* của hai vector là tổng tích của các phần tử cùng vị trí~~]


```python
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```


Tương đương, (**ta có thể tính tích vô hướng của hai vector
bằng cách thực hiện nhân theo từng phần tử rồi lấy tổng:**)


```python
torch.sum(x * y)
```


Tích vô hướng hữu ích trong nhiều ngữ cảnh.
Ví dụ, cho một tập hợp các giá trị,
được ký hiệu bằng vector $\mathbf{x}  \in \mathbb{R}^n$,
và một tập hợp trọng số, được ký hiệu bằng $\mathbf{w} \in \mathbb{R}^n$,
tổng có trọng số của các giá trị trong $\mathbf{x}$
theo trọng số $\mathbf{w}$
có thể được biểu diễn dưới dạng tích vô hướng $\mathbf{x}^\top \mathbf{w}$.
Khi trọng số không âm
và tổng bằng $1$, tức là $\left(\sum_{i=1}^{n} {w_i} = 1\right)$,
tích vô hướng biểu diễn *trung bình có trọng số*.
Sau khi chuẩn hóa hai vector có độ dài đơn vị,
tích vô hướng biểu diễn cosin của góc giữa chúng.
Ở phần sau của phần này, ta sẽ giới thiệu chính thức khái niệm *độ dài* này.


## Tích Ma trận--Vector

Bây giờ ta đã biết cách tính tích vô hướng,
ta có thể bắt đầu hiểu *tích*
giữa ma trận $m \times n$ $\mathbf{A}$
và vector $n$ chiều $\mathbf{x}$.
Để bắt đầu, ta hình dung ma trận của ta
theo các vector hàng của nó

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

trong đó mỗi $\mathbf{a}^\top_{i} \in \mathbb{R}^n$
là vector hàng biểu diễn hàng thứ $i$
của ma trận $\mathbf{A}$.

[**Tích ma trận--vector $\mathbf{A}\mathbf{x}$
đơn giản là vector cột có độ dài $m$,
có phần tử thứ $i$ là tích vô hướng
$\mathbf{a}^\top_i \mathbf{x}$:**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

Ta có thể coi phép nhân với ma trận
$\mathbf{A}\in \mathbb{R}^{m \times n}$
như một phép biến đổi chiếu vector
từ $\mathbb{R}^{n}$ đến $\mathbb{R}^{m}$.
Các phép biến đổi này cực kỳ hữu ích.
Ví dụ, ta có thể biểu diễn các phép quay
dưới dạng phép nhân với một số ma trận vuông nhất định.
Tích ma trận--vector cũng mô tả
phép tính quan trọng liên quan đến tính toán
đầu ra của mỗi lớp trong mạng nơ-ron
dựa trên đầu ra từ lớp trước.


Để biểu diễn tích ma trận--vector trong code,
ta dùng hàm `mv`.
Lưu ý rằng chiều cột của `A`
(độ dài dọc theo trục 1)
phải bằng chiều của `x` (độ dài của nó).
Python có toán tử tiện dụng `@`
có thể thực hiện cả tích ma trận--vector
và tích ma trận--ma trận
(tùy thuộc vào đối số của nó).
Do đó ta có thể viết `A@x`.


```python
A.shape, x.shape, torch.mv(A, x), A@x
```


## Nhân Ma trận--Ma trận

Khi bạn đã quen với tích vô hướng và tích ma trận--vector,
thì *nhân ma trận--ma trận* sẽ đơn giản.

Giả sử ta có hai ma trận
$\mathbf{A} \in \mathbb{R}^{n \times k}$
và $\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$

Gọi $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ ký hiệu
vector hàng biểu diễn hàng thứ $i$
của ma trận $\mathbf{A}$
và gọi $\mathbf{b}_{j} \in \mathbb{R}^k$ ký hiệu
vector cột từ cột thứ $j$
của ma trận $\mathbf{B}$:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$

Để tạo thành tích ma trận $\mathbf{C} \in \mathbb{R}^{n \times m}$,
ta chỉ cần tính mỗi phần tử $c_{ij}$
là tích vô hướng giữa
hàng thứ $i$ của $\mathbf{A}$
và cột thứ $j$ của $\mathbf{B}$,
tức là $\mathbf{a}^\top_i \mathbf{b}_j$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**Ta có thể coi nhân ma trận--ma trận $\mathbf{AB}$
như thực hiện $m$ tích ma trận--vector
hoặc $m \times n$ tích vô hướng
và ghép các kết quả lại
để tạo thành ma trận $n \times m$.**]
Trong đoạn code sau,
ta thực hiện nhân ma trận trên `A` và `B`.
Ở đây, `A` là ma trận hai hàng ba cột,
và `B` là ma trận ba hàng bốn cột.
Sau khi nhân, ta thu được ma trận hai hàng bốn cột.


```python
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```


Thuật ngữ *nhân ma trận--ma trận* thường
được rút gọn thành *nhân ma trận*,
và không nên nhầm lẫn với tích Hadamard.


## Chuẩn
<a id="subsec_lin-algebra-norms"></a>

Một số toán tử hữu ích nhất trong đại số tuyến tính là *chuẩn*.
Nói không chính thức, chuẩn của vector cho ta biết nó *lớn* như thế nào.
Ví dụ, chuẩn $\ell_2$ đo
độ dài (Euclidean) của vector.
Ở đây, ta đang dùng khái niệm *kích thước* liên quan đến độ lớn của các thành phần vector
(không phải số chiều của nó).

Chuẩn là hàm $\| \cdot \|$ ánh xạ vector
sang vô hướng và thỏa mãn ba tính chất sau:

1. Với vector $\mathbf{x}$ bất kỳ, nếu ta nhân (tất cả phần tử của) vector
   với vô hướng $\alpha \in \mathbb{R}$, chuẩn của nó thay đổi theo:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. Với vector $\mathbf{x}$ và $\mathbf{y}$ bất kỳ:
   chuẩn thỏa mãn bất đẳng thức tam giác:
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. Chuẩn của vector không âm và chỉ bằng không khi vector là không:
   $$\|\mathbf{x}\| > 0 \textrm{ với mọi } \mathbf{x} \neq 0.$$

Nhiều hàm là chuẩn hợp lệ và các chuẩn khác nhau
mã hóa các khái niệm khác nhau về kích thước.
Chuẩn Euclidean mà tất cả chúng ta đã học trong hình học tiểu học
khi tính cạnh huyền của tam giác vuông
là căn bậc hai của tổng bình phương các phần tử vector.
Chính thức, đây được gọi là [**chuẩn $\ell_2$**] và được biểu diễn như sau

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**)

Phương thức `norm` tính chuẩn $\ell_2$.


```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```


[**Chuẩn $\ell_1$**] cũng phổ biến
và thước đo liên quan được gọi là khoảng cách Manhattan.
Theo định nghĩa, chuẩn $\ell_1$ tính tổng
giá trị tuyệt đối của các phần tử vector:

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

So với chuẩn $\ell_2$, nó ít nhạy cảm hơn với các ngoại lệ.
Để tính chuẩn $\ell_1$,
ta kết hợp giá trị tuyệt đối
với phép tính tổng.


```python
torch.abs(u).sum()
```


Cả chuẩn $\ell_2$ và $\ell_1$ đều là trường hợp đặc biệt
của *chuẩn $\ell_p$* tổng quát hơn:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

Đối với ma trận, vấn đề phức tạp hơn.
Suy cho cùng, ma trận có thể được xem cả là tập hợp các phần tử riêng lẻ
*và* là các đối tượng tác động lên vector và biến đổi chúng thành các vector khác.
Ví dụ, ta có thể hỏi tích ma trận--vector $\mathbf{X} \mathbf{v}$
có thể dài hơn $\mathbf{v}$ bao nhiêu lần.
Hướng suy nghĩ này dẫn đến cái gọi là chuẩn *phổ*.
Hiện tại, ta giới thiệu [***chuẩn Frobenius*,
dễ tính hơn nhiều**] và được định nghĩa là
căn bậc hai của tổng bình phương
của các phần tử ma trận:

[**$$\|\mathbf{X}\|_\textrm{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

Chuẩn Frobenius hoạt động như thể nó là
chuẩn $\ell_2$ của vector có dạng ma trận.
Gọi hàm sau sẽ tính
chuẩn Frobenius của ma trận.


```python
torch.norm(torch.ones((4, 9)))
```


Mặc dù ta không muốn đi quá xa trước,
ta đã có thể gieo một số trực giác về lý do tại sao các khái niệm này hữu ích.
Trong deep learning, ta thường đang giải quyết các bài toán tối ưu hóa:
*tối đa hóa* xác suất được gán cho dữ liệu quan sát;
*tối đa hóa* doanh thu liên quan đến mô hình gợi ý;
*tối thiểu hóa* khoảng cách giữa dự đoán
và các quan sát thực tế;
*tối thiểu hóa* khoảng cách giữa các biểu diễn
của ảnh cùng một người
trong khi *tối đa hóa* khoảng cách giữa các biểu diễn
của ảnh những người khác nhau.
Các khoảng cách này, cấu thành
mục tiêu của các thuật toán deep learning,
thường được biểu diễn dưới dạng chuẩn.


## Thảo luận

Trong phần này, ta đã ôn lại tất cả đại số tuyến tính
mà bạn cần để hiểu
một phần đáng kể của deep learning hiện đại.
Tuy nhiên, đại số tuyến tính còn nhiều hơn thế,
và nhiều trong số đó hữu ích cho machine learning.
Ví dụ, ma trận có thể được phân tích thành các nhân tử,
và các phân tích này có thể tiết lộ
cấu trúc chiều thấp trong tập dữ liệu thực tế.
Có những phân ngành machine learning riêng biệt
tập trung vào việc dùng phân tích ma trận
và các tổng quát hóa của chúng cho tensor bậc cao
để khám phá cấu trúc trong tập dữ liệu
và giải quyết các bài toán dự đoán.
Nhưng cuốn sách này tập trung vào deep learning.
Và chúng tôi tin rằng bạn sẽ có xu hướng
học nhiều toán học hơn
sau khi bạn đã thực hành
áp dụng machine learning vào tập dữ liệu thực.
Vì vậy, trong khi ta bảo lưu quyền
giới thiệu thêm toán học sau này,
ta kết thúc phần này ở đây.

Nếu bạn muốn tìm hiểu thêm về đại số tuyến tính,
có nhiều sách xuất sắc và tài nguyên trực tuyến.
Để có khóa học cấp tốc nâng cao hơn, hãy xem xét
Strang.1993, Kolter.2008, và Petersen.Pedersen.ea.2008.

Tóm lại:

* Vô hướng, vector, ma trận và tensor là
  các đối tượng toán học cơ bản được dùng trong đại số tuyến tính
  và có không, một, hai và số trục tùy ý tương ứng.
* Tensor có thể được cắt lát hoặc rút gọn dọc theo các trục được chỉ định
  thông qua chỉ mục, hoặc các phép toán như `sum` và `mean` tương ứng.
* Tích theo từng phần tử được gọi là tích Hadamard.
  Ngược lại, tích vô hướng, tích ma trận--vector và tích ma trận--ma trận
  không phải là các phép toán theo từng phần tử và nói chung trả về các đối tượng
  có shape khác với các toán hạng.
* So với tích Hadamard, nhân ma trận--ma trận
  mất thời gian tính toán đáng kể hơn (thời gian bậc ba thay vì bậc hai).
* Chuẩn nắm bắt các khái niệm khác nhau về độ lớn của vector (hoặc ma trận),
  và thường được áp dụng cho hiệu của hai vector
  để đo khoảng cách giữa chúng.
* Các chuẩn vector phổ biến bao gồm chuẩn $\ell_1$ và $\ell_2$,
  và các chuẩn ma trận phổ biến bao gồm chuẩn *phổ* và chuẩn *Frobenius*.


## Bài tập

1. Chứng minh rằng chuyển vị của chuyển vị của ma trận là chính ma trận đó: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. Cho hai ma trận $\mathbf{A}$ và $\mathbf{B}$, chứng minh rằng tổng và chuyển vị giao hoán: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Cho ma trận vuông $\mathbf{A}$ bất kỳ, liệu $\mathbf{A} + \mathbf{A}^\top$ có luôn đối xứng không? Bạn có thể chứng minh kết quả chỉ dùng kết quả của hai bài tập trước không?
1. Ta đã định nghĩa tensor `X` có shape (2, 3, 4) trong phần này. Đầu ra của `len(X)` là gì? Viết câu trả lời mà không cần cài đặt code, sau đó kiểm tra câu trả lời bằng code.
1. Với tensor `X` có shape tùy ý, liệu `len(X)` có luôn tương ứng với độ dài của một trục nào đó của `X` không? Đó là trục nào?
1. Chạy `A / A.sum(axis=1)` và xem điều gì xảy ra. Bạn có thể phân tích kết quả không?
1. Khi di chuyển giữa hai điểm ở trung tâm Manhattan, khoảng cách bạn cần đi là bao nhiêu tính theo tọa độ, tức là theo đại lộ và đường phố? Bạn có thể đi đường chéo không?
1. Xét tensor có shape (2, 3, 4). Các shape của đầu ra tổng dọc theo trục 0, 1 và 2 là gì?
1. Đưa tensor có ba trục trở lên vào hàm `linalg.norm` và quan sát đầu ra của nó. Hàm này tính gì cho tensor có shape tùy ý?
1. Xét ba ma trận lớn, giả sử $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ và $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$, được khởi tạo với các biến ngẫu nhiên Gaussian. Bạn muốn tính tích $\mathbf{A} \mathbf{B} \mathbf{C}$. Có sự khác biệt gì về dung lượng bộ nhớ và tốc độ, tùy thuộc vào việc bạn tính $(\mathbf{A} \mathbf{B}) \mathbf{C}$ hay $\mathbf{A} (\mathbf{B} \mathbf{C})$ không? Tại sao?
1. Xét ba ma trận lớn, giả sử $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ và $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$. Có sự khác biệt gì về tốc độ tùy thuộc vào việc bạn tính $\mathbf{A} \mathbf{B}$ hay $\mathbf{A} \mathbf{C}^\top$ không? Tại sao? Điều gì thay đổi nếu bạn khởi tạo $\mathbf{C} = \mathbf{B}^\top$ mà không sao chép bộ nhớ? Tại sao?
1. Xét ba ma trận, giả sử $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$. Xây dựng tensor với ba trục bằng cách xếp chồng $[\mathbf{A}, \mathbf{B}, \mathbf{C}]$. Số chiều là bao nhiêu? Cắt ra tọa độ thứ hai của trục thứ ba để lấy lại $\mathbf{B}$. Kiểm tra câu trả lời của bạn là đúng.


[Thảo luận](https://discuss.d2l.ai/t/31)
