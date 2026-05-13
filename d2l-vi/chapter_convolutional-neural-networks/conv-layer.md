# Tích chập cho Hình ảnh
<a id="sec_conv_layer"></a>

Bây giờ chúng ta đã hiểu cách các lớp tích chập hoạt động về mặt lý thuyết,
chúng ta sẵn sàng xem chúng hoạt động trong thực tế như thế nào.
Dựa trên động lực của chúng ta về mạng nơ-ron tích chập
như các kiến trúc hiệu quả để khám phá cấu trúc trong dữ liệu hình ảnh,
chúng ta sử dụng hình ảnh làm ví dụ chạy của chúng ta.


```python
from d2l import torch as d2l
import torch
from torch import nn
```


## Phép Tương quan Chéo

Nhớ lại rằng nói chính xác, các lớp tích chập
là tên gọi không chính xác, vì các phép toán mà chúng biểu đạt
được mô tả chính xác hơn là tương quan chéo.
Dựa trên mô tả của chúng ta về các lớp tích chập trong [sec_why-conv](#sec_why-conv),
trong lớp như vậy, một tensor đầu vào
và một tensor nhân được kết hợp
để tạo ra một tensor đầu ra thông qua (**phép tương quan chéo.**)

Hãy bỏ qua các kênh cho bây giờ và xem cách điều này hoạt động
với dữ liệu hai chiều và biểu diễn ẩn.
Trong [fig_correlation](#fig_correlation),
đầu vào là một tensor hai chiều
với chiều cao là 3 và chiều rộng là 3.
Chúng ta đánh dấu hình dạng của tensor là $3 \times 3$ hoặc ($3$, $3$).
Chiều cao và chiều rộng của nhân đều là 2.
Hình dạng của *cửa sổ nhân* (hoặc *cửa sổ tích chập*)
được cho bởi chiều cao và chiều rộng của nhân
(ở đây là $2 \times 2$).

![Phép tương quan chéo hai chiều. Các phần tô bóng là phần tử đầu ra đầu tiên cũng như các phần tử tensor đầu vào và nhân được sử dụng để tính toán đầu ra: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<a id="fig_correlation"></a>

Trong phép tương quan chéo hai chiều,
chúng ta bắt đầu với cửa sổ tích chập được đặt
ở góc trên bên trái của tensor đầu vào
và trượt nó qua tensor đầu vào,
cả từ trái sang phải và từ trên xuống dưới.
Khi cửa sổ tích chập trượt đến một vị trí nhất định,
tensor con đầu vào chứa trong cửa sổ đó
và tensor nhân được nhân theo phần tử
và tensor kết quả được tổng hợp lại
cho một giá trị vô hướng duy nhất.
Kết quả này cho giá trị của tensor đầu ra
tại vị trí tương ứng.
Ở đây, tensor đầu ra có chiều cao là 2 và chiều rộng là 2
và bốn phần tử được rút ra từ
phép tương quan chéo hai chiều:

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

Lưu ý rằng dọc theo mỗi trục, kích thước đầu ra
nhỏ hơn một chút so với kích thước đầu vào.
Vì nhân có chiều rộng và chiều cao lớn hơn $1$,
chúng ta chỉ có thể tính toán đúng tương quan chéo
cho các vị trí mà nhân khớp hoàn toàn trong hình ảnh,
kích thước đầu ra được cho bởi kích thước đầu vào $n_\textrm{h} \times n_\textrm{w}$
trừ đi kích thước của nhân tích chập $k_\textrm{h} \times k_\textrm{w}$
theo công thức

$$(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1).$$

Đây là trường hợp vì chúng ta cần đủ không gian
để "dịch chuyển" nhân tích chập qua hình ảnh.
Sau này chúng ta sẽ thấy cách giữ kích thước không đổi
bằng cách đệm thêm số không vào xung quanh ranh giới của hình ảnh
để có đủ không gian để dịch chuyển nhân.
Tiếp theo, chúng ta triển khai quá trình này trong hàm `corr2d`,
nhận một tensor đầu vào `X` và một tensor nhân `K`
và trả về một tensor đầu ra `Y`.


```python
def corr2d(X, K):  
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```


Chúng ta có thể xây dựng tensor đầu vào `X` và tensor nhân `K`
từ [fig_correlation](#fig_correlation)
để [**xác thực đầu ra của triển khai trên**]
của phép tương quan chéo hai chiều.

```python
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## Lớp Tích chập

Một lớp tích chập tương quan chéo đầu vào và nhân
và thêm một hệ số chặn vô hướng để tạo ra đầu ra.
Hai tham số của một lớp tích chập
là nhân và hệ số chặn vô hướng.
Khi huấn luyện các mô hình dựa trên các lớp tích chập,
chúng ta thường khởi tạo các nhân ngẫu nhiên,
giống như chúng ta làm với một lớp kết nối đầy đủ.

Bây giờ chúng ta sẵn sàng để [**triển khai một lớp tích chập hai chiều**]
dựa trên hàm `corr2d` được định nghĩa ở trên.
Trong phương thức khởi tạo `__init__`,
chúng ta khai báo `weight` và `bias` là hai tham số mô hình.
Phương thức lan truyền xuôi
gọi hàm `corr2d` và thêm hệ số chặn.


```python
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```


Trong
tích chập $h \times w$
hoặc một nhân tích chập $h \times w$,
chiều cao và chiều rộng của nhân tích chập lần lượt là $h$ và $w$.
Chúng ta cũng gọi
một lớp tích chập với một nhân tích chập $h \times w$
đơn giản là lớp tích chập $h \times w$.


## Phát hiện Cạnh Đối tượng trong Hình ảnh

Hãy dừng lại để phân tích [**một ứng dụng đơn giản của lớp tích chập:
phát hiện cạnh của một đối tượng trong hình ảnh**]
bằng cách tìm vị trí thay đổi điểm ảnh.
Đầu tiên, chúng ta xây dựng một "hình ảnh" gồm $6\times 8$ điểm ảnh.
Bốn cột giữa là đen ($0$) và phần còn lại là trắng ($1$).


Tiếp theo, chúng ta xây dựng một nhân `K` có chiều cao là 1 và chiều rộng là 2.
Khi chúng ta thực hiện phép tương quan chéo với đầu vào,
nếu các phần tử lân cận theo chiều ngang là như nhau,
đầu ra là 0. Ngược lại, đầu ra khác không.
Lưu ý rằng nhân này là trường hợp đặc biệt của toán tử sai phân hữu hạn. Tại vị trí $(i,j)$ nó tính $x_{i,j} - x_{(i+1),j}$, tức là nó tính sự khác biệt giữa các giá trị của các điểm ảnh lân cận theo chiều ngang. Đây là xấp xỉ rời rạc của đạo hàm bậc nhất theo chiều ngang. Xét cho cùng, đối với một hàm $f(i,j)$ đạo hàm của nó $-\partial_i f(i,j) = \lim_{\epsilon \to 0} \frac{f(i,j) - f(i+\epsilon,j)}{\epsilon}$. Hãy xem điều này hoạt động trong thực tế.

```python
K = d2l.tensor([[1.0, -1.0]])
```

Chúng ta sẵn sàng thực hiện phép tương quan chéo
với các đối số `X` (đầu vào của chúng ta) và `K` (nhân của chúng ta).
Như bạn có thể thấy, [**chúng ta phát hiện $1$ cho cạnh từ trắng sang đen
và $-1$ cho cạnh từ đen sang trắng.**]
Tất cả các đầu ra khác lấy giá trị $0$.

```python
Y = corr2d(X, K)
Y
```

Bây giờ chúng ta có thể áp dụng nhân cho hình ảnh được chuyển vị.
Như mong đợi, nó biến mất. [**Nhân `K` chỉ phát hiện các cạnh dọc.**]

```python
corr2d(d2l.transpose(X), K)
```

## Học Nhân

Thiết kế một bộ phát hiện cạnh bằng sai phân hữu hạn `[1, -1]` là gọn gàng
nếu chúng ta biết đây chính xác là những gì chúng ta đang tìm kiếm.
Tuy nhiên, khi chúng ta xem xét các nhân lớn hơn,
và xem xét các lớp tích chập liên tiếp,
có thể không thể chỉ định
chính xác những gì mỗi bộ lọc nên làm một cách thủ công.

Bây giờ hãy xem liệu chúng ta có thể [**học nhân tạo ra `Y` từ `X`**]
bằng cách chỉ nhìn vào các cặp đầu vào-đầu ra.
Đầu tiên chúng ta xây dựng một lớp tích chập
và khởi tạo nhân của nó như một tensor ngẫu nhiên.
Tiếp theo, trong mỗi lần lặp, chúng ta sẽ sử dụng lỗi bình phương
để so sánh `Y` với đầu ra của lớp tích chập.
Sau đó chúng ta có thể tính gradient để cập nhật nhân.
Vì đơn giản,
trong phần sau
chúng ta sử dụng lớp tích hợp sẵn
cho các lớp tích chập hai chiều
và bỏ qua hệ số chặn.


```python
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Learning rate

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
```


Lưu ý rằng lỗi đã giảm xuống giá trị nhỏ sau 10 lần lặp. Bây giờ chúng ta sẽ [**xem nhân tensor mà chúng ta đã học.**]


```python
d2l.reshape(conv2d.weight.data, (1, 2))
```


Thật vậy, tensor nhân đã học rất gần
với tensor nhân `K` mà chúng ta đã định nghĩa trước đó.

## Tương quan Chéo và Tích chập

Nhớ lại quan sát của chúng ta từ [sec_why-conv](#sec_why-conv) về sự tương ứng
giữa các phép tương quan chéo và tích chập.
Ở đây hãy tiếp tục xem xét các lớp tích chập hai chiều.
Điều gì xảy ra nếu các lớp như vậy
thực hiện các phép tích chập nghiêm ngặt
như được định nghĩa trong :eqref:`eq_2d-conv-discrete`
thay vì tương quan chéo?
Để có được đầu ra của phép *tích chập* nghiêm ngặt, chúng ta chỉ cần lật tensor nhân hai chiều cả theo chiều ngang và dọc, rồi thực hiện phép *tương quan chéo* với tensor đầu vào.

Đáng lưu ý là vì các nhân được học từ dữ liệu trong deep learning,
đầu ra của các lớp tích chập vẫn không bị ảnh hưởng
cho dù các lớp như vậy
thực hiện
các phép tích chập nghiêm ngặt
hay các phép tương quan chéo.

Để minh họa điều này, giả sử rằng một lớp tích chập thực hiện *tương quan chéo* và học nhân trong [fig_correlation](#fig_correlation), được ký hiệu ở đây là ma trận $\mathbf{K}$.
Giả sử các điều kiện khác không thay đổi,
khi lớp này thay vào đó thực hiện *tích chập* nghiêm ngặt,
nhân đã học $\mathbf{K}'$ sẽ giống như $\mathbf{K}$
sau khi $\mathbf{K}'$ được
lật cả theo chiều ngang và dọc.
Nghĩa là,
khi lớp tích chập
thực hiện *tích chập* nghiêm ngặt
cho đầu vào trong [fig_correlation](#fig_correlation)
và $\mathbf{K}'$,
cùng đầu ra trong [fig_correlation](#fig_correlation)
(tương quan chéo của đầu vào và $\mathbf{K}$)
sẽ được thu được.

Để phù hợp với thuật ngữ tiêu chuẩn trong tài liệu deep learning,
chúng ta sẽ tiếp tục gọi phép tương quan chéo
là phép tích chập mặc dù, nói chính xác, nó hơi khác.
Hơn nữa,
chúng ta sử dụng thuật ngữ *phần tử* để chỉ
một mục nhập (hoặc thành phần) của bất kỳ tensor nào biểu diễn một biểu diễn lớp hoặc một nhân tích chập.


## Bản đồ Đặc trưng và Trường Tiếp nhận

Như được mô tả trong [subsec_why-conv-channels](#subsec_why-conv-channels),
đầu ra lớp tích chập trong
[fig_correlation](#fig_correlation)
đôi khi được gọi là *bản đồ đặc trưng*,
vì nó có thể được coi là
các biểu diễn (đặc trưng) đã học
trong các chiều không gian (ví dụ: chiều rộng và chiều cao)
cho lớp tiếp theo.
Trong CNN,
đối với bất kỳ phần tử $x$ nào của một lớp nào đó,
*trường tiếp nhận* của nó chỉ đến
tất cả các phần tử (từ tất cả các lớp trước)
có thể ảnh hưởng đến việc tính toán $x$
trong quá trình lan truyền xuôi.
Lưu ý rằng trường tiếp nhận
có thể lớn hơn kích thước thực tế của đầu vào.

Hãy tiếp tục sử dụng [fig_correlation](#fig_correlation) để giải thích trường tiếp nhận.
Cho nhân tích chập $2 \times 2$,
trường tiếp nhận của phần tử đầu ra được tô bóng (có giá trị $19$)
là
bốn phần tử trong phần tô bóng của đầu vào.
Bây giờ hãy ký hiệu đầu ra $2 \times 2$
là $\mathbf{Y}$
và xem xét một CNN sâu hơn
với một lớp tích chập $2 \times 2$ bổ sung nhận $\mathbf{Y}$
làm đầu vào của nó, xuất ra
một phần tử duy nhất $z$.
Trong trường hợp này,
trường tiếp nhận của $z$
trên $\mathbf{Y}$ bao gồm tất cả bốn phần tử của $\mathbf{Y}$,
trong khi
trường tiếp nhận
trên đầu vào bao gồm tất cả chín phần tử đầu vào.
Vì vậy,
khi bất kỳ phần tử nào trong bản đồ đặc trưng
cần một trường tiếp nhận lớn hơn
để phát hiện các đặc trưng đầu vào trên một vùng rộng hơn,
chúng ta có thể xây dựng một mạng sâu hơn.


Trường tiếp nhận lấy tên từ sinh lý thần kinh.
Một loạt thí nghiệm trên nhiều loài động vật sử dụng các kích thích khác nhau
[Hubel.Wiesel.1959, Hubel.Wiesel.1962, Hubel.Wiesel.1968] đã khám phá phản ứng của cái được gọi là vỏ
thị giác đối với các kích thích nói trên. Nhìn chung họ thấy rằng các cấp thấp hơn phản ứng với các cạnh và các hình dạng liên quan. Sau đó, Field.1987 đã minh họa hiệu ứng này trên các hình ảnh tự nhiên với, chỉ có thể gọi là, nhân tích chập.
Chúng ta in lại một hình chính trong [field_visual](#field_visual) để minh họa những điểm tương đồng nổi bật.

![Hình và chú thích lấy từ Field.1987: Ví dụ về mã hóa với sáu kênh khác nhau. (Trái) Ví dụ về sáu loại cảm biến liên quan đến mỗi kênh. (Phải) Tích chập của hình ảnh ở (Giữa) với sáu cảm biến được hiển thị ở (Trái). Phản ứng của các cảm biến riêng lẻ được xác định bằng cách lấy mẫu các hình ảnh đã lọc này ở khoảng cách tỷ lệ với kích thước của cảm biến (hiển thị với các dấu chấm). Sơ đồ này hiển thị phản ứng của chỉ các cảm biến đối xứng chẵn.](../img/field-visual.png)
<a id="field_visual"></a>

Hóa ra, mối quan hệ này thậm chí còn áp dụng cho các đặc trưng được tính toán bởi các lớp sâu hơn của các mạng được huấn luyện trên các tác vụ phân loại hình ảnh, như được chứng minh trong, ví dụ, Kuzovkin.Vicente.Petton.ea.2018. Chỉ cần nói, các phép tích chập đã chứng minh là một công cụ cực kỳ mạnh mẽ cho thị giác máy tính, cả trong sinh học và trong code. Do đó, không ngạc nhiên (nhìn lại) khi chúng báo trước thành công gần đây trong deep learning.

## Tóm tắt

Tính toán cốt lõi cần thiết cho một lớp tích chập là phép tương quan chéo. Chúng ta đã thấy rằng một vòng lặp for lồng nhau đơn giản là tất cả những gì cần thiết để tính giá trị của nó. Nếu chúng ta có nhiều kênh đầu vào và đầu ra, chúng ta đang thực hiện một phép toán ma trận-ma trận giữa các kênh. Có thể thấy, tính toán là đơn giản và quan trọng nhất, rất *cục bộ*. Điều này cho phép tối ưu hóa phần cứng đáng kể và nhiều kết quả gần đây trong thị giác máy tính chỉ có thể thực hiện được nhờ điều đó. Xét cho cùng, điều này có nghĩa là các nhà thiết kế chip có thể đầu tư vào tính toán nhanh thay vì bộ nhớ khi tối ưu hóa cho các phép tích chập. Mặc dù điều này có thể không dẫn đến các thiết kế tối ưu cho các ứng dụng khác, nhưng nó mở ra cánh cửa cho thị giác máy tính phổ biến và giá cả phải chăng.

Về bản thân các phép tích chập, chúng có thể được sử dụng cho nhiều mục đích, ví dụ phát hiện cạnh và đường, làm mờ hình ảnh, hoặc làm sắc nét chúng. Quan trọng nhất, không cần thiết phải phát minh các bộ lọc phù hợp. Thay vào đó, chúng ta chỉ cần *học* chúng từ dữ liệu. Điều này thay thế các phương pháp phỏng đoán kỹ thuật đặc trưng bằng thống kê dựa trên bằng chứng. Cuối cùng, và thú vị thay, các bộ lọc này không chỉ có lợi cho việc xây dựng các mạng sâu mà chúng còn tương ứng với trường tiếp nhận và bản đồ đặc trưng trong não. Điều này cho chúng ta sự tự tin rằng chúng ta đang đi đúng hướng.

## Bài tập

1. Xây dựng một hình ảnh `X` với các cạnh chéo.
    1. Điều gì xảy ra nếu bạn áp dụng nhân `K` trong phần này cho nó?
    1. Điều gì xảy ra nếu bạn chuyển vị `X`?
    1. Điều gì xảy ra nếu bạn chuyển vị `K`?
1. Thiết kế một số nhân thủ công.
    1. Cho một vectơ hướng $\mathbf{v} = (v_1, v_2)$, rút ra một nhân phát hiện cạnh phát hiện
       các cạnh vuông góc với $\mathbf{v}$, tức là các cạnh theo hướng $(v_2, -v_1)$.
    1. Rút ra một toán tử sai phân hữu hạn cho đạo hàm bậc hai. Kích thước tối thiểu
       của nhân tích chập liên quan đến nó là bao nhiêu? Những cấu trúc nào trong hình ảnh phản ứng mạnh nhất với nó?
    1. Bạn sẽ thiết kế một nhân làm mờ như thế nào? Tại sao bạn muốn sử dụng nhân như vậy?
    1. Kích thước tối thiểu của nhân để có được đạo hàm bậc $d$ là bao nhiêu?
1. Khi bạn cố gắng tự động tìm gradient cho lớp `Conv2D` mà chúng ta đã tạo, bạn thấy loại thông báo lỗi nào?
1. Bạn biểu diễn phép tương quan chéo như một phép nhân ma trận như thế nào bằng cách thay đổi các tensor đầu vào và nhân?


[Discussions](https://discuss.d2l.ai/t/66)
