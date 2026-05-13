# Attention Pooling Theo Độ Tương Đồng

<a id="sec_attention-pooling"></a>

Bây giờ chúng ta đã giới thiệu các thành phần chính của cơ chế attention, hãy sử dụng chúng trong một thiết lập khá cổ điển, cụ thể là hồi quy và phân loại qua ước lượng mật độ kernel [Nadaraya.1964, Watson.1964]. Bài giải thích thêm này đơn giản chỉ cung cấp nền tảng bổ sung: hoàn toàn tùy chọn và có thể bỏ qua nếu cần thiết.
Về cốt lõi, các ước lượng Nadaraya--Watson dựa vào một số kernel độ tương đồng $\alpha(\mathbf{q}, \mathbf{k})$ liên kết các truy vấn $\mathbf{q}$ với các khóa $\mathbf{k}$. Một số kernel phổ biến là

$$\begin{aligned}
\alpha(\mathbf{q}, \mathbf{k}) & = \exp\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2 \right) && \textrm{Gaussian;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = 1 \textrm{ nếu } \|\mathbf{q} - \mathbf{k}\| \leq 1 && \textrm{Boxcar;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = \mathop{\mathrm{max}}\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) && \textrm{Epanechikov.}
\end{aligned}
$$

Có nhiều lựa chọn khác mà chúng ta có thể chọn. Xem [bài viết Wikipedia](https://en.wikipedia.org/wiki/Kernel_(statistics)) để có cái nhìn sâu hơn và cách lựa chọn kernel liên quan đến ước lượng mật độ kernel, đôi khi còn được gọi là *Cửa sổ Parzen* [parzen1957consistent]. Tất cả các kernel đều là heuristic và có thể được điều chỉnh. Ví dụ, chúng ta có thể điều chỉnh độ rộng, không chỉ trên cơ sở toàn cục mà thậm chí trên cơ sở từng tọa độ. Dù sao, tất cả chúng đều dẫn đến phương trình sau cho cả hồi quy và phân loại:

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}.$$

Trong trường hợp hồi quy (vô hướng) với các quan sát $(\mathbf{x}_i, y_i)$ cho đặc trưng và nhãn tương ứng, $\mathbf{v}_i = y_i$ là vô hướng, $\mathbf{k}_i = \mathbf{x}_i$ là vector, và truy vấn $\mathbf{q}$ biểu thị vị trí mới mà $f$ nên được đánh giá. Trong trường hợp phân loại (đa lớp), chúng ta sử dụng mã hóa one-hot của $y_i$ để thu được $\mathbf{v}_i$. Một trong những thuộc tính tiện lợi của ước lượng này là nó không yêu cầu huấn luyện. Hơn nữa, nếu chúng ta thu hẹp kernel một cách phù hợp với lượng dữ liệu ngày càng tăng, cách tiếp cận là nhất quán [mack1982weak], tức là nó sẽ hội tụ đến một số nghiệm tối ưu về mặt thống kê. Hãy bắt đầu bằng cách kiểm tra một số kernel.


```python
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

d2l.use_svg_display()
```


## [**Các Kernel và Dữ Liệu**]

Tất cả các kernel $\alpha(\mathbf{k}, \mathbf{q})$ được định nghĩa trong phần này đều là *bất biến tịnh tiến và quay*; tức là, nếu chúng ta dịch chuyển và quay $\mathbf{k}$ và $\mathbf{q}$ theo cùng một cách, giá trị của $\alpha$ vẫn không thay đổi. Để đơn giản, chúng ta chọn các tham số vô hướng $k, q \in \mathbb{R}$ và chọn khóa $k = 0$ là gốc tọa độ. Điều này mang lại:

```python
# Define some kernels
def gaussian(x):
    return d2l.exp(-x**2 / 2)

def boxcar(x):
    return d2l.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x
 
if tab.selected('pytorch'):
    def epanechikov(x):
        return torch.max(1 - d2l.abs(x), torch.zeros_like(x))
if tab.selected('mxnet'):
    def epanechikov(x):
        return np.maximum(1 - d2l.abs(x), 0)
if tab.selected('tensorflow'):
    def epanechikov(x):
        return tf.maximum(1 - d2l.abs(x), 0)
if tab.selected('jax'):
    def epanechikov(x):
        return jnp.maximum(1 - d2l.abs(x), 0)
```

```python
fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))

kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
x = d2l.arange(-2.5, 2.5, 0.1)
for kernel, name, ax in zip(kernels, names, axes):
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        ax.plot(d2l.numpy(x), d2l.numpy(kernel(x)))
    if tab.selected('jax'):
        ax.plot(x, kernel(x))
    ax.set_xlabel(name)

d2l.plt.show()
```

Các kernel khác nhau tương ứng với các khái niệm khác nhau về phạm vi và độ mượt. Ví dụ, kernel boxcar chỉ chú ý đến các quan sát trong khoảng cách $1$ (hoặc một siêu tham số được định nghĩa khác) và làm vậy một cách không phân biệt.

Để xem ước lượng Nadaraya--Watson trong hoạt động, hãy định nghĩa một số dữ liệu huấn luyện. Trong phần sau chúng ta sử dụng sự phụ thuộc

$$y_i = 2\sin(x_i) + x_i + \epsilon,$$

trong đó $\epsilon$ được lấy từ phân phối chuẩn với trung bình không và phương sai đơn vị. Chúng ta lấy 40 ví dụ huấn luyện.

```python
def f(x):
    return 2 * d2l.sin(x) + x

n = 40
if tab.selected('pytorch'):
    x_train, _ = torch.sort(d2l.rand(n) * 5)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('mxnet'):
    x_train = np.sort(d2l.rand(n) * 5, axis=None)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('tensorflow'):
    x_train = tf.sort(d2l.rand((n,1)) * 5, 0)
    y_train = f(x_train) + d2l.normal((n, 1))
if tab.selected('jax'):
    x_train = jnp.sort(jax.random.uniform(d2l.get_key(), (n,)) * 5)
    y_train = f(x_train) + jax.random.normal(d2l.get_key(), (n,))
x_val = d2l.arange(0, 5, 0.1)
y_val = f(x_val)
```

## [**Attention Pooling qua Hồi Quy Nadaraya--Watson**]

Bây giờ chúng ta đã có dữ liệu và kernel, tất cả những gì chúng ta cần là một hàm tính toán các ước lượng hồi quy kernel. Lưu ý rằng chúng ta cũng muốn thu được các trọng số kernel tương đối để thực hiện một số chẩn đoán nhỏ. Do đó trước tiên chúng ta tính toán kernel giữa tất cả các đặc trưng huấn luyện (hiệp biến) `x_train` và tất cả các đặc trưng kiểm định `x_val`. Điều này cho ra một ma trận, mà sau đó chúng ta chuẩn hóa. Khi nhân với nhãn huấn luyện `y_train` chúng ta thu được các ước lượng.

Nhớ lại attention pooling trong :eqref:`eq_attention_pooling`. Hãy để mỗi đặc trưng kiểm định là một truy vấn, và mỗi cặp đặc trưng--nhãn huấn luyện là một cặp khóa--giá trị. Kết quả là, các trọng số kernel tương đối đã chuẩn hóa (`attention_w` bên dưới) là *trọng số attention*.

```python
def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = d2l.reshape(x_train, (-1, 1)) - d2l.reshape(x_val, (1, -1))
    # Each column/row corresponds to each query/key
    k = d2l.astype(kernel(dists), d2l.float32)
    # Normalization over keys for each query
    attention_w = k / d2l.reduce_sum(k, 0)
    if tab.selected('pytorch'):
        y_hat = y_train@attention_w
    if tab.selected('mxnet'):
        y_hat = np.dot(y_train, attention_w)
    if tab.selected('tensorflow'):
        y_hat = d2l.transpose(d2l.transpose(y_train)@attention_w)
    if tab.selected('jax'):
        y_hat = y_train@attention_w
    return y_hat, attention_w
```

Hãy xem loại ước lượng mà các kernel khác nhau tạo ra.

```python
def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(attention_w), cmap='Reds')
            if tab.selected('jax'):
                pcm = ax.imshow(attention_w, cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)
```

```python
plot(x_train, y_train, x_val, y_val, kernels, names)
```

Điều đầu tiên nổi bật là ba kernel không tầm thường (Gaussian, Boxcar và Epanechikov) tạo ra các ước lượng khá khả dụng không quá xa so với hàm thực. Chỉ có kernel hằng số dẫn đến ước lượng tầm thường $f(x) = \frac{1}{n} \sum_i y_i$ tạo ra kết quả khá không thực tế. Hãy kiểm tra các trọng số attention kỹ hơn một chút:

```python
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

Trực quan hóa rõ ràng cho thấy tại sao các ước lượng cho Gaussian, Boxcar và Epanechikov rất giống nhau: suy cho cùng, chúng được dẫn xuất từ các trọng số attention rất giống nhau, mặc dù có dạng hàm kernel khác nhau. Điều này đặt ra câu hỏi liệu điều này có phải luôn như vậy không.

## [**Điều Chỉnh Attention Pooling**]

Chúng ta có thể thay thế kernel Gaussian bằng một kernel có độ rộng khác. Tức là, chúng ta có thể sử dụng
$\alpha(\mathbf{q}, \mathbf{k}) = \exp\left(-\frac{1}{2 \sigma^2} \|\mathbf{q} - \mathbf{k}\|^2 \right)$ trong đó $\sigma^2$ xác định độ rộng của kernel. Hãy xem điều này có ảnh hưởng đến kết quả không.

```python
sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]

def gaussian_with_width(sigma): 
    return (lambda x: d2l.exp(-x**2 / (2*sigma**2)))

kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot(x_train, y_train, x_val, y_val, kernels, names)
```

Rõ ràng là kernel càng hẹp thì ước lượng càng kém mượt mà. Đồng thời, nó thích nghi tốt hơn với các biến đổi cục bộ. Hãy nhìn vào các trọng số attention tương ứng.

```python
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

Như chúng ta mong đợi, kernel càng hẹp thì phạm vi trọng số attention lớn càng hẹp. Cũng rõ ràng rằng việc chọn cùng một độ rộng có thể không phải là lý tưởng. Trên thực tế, Silverman86 đề xuất một heuristic phụ thuộc vào mật độ cục bộ. Nhiều "thủ thuật" hơn như vậy đã được đề xuất. Ví dụ, norelli2022asif sử dụng kỹ thuật nội suy láng giềng gần nhất tương tự để thiết kế các biểu diễn ảnh và văn bản đa phương thức.

Người đọc tinh tế có thể thắc mắc tại sao chúng ta cung cấp phần giải thích sâu này cho một phương pháp đã hơn nửa thế kỷ tuổi. Thứ nhất, đây là một trong những tiền thân sớm nhất của các cơ chế attention hiện đại. Thứ hai, nó rất tốt cho việc trực quan hóa. Thứ ba, và quan trọng không kém, nó thể hiện giới hạn của các cơ chế attention thủ công. Một chiến lược tốt hơn nhiều là *học* cơ chế, bằng cách học các biểu diễn cho các truy vấn và khóa. Đây là điều chúng ta sẽ bắt đầu trong các phần sau.


## Tóm Tắt

Hồi quy kernel Nadaraya--Watson là tiền thân sớm của các cơ chế attention hiện tại.
Nó có thể được sử dụng trực tiếp với ít hoặc không cần huấn luyện hoặc điều chỉnh, cho cả phân loại hoặc hồi quy.
Trọng số attention được gán theo độ tương đồng (hoặc khoảng cách) giữa truy vấn và khóa, và theo số lượng quan sát tương tự có sẵn.

## Bài Tập

1. Ước lượng mật độ cửa sổ Parzen được cho bởi $\hat{p}(\mathbf{x}) = \frac{1}{n} \sum_i k(\mathbf{x}, \mathbf{x}_i)$. Chứng minh rằng với phân loại nhị phân, hàm $\hat{p}(\mathbf{x}, y=1) - \hat{p}(\mathbf{x}, y=-1)$, như thu được bằng cửa sổ Parzen, tương đương với phân loại Nadaraya--Watson.
1. Lập trình gradient descent ngẫu nhiên để học giá trị tốt cho độ rộng kernel trong hồi quy Nadaraya--Watson.
    1. Điều gì xảy ra nếu bạn chỉ sử dụng các ước lượng trên để tối thiểu hóa $(f(\mathbf{x_i}) - y_i)^2$ trực tiếp? Gợi ý: $y_i$ là một phần của các số hạng được sử dụng để tính $f$.
    1. Loại bỏ $(\mathbf{x}_i, y_i)$ khỏi ước lượng cho $f(\mathbf{x}_i)$ và tối ưu hóa trên độ rộng kernel. Bạn có còn quan sát thấy quá khớp không?
1. Giả sử rằng tất cả $\mathbf{x}$ nằm trên mặt cầu đơn vị, tức là tất cả đều thỏa mãn $\|\mathbf{x}\| = 1$. Bạn có thể đơn giản hóa số hạng $\|\mathbf{x} - \mathbf{x}_i\|^2$ trong hàm mũ không? Gợi ý: chúng ta sẽ thấy sau này rằng điều này liên quan rất chặt chẽ đến attention tích vô hướng.
1. Nhớ lại rằng mack1982weak đã chứng minh rằng ước lượng Nadaraya--Watson là nhất quán. Bạn nên giảm tỷ lệ cho cơ chế attention nhanh như thế nào khi bạn có thêm dữ liệu? Cung cấp một số trực giác cho câu trả lời của bạn. Điều đó có phụ thuộc vào chiều của dữ liệu không? Như thế nào?


[Thảo luận](https://discuss.d2l.ai/t/1599)
