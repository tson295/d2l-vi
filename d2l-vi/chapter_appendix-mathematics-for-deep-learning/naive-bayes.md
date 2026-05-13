# Naive Bayes
<a id="sec_naive_bayes"></a>

Trong các phần trước, chúng ta đã học về lý thuyết xác suất và biến ngẫu nhiên. Để đưa lý thuyết này vào sử dụng, hãy giới thiệu bộ phân loại *naive Bayes*. Bộ phân loại này chỉ dùng các nền tảng xác suất để cho phép ta thực hiện phân loại chữ số.

Học là việc đưa ra các giả định. Nếu muốn phân loại một ví dụ dữ liệu mới mà ta chưa từng thấy trước đó, ta phải đưa ra một số giả định về việc các ví dụ dữ liệu nào giống nhau. Bộ phân loại naive Bayes, một thuật toán phổ biến và rõ ràng đáng kể, giả định tất cả các đặc trưng độc lập với nhau để đơn giản hóa phép tính. Trong phần này, ta sẽ áp dụng mô hình này để nhận dạng ký tự trong ảnh.

```python
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```python
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```python
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## Nhận Dạng Ký Tự Quang Học

MNIST [LeCun.Bottou.Bengio.ea.1998] là một trong những tập dữ liệu được sử dụng rộng rãi. Nó chứa 60.000 ảnh để huấn luyện và 10.000 ảnh để kiểm định. Mỗi ảnh chứa một chữ số viết tay từ 0 đến 9. Nhiệm vụ là phân loại từng ảnh vào chữ số tương ứng.

Gluon cung cấp một lớp `MNIST` trong mô-đun `data.vision` để tự động tải tập dữ liệu từ Internet. Sau đó, Gluon sẽ dùng bản sao cục bộ đã tải xuống. Ta chỉ định mình đang yêu cầu tập huấn luyện hay tập kiểm tra bằng cách đặt giá trị của tham số `train` lần lượt thành `True` hoặc `False`. Mỗi ảnh là một ảnh thang xám có cả chiều rộng và chiều cao bằng $28$ với hình dạng ($28$,$28$,$1$). Ta dùng một phép biến đổi tùy chỉnh để loại bỏ chiều kênh cuối cùng. Ngoài ra, tập dữ liệu biểu diễn mỗi pixel bằng một số nguyên không dấu $8$-bit. Ta lượng tử hóa chúng thành các đặc trưng nhị phân để đơn giản hóa bài toán.

```python
#@tab mxnet
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```python
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```python
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# Original pixel values of MNIST range from 0-255 (as the digits are stored as
# uint8). For this section, pixel values that are greater than 128 (in the
# original image) are converted to 1 and values that are less than 128 are
# converted to 0. See section 18.9.2 and 18.9.3 for why
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

Ta có thể truy cập một ví dụ cụ thể, chứa ảnh và nhãn tương ứng.

```python
#@tab mxnet
image, label = mnist_train[2]
image.shape, label
```

```python
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```python
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

Ví dụ của ta, được lưu ở đây trong biến `image`, tương ứng với một ảnh có chiều cao và chiều rộng là $28$ pixel.

```python
#@tab all
image.shape, image.dtype
```

Code của ta lưu nhãn của mỗi ảnh dưới dạng một đại lượng vô hướng. Kiểu của nó là số nguyên $32$-bit.

```python
#@tab mxnet
label, type(label), label.dtype
```

```python
#@tab pytorch
label, type(label)
```

```python
#@tab tensorflow
label.numpy(), label.dtype
```

Ta cũng có thể truy cập nhiều ví dụ cùng lúc.

```python
#@tab mxnet
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```python
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```python
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

Hãy trực quan hóa các ví dụ này.

```python
#@tab all
d2l.show_images(images, 2, 9);
```

## Mô Hình Xác Suất Cho Phân Loại

Trong một tác vụ phân loại, ta ánh xạ một ví dụ vào một hạng mục. Ở đây, một ví dụ là ảnh thang xám $28\times 28$, và một hạng mục là một chữ số. (Xem [sec_softmax](#sec_softmax) để có giải thích chi tiết hơn.)
Một cách tự nhiên để biểu diễn tác vụ phân loại là thông qua câu hỏi xác suất: nhãn nào có khả năng nhất khi biết các đặc trưng (tức các pixel ảnh)? Ký hiệu $\mathbf x\in\mathbb R^d$ là các đặc trưng của ví dụ và $y\in\mathbb R$ là nhãn. Ở đây đặc trưng là các pixel ảnh, trong đó ta có thể định dạng lại một ảnh $2$ chiều thành một vector sao cho $d=28^2=784$, và nhãn là các chữ số.
Xác suất của nhãn khi biết các đặc trưng là $p(y  \mid  \mathbf{x})$. Nếu ta có thể tính các xác suất này, tức $p(y  \mid  \mathbf{x})$ cho $y=0, \ldots,9$ trong ví dụ của ta, thì bộ phân loại sẽ xuất ra dự đoán $\hat{y}$ được cho bởi biểu thức:

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

Đáng tiếc, điều này đòi hỏi ta ước lượng $p(y  \mid  \mathbf{x})$ cho mọi giá trị của $\mathbf{x} = x_1, ..., x_d$. Hãy tưởng tượng mỗi đặc trưng có thể nhận một trong $2$ giá trị. Ví dụ, đặc trưng $x_1 = 1$ có thể biểu thị rằng từ apple xuất hiện trong một tài liệu cho trước, còn $x_1 = 0$ biểu thị rằng nó không xuất hiện. Nếu ta có $30$ đặc trưng nhị phân như vậy, điều đó có nghĩa là ta cần sẵn sàng phân loại bất kỳ trong số $2^{30}$ (hơn 1 tỷ!) giá trị có thể của vector đầu vào $\mathbf{x}$.

Hơn nữa, việc học nằm ở đâu? Nếu ta cần thấy từng ví dụ có thể có để dự đoán nhãn tương ứng, thì ta không thực sự học một mẫu hình mà chỉ ghi nhớ tập dữ liệu.

## Bộ Phân Loại Naive Bayes

May mắn thay, bằng cách đưa ra một số giả định về độc lập có điều kiện, ta có thể đưa vào một thiên kiến quy nạp và xây dựng một mô hình có khả năng tổng quát hóa từ một tập ví dụ huấn luyện tương đối khiêm tốn. Để bắt đầu, hãy dùng định lý Bayes để biểu diễn bộ phân loại là

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

Lưu ý rằng mẫu số là hạng chuẩn hóa $p(\mathbf{x})$, không phụ thuộc vào giá trị của nhãn $y$. Do đó, ta chỉ cần quan tâm đến việc so sánh tử số giữa các giá trị khác nhau của $y$. Ngay cả nếu việc tính mẫu số là bất khả thi, ta vẫn có thể bỏ qua nó, miễn là ta có thể đánh giá tử số. May mắn thay, ngay cả nếu muốn khôi phục hằng số chuẩn hóa, ta cũng có thể làm được. Ta luôn có thể khôi phục hạng chuẩn hóa vì $\sum_y p(y  \mid  \mathbf{x}) = 1$.

Bây giờ, hãy tập trung vào $p( \mathbf{x}  \mid  y)$. Dùng quy tắc chuỗi của xác suất, ta có thể biểu diễn hạng $p( \mathbf{x}  \mid  y)$ là

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

Tự thân biểu thức này chưa giúp ta tiến xa hơn. Ta vẫn phải ước lượng khoảng $2^d$ tham số. Tuy nhiên, nếu ta giả định rằng *các đặc trưng độc lập có điều kiện với nhau khi biết nhãn*, thì tình hình đột nhiên tốt hơn nhiều, vì hạng này đơn giản hóa thành $\prod_i p(x_i  \mid  y)$, cho ta bộ dự đoán

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

Nếu ta có thể ước lượng $p(x_i=1  \mid  y)$ cho mọi $i$ và $y$, rồi lưu giá trị của nó trong $P_{xy}[i, y]$, ở đây $P_{xy}$ là một ma trận $d\times n$ với $n$ là số lớp và $y\in\{1, \ldots, n\}$, thì ta cũng có thể dùng điều này để ước lượng $p(x_i = 0 \mid y)$, tức là

$$
p(x_i = t_i \mid y) =
\begin{cases}
    P_{xy}[i, y] & \textrm{for } t_i=1 ;\\
    1 - P_{xy}[i, y] & \textrm{for } t_i = 0 .
\end{cases}
$$

Ngoài ra, ta ước lượng $p(y)$ cho mọi $y$ và lưu nó trong $P_y[y]$, với $P_y$ là một vector độ dài $n$. Khi đó, với bất kỳ ví dụ mới nào $\mathbf t = (t_1, t_2, \ldots, t_d)$, ta có thể tính

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$

cho bất kỳ $y$ nào. Vì vậy giả định độc lập có điều kiện đã đưa độ phức tạp của mô hình từ phụ thuộc mũ theo số đặc trưng $\mathcal{O}(2^dn)$ xuống phụ thuộc tuyến tính, tức $\mathcal{O}(dn)$.


## Huấn Luyện

Vấn đề bây giờ là ta không biết $P_{xy}$ và $P_y$. Vì vậy trước hết ta cần ước lượng các giá trị của chúng từ một số dữ liệu huấn luyện. Đây là việc *huấn luyện* mô hình. Ước lượng $P_y$ không quá khó. Vì ta chỉ xử lý $10$ lớp, ta có thể đếm số lần xuất hiện $n_y$ cho từng chữ số và chia cho tổng lượng dữ liệu $n$. Chẳng hạn, nếu chữ số 8 xuất hiện $n_8 = 5,800$ lần và ta có tổng cộng $n = 60,000$ ảnh, ước lượng xác suất là $p(y=8) = 0.0967$.

```python
#@tab mxnet
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```python
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```python
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

Bây giờ đến phần hơi khó hơn, $P_{xy}$. Vì ta chọn ảnh đen trắng, $p(x_i  \mid  y)$ biểu thị xác suất pixel $i$ được bật cho lớp $y$. Giống như trước, ta có thể đi đếm số lần $n_{iy}$ sao cho một biến cố xảy ra và chia cho tổng số lần xuất hiện của $y$, tức $n_y$. Nhưng có một điều hơi đáng lo: một số pixel có thể không bao giờ đen (ví dụ, với các ảnh được cắt gọn tốt, các pixel ở góc có thể luôn trắng). Một cách thuận tiện để các nhà thống kê xử lý vấn đề này là thêm số đếm giả vào mọi trường hợp xuất hiện. Do đó, thay vì $n_{iy}$, ta dùng $n_{iy}+1$, và thay vì $n_y$, ta dùng $n_{y}+2$ (vì có hai giá trị có thể mà pixel $i$ có thể nhận: nó có thể đen hoặc trắng). Điều này còn được gọi là *làm trơn Laplace*. Nó có vẻ tùy tiện, tuy nhiên có thể được thúc đẩy từ một góc nhìn Bayes bằng mô hình Beta-binomial.

```python
#@tab mxnet
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```python
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```python
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 2), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

Bằng cách trực quan hóa các xác suất $10\times 28\times 28$ này (cho từng pixel của từng lớp), ta có thể thu được các chữ số trông giống dạng trung bình.

Bây giờ ta có thể dùng :eqref:`eq_naive_bayes_estimation` để dự đoán một ảnh mới. Cho $\mathbf x$, các hàm sau tính $p(\mathbf x \mid y)p(y)$ cho mọi $y$.

```python
#@tab mxnet
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```python
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```python
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

Điều này sai nghiêm trọng! Để tìm hiểu vì sao, hãy nhìn vào các xác suất theo từng pixel. Chúng thường là các số nằm giữa $0.001$ và $1$. Ta đang nhân $784$ số như vậy. Tại thời điểm này, đáng nói rằng ta đang tính các số này trên máy tính, do đó với một khoảng cố định cho số mũ. Điều xảy ra là ta gặp *tràn dưới số học*, tức việc nhân tất cả các số nhỏ dẫn đến một thứ còn nhỏ hơn nữa cho đến khi nó bị làm tròn xuống không. Ta đã thảo luận điều này như một vấn đề lý thuyết trong [sec_maximum_likelihood](#sec_maximum_likelihood), nhưng ở đây ta thấy hiện tượng rõ ràng trong thực tế.

Như đã thảo luận trong phần đó, ta khắc phục bằng cách dùng thực tế rằng $\log a b = \log a + \log b$, tức là ta chuyển sang cộng các logarit.
Ngay cả nếu cả $a$ và $b$ đều là các số nhỏ, các giá trị logarit vẫn nên nằm trong một khoảng phù hợp.

```python
#@tab mxnet
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```python
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```python
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

Vì logarit là một hàm tăng, ta có thể viết lại :eqref:`eq_naive_bayes_estimation` thành

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

Ta có thể cài đặt phiên bản ổn định sau:

```python
#@tab mxnet
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```python
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```python
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

Bây giờ ta có thể kiểm tra xem dự đoán có đúng không.

```python
#@tab mxnet
# Convert label which is a scalar tensor of int32 dtype to a Python scalar
# integer for comparison
py.argmax(axis=0) == int(label)
```

```python
#@tab pytorch
py.argmax(dim=0) == label
```

```python
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
```

Nếu bây giờ ta dự đoán một vài ví dụ kiểm định, ta có thể thấy bộ phân loại Bayes hoạt động khá tốt.

```python
#@tab mxnet
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```python
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item()
            for x in X]

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```python
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

Cuối cùng, hãy tính độ chính xác tổng thể của bộ phân loại.

```python
#@tab mxnet
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```python
#@tab pytorch
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```python
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# Validation accuracy
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

Các mạng deep learning hiện đại đạt tỷ lệ lỗi dưới $0.01$. Hiệu năng tương đối kém ở đây là do các giả định thống kê không đúng mà ta đã đưa vào mô hình: ta giả định rằng từng pixel *độc lập* được sinh ra, chỉ phụ thuộc vào nhãn. Điều này rõ ràng không phải cách con người viết chữ số, và giả định sai này đã dẫn đến thất bại của bộ phân loại (Bayes) quá ngây thơ của ta.

## Tóm Tắt
* Dùng quy tắc Bayes, ta có thể tạo một bộ phân loại bằng cách giả định tất cả các đặc trưng quan sát được là độc lập.
* Bộ phân loại này có thể được huấn luyện trên một tập dữ liệu bằng cách đếm số lần xuất hiện của các tổ hợp nhãn và giá trị pixel.
* Bộ phân loại này từng là chuẩn vàng trong nhiều thập kỷ cho các tác vụ như phát hiện thư rác.

## Bài Tập
1. Xét tập dữ liệu $[[0,0], [0,1], [1,0], [1,1]]$ với nhãn được cho bởi XOR của hai phần tử $[0,1,1,0]$. Các xác suất cho một bộ phân loại Naive Bayes xây dựng trên tập dữ liệu này là gì? Nó có phân loại thành công các điểm của ta không? Nếu không, những giả định nào bị vi phạm?
1. Giả sử ta không dùng làm trơn Laplace khi ước lượng xác suất và một ví dụ dữ liệu đến tại thời điểm kiểm tra chứa một giá trị chưa từng quan sát trong huấn luyện. Mô hình sẽ xuất ra gì?
1. Bộ phân loại naive Bayes là một ví dụ cụ thể của mạng Bayes, trong đó sự phụ thuộc của các biến ngẫu nhiên được mã hóa bằng một cấu trúc đồ thị. Dù lý thuyết đầy đủ nằm ngoài phạm vi của phần này (xem Koller.Friedman.2009 để biết chi tiết đầy đủ), hãy giải thích vì sao việc cho phép phụ thuộc tường minh giữa hai biến đầu vào trong mô hình XOR cho phép tạo ra một bộ phân loại thành công.


[Thảo luận](https://discuss.d2l.ai/t/1100)
