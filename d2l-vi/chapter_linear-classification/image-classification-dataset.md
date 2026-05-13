# Bộ Dữ liệu Phân loại Ảnh
<a id="sec_fashion_mnist"></a>

(~~Bộ dữ liệu MNIST là một trong những bộ dữ liệu được sử dụng rộng rãi cho phân loại ảnh, nhưng nó quá đơn giản để làm benchmark. Chúng ta sẽ sử dụng bộ dữ liệu Fashion-MNIST tương tự nhưng phức tạp hơn ~~)

Một bộ dữ liệu được sử dụng rộng rãi cho phân loại ảnh là [bộ dữ liệu MNIST](https://en.wikipedia.org/wiki/MNIST_database) [LeCun.Bottou.Bengio.ea.1998] gồm các chữ số viết tay. Tại thời điểm phát hành vào những năm 1990, nó đặt ra một thách thức đáng kể cho hầu hết các thuật toán machine learning, bao gồm 60,000 ảnh có độ phân giải $28 \times 28$ pixel (cùng với một tập dữ liệu kiểm tra gồm 10,000 ảnh). Để có cái nhìn tổng thể, vào năm 1995, một máy Sun SPARCStation 5 với 64MB RAM và tốc độ 5 MFLOPs được coi là thiết bị tiên tiến nhất cho machine learning tại Phòng thí nghiệm Bell của AT&T. Đạt độ chính xác cao trong nhận dạng chữ số là thành phần then chốt trong việc tự động hóa phân loại thư cho USPS vào những năm 1990. Các mạng sâu như LeNet-5 [LeCun.Jackel.Bottou.ea.1995], máy vector hỗ trợ với bất biến [Scholkopf.Burges.Vapnik.1996], và các bộ phân loại khoảng cách tiếp tuyến [Simard.LeCun.Denker.ea.1998] đều có thể đạt tỷ lệ lỗi dưới 1%.

Trong hơn một thập kỷ, MNIST đóng vai trò là điểm tham chiếu để so sánh các thuật toán machine learning.
Mặc dù nó đã có một thời gian dài là benchmark,
ngay cả các mô hình đơn giản theo tiêu chuẩn ngày nay cũng đạt độ chính xác phân loại trên 95%,
khiến nó không còn phù hợp để phân biệt giữa các mô hình mạnh và yếu. Hơn nữa, bộ dữ liệu cho phép đạt *rất* cao về độ chính xác, điều thường không thấy trong nhiều bài toán phân loại. Điều này làm lệch hướng phát triển thuật toán về phía các họ thuật toán cụ thể có thể tận dụng dữ liệu sạch, chẳng hạn như các phương pháp active set và thuật toán active set tìm biên.
Ngày nay, MNIST đóng vai trò như một bài kiểm tra tỉnh hơn là một benchmark. ImageNet [Deng.Dong.Socher.ea.2009] đặt ra thách thức phù hợp hơn nhiều. Thật không may, ImageNet quá lớn cho nhiều ví dụ và minh họa trong cuốn sách này, vì sẽ mất quá nhiều thời gian để huấn luyện nhằm làm cho các ví dụ có thể tương tác được. Thay thế cho nó, chúng ta sẽ tập trung thảo luận trong các phần sắp tới vào bộ dữ liệu Fashion-MNIST [Xiao.Rasul.Vollgraf.2017] có chất lượng tương tự nhưng nhỏ hơn nhiều, được phát hành vào năm 2017. Nó chứa ảnh của 10 danh mục quần áo với độ phân giải $28 \times 28$ pixel.


```python
%matplotlib inline
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms

d2l.use_svg_display()
```


## Tải Bộ Dữ liệu

Vì bộ dữ liệu Fashion-MNIST rất hữu ích, tất cả các framework lớn đều cung cấp phiên bản đã được tiền xử lý của nó. Chúng ta có thể [**tải xuống và đọc nó vào bộ nhớ bằng các tiện ích tích hợp của framework.**]


```python
class FashionMNIST(d2l.DataModule):  
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```


Fashion-MNIST gồm ảnh từ 10 danh mục, mỗi danh mục được đại diện bởi 6000 ảnh trong tập huấn luyện và 1000 ảnh trong tập kiểm tra.
*Tập dữ liệu kiểm tra* được dùng để đánh giá hiệu suất mô hình (không được dùng để huấn luyện).
Do đó, tập huấn luyện và tập kiểm tra
lần lượt chứa 60,000 và 10,000 ảnh.


Các ảnh là ảnh xám và được phóng to lên độ phân giải $32 \times 32$ pixel ở trên. Điều này tương tự với bộ dữ liệu MNIST gốc vốn gồm các ảnh đen trắng (nhị phân). Tuy nhiên, hãy lưu ý rằng hầu hết dữ liệu ảnh hiện đại có ba kênh (đỏ, xanh lá, xanh dương) và ảnh siêu phổ có thể có hơn 100 kênh (cảm biến HyMap có 126 kênh).
Theo quy ước, chúng ta lưu trữ ảnh dưới dạng tensor $c \times h \times w$, trong đó $c$ là số kênh màu, $h$ là chiều cao và $w$ là chiều rộng.

```python
data.train[0][0].shape
```

[~~Hai hàm tiện ích để trực quan hóa bộ dữ liệu~~]

Các danh mục của Fashion-MNIST có tên dễ hiểu đối với con người.
Phương thức tiện ích sau chuyển đổi giữa nhãn số và tên của chúng.

```python
@d2l.add_to_class(FashionMNIST)  
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## Đọc một Minibatch

Để thuận tiện khi đọc từ tập huấn luyện và tập kiểm tra,
chúng ta sử dụng data iterator tích hợp thay vì tự tạo từ đầu.
Nhớ lại rằng ở mỗi vòng lặp, một data iterator
[**đọc một minibatch dữ liệu với kích thước `batch_size`.**]
Chúng ta cũng xáo trộn ngẫu nhiên các mẫu cho data iterator của tập huấn luyện.


```python
@d2l.add_to_class(FashionMNIST)  
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```


Để xem cách hoạt động, hãy tải một minibatch ảnh bằng cách gọi phương thức `train_dataloader`. Nó chứa 64 ảnh.

```python
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

Hãy xem thời gian cần để đọc các ảnh. Mặc dù đây là bộ tải tích hợp, nhưng nó không nhanh như chớp. Tuy vậy, điều này là đủ vì việc xử lý ảnh với mạng sâu mất nhiều thời gian hơn đáng kể. Do đó, huấn luyện mạng sẽ không bị giới hạn bởi I/O là đủ tốt.

```python
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

## Trực quan hóa

Chúng ta thường xuyên sử dụng bộ dữ liệu Fashion-MNIST. Hàm tiện ích `show_images` có thể được dùng để trực quan hóa ảnh cùng các nhãn liên quan.
Bỏ qua chi tiết cài đặt, chúng ta chỉ cần biết cách gọi `d2l.show_images` thay vì cách nó hoạt động đối với các hàm tiện ích như vậy.

```python
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  
    """Plot a list of images."""
    raise NotImplementedError
```

Hãy sử dụng nó đúng cách. Nhìn chung, trực quan hóa và kiểm tra dữ liệu mà bạn đang huấn luyện là một ý tưởng tốt.
Con người rất giỏi phát hiện những điều kỳ lạ và vì vậy, trực quan hóa đóng vai trò như một biện pháp bảo vệ bổ sung chống lại các sai lầm và lỗi trong thiết kế thực nghiệm. Dưới đây là [**các ảnh và nhãn tương ứng**] (dạng văn bản)
cho một vài mẫu đầu tiên trong tập huấn luyện.

```python
@d2l.add_to_class(FashionMNIST)  
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    if tab.selected('mxnet', 'pytorch'):
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
    if tab.selected('tensorflow'):
        d2l.show_images(tf.squeeze(X), nrows, ncols, titles=labels)
    if tab.selected('jax'):
        d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

Bây giờ chúng ta đã sẵn sàng làm việc với bộ dữ liệu Fashion-MNIST trong các phần tiếp theo.

## Tóm tắt

Bây giờ chúng ta có một bộ dữ liệu thực tế hơn để sử dụng cho phân loại. Fashion-MNIST là bộ dữ liệu phân loại trang phục gồm các ảnh đại diện cho 10 danh mục. Chúng ta sẽ sử dụng bộ dữ liệu này trong các phần và chương tiếp theo để đánh giá các thiết kế mạng khác nhau, từ mô hình tuyến tính đơn giản đến mạng phần dư tiên tiến. Như thông thường với ảnh, chúng ta đọc chúng dưới dạng tensor có hình dạng (batch size, số kênh, chiều cao, chiều rộng). Hiện tại, chúng ta chỉ có một kênh vì các ảnh là ảnh xám (hình ảnh trực quan hóa ở trên sử dụng bảng màu giả để hiển thị tốt hơn).

Cuối cùng, data iterator là thành phần quan trọng để có hiệu suất hiệu quả. Ví dụ, chúng ta có thể dùng GPU để giải nén ảnh hiệu quả, chuyển mã video, hoặc các bước tiền xử lý khác. Bất cứ khi nào có thể, bạn nên dựa vào các data iterator được cài đặt tốt tận dụng tính toán hiệu suất cao để tránh làm chậm vòng lặp huấn luyện.


## Bài tập

1. Việc giảm `batch_size` (chẳng hạn xuống 1) có ảnh hưởng đến hiệu suất đọc không?
1. Hiệu suất của data iterator rất quan trọng. Bạn có nghĩ cài đặt hiện tại đủ nhanh không? Khám phá các tùy chọn khác nhau để cải thiện nó. Dùng bộ phân tích hệ thống để tìm ra các điểm nghẽn cổ chai.
1. Kiểm tra tài liệu API trực tuyến của framework. Có những bộ dữ liệu nào khác?


[Discussions](https://discuss.d2l.ai/t/49)
