# Tập Dữ Liệu Phát Hiện Đối Tượng
<a id="sec_object-detection-dataset"></a>

Trong lĩnh vực phát hiện đối tượng không có tập dữ liệu nhỏ như MNIST và Fashion-MNIST.
Để nhanh chóng minh họa các mô hình phát hiện đối tượng,
[**chúng tôi đã thu thập và gán nhãn một tập dữ liệu nhỏ**].
Trước tiên, chúng tôi chụp ảnh các quả chuối miễn phí trong văn phòng
và sinh ra
1000 ảnh chuối với các góc xoay và kích thước khác nhau.
Sau đó, chúng tôi đặt mỗi ảnh chuối
ở một vị trí ngẫu nhiên trên một ảnh nền nào đó.
Cuối cùng, chúng tôi gán nhãn bounding box cho các quả chuối trong các ảnh đó.


## [**Tải Xuống Tập Dữ Liệu**]

Tập dữ liệu phát hiện chuối cùng với toàn bộ ảnh và
các tệp nhãn csv có thể được tải trực tiếp từ Internet.

```python
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```python
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```python
#@tab all
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## Đọc Tập Dữ Liệu

Chúng ta sẽ [**đọc tập dữ liệu phát hiện chuối**] trong hàm `read_data_bananas`
bên dưới.
Tập dữ liệu bao gồm một tệp csv cho
nhãn lớp đối tượng và
tọa độ ground-truth bounding box
ở góc trên bên trái và góc dưới bên phải.

```python
#@tab mxnet
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```python
#@tab pytorch
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

Bằng cách dùng hàm `read_data_bananas` để đọc ảnh và nhãn,
lớp `BananasDataset` sau
sẽ cho phép chúng ta [**tạo một thực thể `Dataset` tùy chỉnh**]
để tải tập dữ liệu phát hiện chuối.

```python
#@tab mxnet
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```python
#@tab pytorch
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

Cuối cùng, chúng ta định nghĩa
hàm `load_data_bananas` để [**trả về hai
iterator dữ liệu cho cả tập huấn luyện và tập kiểm tra.**]
Với tập dữ liệu kiểm tra,
không cần đọc theo thứ tự ngẫu nhiên.

```python
#@tab mxnet
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```python
#@tab pytorch
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

Hãy [**đọc một minibatch và in hình dạng của
cả ảnh lẫn nhãn**] trong minibatch này.
Hình dạng của minibatch ảnh,
(kích thước batch, số kênh, chiều cao, chiều rộng),
trông quen thuộc:
nó giống với các tác vụ phân loại ảnh trước đây của chúng ta.
Hình dạng của minibatch nhãn là
(kích thước batch, $m$, 5),
trong đó $m$ là số bounding box lớn nhất có thể
mà bất kỳ ảnh nào trong tập dữ liệu có.

Mặc dù tính toán theo minibatch hiệu quả hơn,
nó yêu cầu tất cả các ví dụ ảnh
có cùng số lượng bounding box để tạo thành một minibatch bằng phép nối.
Nhìn chung,
các ảnh có thể có số lượng bounding box khác nhau;
do đó,
các ảnh có ít hơn $m$ bounding box
sẽ được đệm bằng các bounding box không hợp lệ
cho đến khi đạt $m$.
Khi đó
nhãn của mỗi bounding box được biểu diễn bằng một mảng độ dài 5.
Phần tử đầu tiên trong mảng là lớp của đối tượng trong bounding box,
trong đó -1 biểu thị một bounding box không hợp lệ dùng cho đệm.
Bốn phần tử còn lại của mảng là
các giá trị tọa độ ($x$, $y$)
của góc trên bên trái và góc dưới bên phải
của bounding box (phạm vi nằm giữa 0 và 1).
Với tập dữ liệu chuối,
vì chỉ có một bounding box trên mỗi ảnh,
chúng ta có $m=1$.

```python
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**Minh Họa**]

Hãy minh họa mười ảnh cùng với các ground-truth bounding box đã được gán nhãn.
Ta có thể thấy rằng góc xoay, kích thước, và vị trí của chuối thay đổi trên các ảnh này.
Dĩ nhiên, đây chỉ là một tập dữ liệu nhân tạo đơn giản.
Trong thực tế, các tập dữ liệu đời thực thường phức tạp hơn nhiều.

```python
#@tab mxnet
imgs = (batch[0][:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```python
#@tab pytorch
imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## Tóm Tắt

* Tập dữ liệu phát hiện chuối mà chúng ta thu thập có thể được dùng để minh họa các mô hình phát hiện đối tượng.
* Việc tải dữ liệu cho phát hiện đối tượng tương tự như cho phân loại ảnh. Tuy nhiên, trong phát hiện đối tượng, nhãn cũng chứa thông tin về ground-truth bounding box, điều không có trong phân loại ảnh.


## Bài Tập

1. Minh họa các ảnh khác cùng với ground-truth bounding box trong tập dữ liệu phát hiện chuối. Chúng khác nhau như thế nào về bounding box và đối tượng?
1. Giả sử chúng ta muốn áp dụng tăng cường dữ liệu, chẳng hạn cắt ngẫu nhiên, cho phát hiện đối tượng. Điều này có thể khác với phân loại ảnh như thế nào? Gợi ý: điều gì xảy ra nếu ảnh bị cắt chỉ chứa một phần nhỏ của đối tượng?


[Thảo luận](https://discuss.d2l.ai/t/1608)
