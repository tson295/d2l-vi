# Phân Đoạn Ngữ Nghĩa và Tập Dữ Liệu
<a id="sec_semantic_segmentation"></a>

Khi thảo luận các tác vụ phát hiện đối tượng
trong [sec_bbox](#sec_bbox)--[sec_rcnn](#sec_rcnn),
các bounding box hình chữ nhật
được dùng để gán nhãn và dự đoán đối tượng trong ảnh.
Mục này sẽ thảo luận bài toán *phân đoạn ngữ nghĩa*,
tập trung vào cách chia một ảnh thành các vùng thuộc các lớp ngữ nghĩa khác nhau.
Khác với phát hiện đối tượng,
phân đoạn ngữ nghĩa
nhận dạng và hiểu
những gì có trong ảnh ở mức pixel:
việc gán nhãn và dự đoán các vùng ngữ nghĩa của nó đều
ở mức pixel.
[fig_segmentation](#fig_segmentation) minh họa các nhãn
của chó, mèo, và nền trong ảnh ở bài toán phân đoạn ngữ nghĩa.
So với phát hiện đối tượng,
biên ở mức pixel được gán nhãn
trong phân đoạn ngữ nghĩa rõ ràng chi tiết hơn nhiều.


![Nhãn của chó, mèo, và nền trong ảnh ở bài toán phân đoạn ngữ nghĩa.](../img/segmentation.svg)
<a id="fig_segmentation"></a>


## Phân Đoạn Ảnh và Phân Đoạn Thực Thể

Cũng có hai tác vụ quan trọng
trong lĩnh vực thị giác máy tính tương tự với phân đoạn ngữ nghĩa,
đó là phân đoạn ảnh và phân đoạn thực thể.
Chúng ta sẽ phân biệt ngắn gọn
chúng với phân đoạn ngữ nghĩa như sau.

* *Phân đoạn ảnh* chia một ảnh thành nhiều vùng cấu thành. Các phương pháp cho loại bài toán này thường khai thác sự tương quan giữa các pixel trong ảnh. Nó không cần thông tin nhãn về các pixel ảnh trong quá trình huấn luyện, và không thể đảm bảo rằng các vùng được phân đoạn sẽ có ngữ nghĩa mà chúng ta muốn thu được trong khi dự đoán. Lấy ảnh trong [fig_segmentation](#fig_segmentation) làm đầu vào, phân đoạn ảnh có thể chia con chó thành hai vùng: một vùng bao phủ miệng và mắt, chủ yếu màu đen, và vùng còn lại bao phủ phần thân còn lại, chủ yếu màu vàng.
* *Phân đoạn thực thể* cũng được gọi là *phát hiện và phân đoạn đồng thời*. Nó nghiên cứu cách nhận dạng các vùng ở mức pixel của từng thực thể đối tượng trong một ảnh. Khác với phân đoạn ngữ nghĩa, phân đoạn thực thể cần phân biệt không chỉ ngữ nghĩa mà còn cả các thực thể đối tượng khác nhau. Ví dụ, nếu có hai con chó trong ảnh, phân đoạn thực thể cần phân biệt một pixel thuộc về con chó nào trong hai con chó đó.


## Tập Dữ Liệu Phân Đoạn Ngữ Nghĩa Pascal VOC2012

[**Một trong những tập dữ liệu phân đoạn ngữ nghĩa quan trọng nhất
là [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).**]
Sau đây,
chúng ta sẽ xem xét tập dữ liệu này.

```python
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

```python
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
```

Tệp tar của tập dữ liệu có dung lượng khoảng 2 GB,
vì vậy việc tải tệp có thể mất một lúc.
Tập dữ liệu sau khi giải nén nằm tại `../data/VOCdevkit/VOC2012`.

```python
#@tab all
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

Sau khi đi vào đường dẫn `../data/VOCdevkit/VOC2012`,
chúng ta có thể thấy các thành phần khác nhau của tập dữ liệu.
Đường dẫn `ImageSets/Segmentation` chứa các tệp văn bản
chỉ định các mẫu huấn luyện và kiểm tra,
trong khi các đường dẫn `JPEGImages` và `SegmentationClass`
lần lượt lưu ảnh đầu vào và nhãn cho từng ví dụ.
Nhãn ở đây cũng ở định dạng ảnh,
có cùng kích thước
với ảnh đầu vào được gán nhãn của nó.
Bên cạnh đó,
các pixel có cùng màu trong bất kỳ ảnh nhãn nào đều thuộc cùng một lớp ngữ nghĩa.
Sau đây định nghĩa hàm `read_voc_images` để [**đọc tất cả ảnh đầu vào và nhãn vào bộ nhớ**].

```python
#@tab mxnet
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(image.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(image.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

```python
#@tab pytorch
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

Chúng ta [**vẽ năm ảnh đầu vào đầu tiên và nhãn của chúng**].
Trong các ảnh nhãn, màu trắng và đen lần lượt biểu diễn biên và nền, còn các màu khác tương ứng với các lớp khác nhau.

```python
#@tab mxnet
n = 5
imgs = train_features[:n] + train_labels[:n]
d2l.show_images(imgs, 2, n);
```

```python
#@tab pytorch
n = 5
imgs = train_features[:n] + train_labels[:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n);
```

Tiếp theo, chúng ta [**liệt kê
các giá trị màu RGB và tên lớp**]
cho tất cả nhãn trong tập dữ liệu này.

```python
#@tab all
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

Với hai hằng số đã định nghĩa ở trên,
chúng ta có thể thuận tiện
[**tìm chỉ số lớp cho từng pixel trong một nhãn**].
Chúng ta định nghĩa hàm `voc_colormap2label`
để xây dựng ánh xạ từ các giá trị màu RGB ở trên
sang chỉ số lớp,
và hàm `voc_label_indices`
để ánh xạ bất kỳ giá trị RGB nào sang chỉ số lớp của chúng trong tập dữ liệu Pascal VOC2012 này.

```python
#@tab mxnet
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

```python
#@tab pytorch
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

[**Ví dụ**], trong ảnh ví dụ đầu tiên,
chỉ số lớp cho phần trước của máy bay là 1,
trong khi chỉ số nền là 0.

```python
#@tab all
y = voc_label_indices(train_labels[0], voc_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]
```

### Tiền Xử Lý Dữ Liệu

Trong các thí nghiệm trước
như [sec_alexnet](#sec_alexnet)--[sec_googlenet](#sec_googlenet),
ảnh được đổi tỉ lệ
để khớp với hình dạng đầu vào mà mô hình yêu cầu.
Tuy nhiên, trong phân đoạn ngữ nghĩa,
làm như vậy
đòi hỏi phải đổi tỉ lệ các lớp pixel dự đoán
trở lại hình dạng ban đầu của ảnh đầu vào.
Việc đổi tỉ lệ như vậy có thể không chính xác,
đặc biệt với các vùng được phân đoạn thuộc các lớp khác nhau. Để tránh vấn đề này,
chúng ta cắt ảnh về một hình dạng *cố định* thay vì đổi tỉ lệ. Cụ thể, [**dùng cắt ngẫu nhiên từ tăng cường ảnh, chúng ta cắt cùng một vùng của
ảnh đầu vào và nhãn**].

```python
#@tab mxnet
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label
```

```python
#@tab pytorch
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
```

```python
#@tab mxnet
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

```python
#@tab pytorch
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

### [**Lớp Tập Dữ Liệu Phân Đoạn Ngữ Nghĩa Tùy Chỉnh**]

Chúng ta định nghĩa một lớp tập dữ liệu phân đoạn ngữ nghĩa tùy chỉnh `VOCSegDataset` bằng cách kế thừa lớp `Dataset` do API cấp cao cung cấp.
Bằng cách hiện thực hàm `__getitem__`,
chúng ta có thể truy cập tùy ý ảnh đầu vào có chỉ số `idx` trong tập dữ liệu và chỉ số lớp của từng pixel trong ảnh này.
Vì một số ảnh trong tập dữ liệu
có kích thước nhỏ hơn
kích thước đầu ra của phép cắt ngẫu nhiên,
các ví dụ này được lọc bỏ
bằng một hàm `filter` tùy chỉnh.
Ngoài ra, chúng ta cũng
định nghĩa hàm `normalize_image` để
chuẩn hóa các giá trị của ba kênh RGB của ảnh đầu vào.

```python
#@tab mxnet
class VOCSegDataset(gluon.data.Dataset):
    """A customized dataset to load the VOC dataset."""
    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

```python
#@tab pytorch
class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

### [**Đọc Tập Dữ Liệu**]

Chúng ta dùng lớp `VOCSegDataset` tùy chỉnh để
tạo các thực thể của tập huấn luyện và tập kiểm tra, tương ứng.
Giả sử rằng
chúng ta chỉ định hình dạng đầu ra của các ảnh được cắt ngẫu nhiên là $320\times 480$.
Bên dưới, ta có thể xem số lượng ví dụ
được giữ lại trong tập huấn luyện và tập kiểm tra.

```python
#@tab all
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
```

Đặt kích thước batch là 64,
chúng ta định nghĩa iterator dữ liệu cho tập huấn luyện.
Hãy in hình dạng của minibatch đầu tiên.
Khác với phân loại ảnh hoặc phát hiện đối tượng, nhãn ở đây là các tensor ba chiều.

```python
#@tab mxnet
batch_size = 64
train_iter = gluon.data.DataLoader(voc_train, batch_size, shuffle=True,
                                   last_batch='discard',
                                   num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

```python
#@tab pytorch
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

### [**Gộp Tất Cả Lại**]

Cuối cùng, chúng ta định nghĩa hàm `load_data_voc` sau
để tải xuống và đọc tập dữ liệu phân đoạn ngữ nghĩa Pascal VOC2012.
Nó trả về các iterator dữ liệu cho cả tập huấn luyện và tập kiểm tra.

```python
#@tab mxnet
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter
```

```python
#@tab pytorch
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
```

## Tóm Tắt

* Phân đoạn ngữ nghĩa nhận dạng và hiểu những gì có trong ảnh ở mức pixel bằng cách chia ảnh thành các vùng thuộc các lớp ngữ nghĩa khác nhau.
* Một trong những tập dữ liệu phân đoạn ngữ nghĩa quan trọng nhất là Pascal VOC2012.
* Trong phân đoạn ngữ nghĩa, vì ảnh đầu vào và nhãn tương ứng một-một ở mức pixel, ảnh đầu vào được cắt ngẫu nhiên về một hình dạng cố định thay vì đổi tỉ lệ.


## Bài Tập

1. Phân đoạn ngữ nghĩa có thể được áp dụng như thế nào trong xe tự lái và chẩn đoán ảnh y tế? Bạn có thể nghĩ ra các ứng dụng khác không?
1. Nhắc lại các mô tả về tăng cường dữ liệu trong [sec_image_augmentation](#sec_image_augmentation). Những phương pháp tăng cường ảnh nào được dùng trong phân loại ảnh sẽ không khả thi khi áp dụng trong phân đoạn ngữ nghĩa?


[Thảo luận](https://discuss.d2l.ai/t/1480)
