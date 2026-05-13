# Phát Hiện Đối Tượng và Bounding Box
<a id="sec_bbox"></a>


Trong các phần trước (ví dụ [sec_alexnet](#sec_alexnet)--[sec_googlenet](#sec_googlenet)),
chúng ta đã giới thiệu nhiều mô hình cho phân loại ảnh.
Trong các tác vụ phân loại ảnh,
chúng ta giả định rằng chỉ có *một*
đối tượng chính
trong ảnh và chỉ tập trung vào cách
nhận dạng hạng mục của nó.
Tuy nhiên, thường có *nhiều* đối tượng
trong ảnh quan tâm.
Chúng ta không chỉ muốn biết hạng mục của chúng, mà còn cả vị trí cụ thể của chúng trong ảnh.
Trong thị giác máy tính, chúng ta gọi các tác vụ như vậy là *phát hiện đối tượng* (hoặc *nhận dạng đối tượng*).

Phát hiện đối tượng đã được
áp dụng rộng rãi trong nhiều lĩnh vực.
Ví dụ, xe tự lái cần lập kế hoạch
lộ trình di chuyển
bằng cách phát hiện vị trí
của xe cộ, người đi bộ, đường và chướng ngại vật trong các ảnh video thu được.
Bên cạnh đó,
robot có thể dùng kỹ thuật này
để phát hiện và định vị các đối tượng quan tâm
trong suốt quá trình điều hướng trong một môi trường.
Hơn nữa,
các hệ thống an ninh
có thể cần phát hiện các đối tượng bất thường, chẳng hạn kẻ xâm nhập hoặc bom.

Trong vài phần tiếp theo, chúng ta sẽ giới thiệu
một số phương pháp deep learning cho phát hiện đối tượng.
Chúng ta sẽ bắt đầu với phần giới thiệu
về *vị trí* (hoặc *tọa độ*) của đối tượng.

```python
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```python
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```python
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

Chúng ta sẽ tải ảnh mẫu được dùng trong phần này. Có thể thấy có một con chó ở bên trái ảnh và một con mèo ở bên phải.
Chúng là hai đối tượng chính trong ảnh này.

```python
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```python
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## Bounding Box


Trong phát hiện đối tượng,
chúng ta thường dùng một *bounding box* để mô tả vị trí không gian của một đối tượng.
Bounding box là một hình chữ nhật, được xác định bởi tọa độ $x$ và $y$ của góc trên bên trái của hình chữ nhật và các tọa độ tương ứng của góc dưới bên phải.
Một biểu diễn bounding box thường dùng khác là tọa độ theo trục $(x, y)$
của tâm bounding box, cùng chiều rộng và chiều cao của hộp.

[**Ở đây chúng ta định nghĩa các hàm để chuyển đổi giữa**] hai (**biểu diễn
này**):
`box_corner_to_center` chuyển từ biểu diễn hai góc
sang biểu diễn tâm-chiều rộng-chiều cao,
và `box_center_to_corner` làm ngược lại.
Đối số đầu vào `boxes` nên là một tensor hai chiều có
hình dạng ($n$, 4), trong đó $n$ là số lượng bounding box.

```python
#@tab all
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

Chúng ta sẽ [**định nghĩa bounding box của con chó và con mèo trong ảnh**] dựa
trên thông tin tọa độ.
Gốc tọa độ trong ảnh
là góc trên bên trái của ảnh, và sang phải cũng như xuống dưới lần lượt là
các hướng dương của trục $x$ và $y$.

```python
#@tab all
# Here `bbox` is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

Chúng ta có thể xác minh tính đúng đắn của hai
hàm chuyển đổi bounding box bằng cách chuyển đổi hai lần.

```python
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

Hãy [**vẽ các bounding box trong ảnh**] để kiểm tra xem chúng có chính xác không.
Trước khi vẽ, chúng ta sẽ định nghĩa một hàm trợ giúp `bbox_to_rect`. Nó biểu diễn bounding box theo định dạng bounding box của gói `matplotlib`.

```python
#@tab all
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

Sau khi thêm các bounding box lên ảnh,
chúng ta có thể thấy đường viền chính của hai đối tượng về cơ bản nằm trong hai hộp.

```python
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## Tóm Tắt

* Phát hiện đối tượng không chỉ nhận dạng tất cả đối tượng quan tâm trong ảnh, mà còn cả vị trí của chúng. Vị trí thường được biểu diễn bằng một bounding box hình chữ nhật.
* Chúng ta có thể chuyển đổi giữa hai biểu diễn bounding box thường dùng.

## Bài Tập

1. Tìm một ảnh khác và thử gán nhãn một bounding box chứa đối tượng. So sánh việc gán nhãn bounding box và hạng mục: việc nào thường mất lâu hơn?
1. Vì sao chiều trong cùng của đối số đầu vào `boxes` của `box_corner_to_center` và `box_center_to_corner` luôn là 4?


[Discussions](https://discuss.d2l.ai/t/1527)
