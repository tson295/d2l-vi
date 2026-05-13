# Phát Hiện Đối Tượng Đa Tỉ Lệ
<a id="sec_multiscale-object-detection"></a>


Trong [sec_anchor](#sec_anchor),
chúng ta đã sinh nhiều anchor box có tâm tại mỗi pixel của ảnh đầu vào.
Về bản chất, các anchor box này
biểu diễn các mẫu của
những vùng khác nhau trong ảnh.
Tuy nhiên,
chúng ta có thể có quá nhiều anchor box cần tính toán
nếu chúng được sinh cho *mọi* pixel.
Hãy xét một ảnh đầu vào $561 \times 728$.
Nếu năm anchor box
với các hình dạng khác nhau
được sinh cho mỗi pixel làm tâm,
hơn hai triệu anchor box ($561 \times 728 \times 5$) cần được gán nhãn và dự đoán trên ảnh.

## Anchor Box Đa Tỉ Lệ
<a id="subsec_multiscale-anchor-boxes"></a>

Bạn có thể nhận ra rằng
việc giảm số anchor box trên một ảnh không khó.
Chẳng hạn, chúng ta có thể chỉ
lấy mẫu đều một phần nhỏ các pixel
từ ảnh đầu vào
để sinh anchor box có tâm tại chúng.
Ngoài ra,
ở các tỉ lệ khác nhau,
chúng ta có thể sinh các số lượng anchor box khác nhau
với kích thước khác nhau.
Trực giác là
các đối tượng nhỏ có khả năng
xuất hiện trong ảnh cao hơn các đối tượng lớn.
Ví dụ,
các đối tượng $1 \times 1$, $1 \times 2$, và $2 \times 2$
có thể xuất hiện trên một ảnh $2 \times 2$
lần lượt theo 4, 2, và 1 cách khả dĩ.
Vì vậy, khi dùng các anchor box nhỏ hơn để phát hiện các đối tượng nhỏ hơn, chúng ta có thể lấy mẫu nhiều vùng hơn,
trong khi với các đối tượng lớn hơn, ta có thể lấy mẫu ít vùng hơn.

Để minh họa cách sinh anchor box
ở nhiều tỉ lệ, hãy đọc một ảnh.
Chiều cao và chiều rộng của ảnh lần lượt là 561 và 728 pixel.

```python
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```python
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

Nhắc lại rằng trong [sec_conv_layer](#sec_conv_layer),
chúng ta gọi đầu ra dạng mảng hai chiều của
một tầng tích chập là một feature map.
Bằng cách định nghĩa hình dạng feature map,
chúng ta có thể xác định tâm của các anchor box được lấy mẫu đều trên bất kỳ ảnh nào.


Hàm `display_anchors` được định nghĩa bên dưới.
[**Chúng ta sinh anchor box (`anchors`) trên feature map (`fmap`) với mỗi đơn vị (pixel) làm tâm anchor box.**]
Vì các giá trị tọa độ theo trục $(x, y)$
trong các anchor box (`anchors`) đã được chia cho chiều rộng và chiều cao của feature map (`fmap`),
các giá trị này nằm giữa 0 và 1,
biểu thị vị trí tương đối của
anchor box trong feature map.

Vì tâm của các anchor box (`anchors`)
được trải trên toàn bộ các đơn vị của feature map (`fmap`),
các tâm này phải được phân bố *đều*
trên mọi ảnh đầu vào
xét theo vị trí không gian tương đối.
Cụ thể hơn,
cho chiều rộng và chiều cao của feature map lần lượt là `fmap_w` và `fmap_h`,
hàm sau sẽ lấy mẫu *đều*
các pixel trong `fmap_h` hàng và `fmap_w` cột
trên bất kỳ ảnh đầu vào nào.
Với tâm tại các pixel được lấy mẫu đều này,
các anchor box có tỉ lệ `s` (giả sử độ dài của danh sách `s` là 1) và các tỉ số cạnh khác nhau (`ratios`)
sẽ được sinh ra.

```python
#@tab mxnet
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```python
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

Trước tiên, hãy [**xét
việc phát hiện các đối tượng nhỏ**].
Để dễ phân biệt khi hiển thị, các anchor box với tâm khác nhau ở đây không chồng lấp:
tỉ lệ anchor box được đặt là 0.15
và chiều cao cũng như chiều rộng của feature map được đặt là 4. Ta có thể thấy
rằng tâm của các anchor box trong 4 hàng và 4 cột trên ảnh được phân bố đều.

```python
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

Tiếp theo, chúng ta [**giảm một nửa chiều cao và chiều rộng của feature map và dùng các anchor box lớn hơn để phát hiện các đối tượng lớn hơn**]. Khi tỉ lệ được đặt là 0.4,
một số anchor box sẽ chồng lấp với nhau.

```python
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

Cuối cùng, chúng ta [**tiếp tục giảm một nửa chiều cao và chiều rộng của feature map và tăng tỉ lệ anchor box lên 0.8**]. Lúc này tâm của anchor box là tâm của ảnh.

```python
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## Phát Hiện Đa Tỉ Lệ


Vì chúng ta đã sinh các anchor box đa tỉ lệ,
ta sẽ dùng chúng để phát hiện các đối tượng có kích thước khác nhau
ở các tỉ lệ khác nhau.
Sau đây
chúng ta giới thiệu một phương pháp phát hiện đối tượng đa tỉ lệ dựa trên CNN
mà chúng ta sẽ hiện thực
trong [sec_ssd](#sec_ssd).

Ở một tỉ lệ nào đó,
giả sử chúng ta có $c$ feature map với hình dạng $h \times w$.
Dùng phương pháp trong [subsec_multiscale-anchor-boxes](#subsec_multiscale-anchor-boxes),
chúng ta sinh $hw$ tập anchor box,
trong đó mỗi tập có $a$ anchor box cùng tâm.
Ví dụ,
ở tỉ lệ đầu tiên trong các thí nghiệm ở [subsec_multiscale-anchor-boxes](#subsec_multiscale-anchor-boxes),
với mười feature map (số kênh) kích thước $4 \times 4$,
chúng ta đã sinh 16 tập anchor box,
trong đó mỗi tập chứa 3 anchor box cùng tâm.
Tiếp theo, mỗi anchor box được gán nhãn
lớp và offset dựa trên các ground-truth bounding box. Ở tỉ lệ hiện tại, mô hình phát hiện đối tượng cần dự đoán lớp và offset của $hw$ tập anchor box trên ảnh đầu vào, trong đó các tập khác nhau có tâm khác nhau.


Giả sử $c$ feature map ở đây
là các đầu ra trung gian thu được
bởi lan truyền xuôi của CNN dựa trên ảnh đầu vào. Vì có $hw$ vị trí không gian khác nhau trên mỗi feature map,
cùng một vị trí không gian có thể được
xem là có $c$ đơn vị.
Theo
định nghĩa trường tiếp nhận trong [sec_conv_layer](#sec_conv_layer),
$c$ đơn vị này tại cùng một vị trí không gian
của các feature map
có cùng trường tiếp nhận trên ảnh đầu vào:
chúng biểu diễn thông tin ảnh đầu vào
trong cùng một trường tiếp nhận.
Do đó, chúng ta có thể biến đổi $c$ đơn vị
của các feature map tại cùng một vị trí không gian
thành
lớp và offset của $a$ anchor box
được sinh bằng vị trí không gian này.
Về bản chất,
chúng ta dùng thông tin của ảnh đầu vào trong một trường tiếp nhận nhất định
để dự đoán lớp và offset của các anchor box
nằm gần trường tiếp nhận đó
trên ảnh đầu vào.


Khi các feature map ở các tầng khác nhau
có trường tiếp nhận với kích thước khác nhau trên ảnh đầu vào, chúng có thể được dùng để phát hiện các đối tượng có kích thước khác nhau.
Ví dụ, chúng ta có thể thiết kế một mạng nơ-ron trong đó
các đơn vị của feature map gần tầng đầu ra hơn
có trường tiếp nhận rộng hơn,
nên chúng có thể phát hiện các đối tượng lớn hơn từ ảnh đầu vào.

Tóm lại, chúng ta có thể tận dụng
các biểu diễn theo tầng của ảnh ở nhiều mức
bằng mạng nơ-ron sâu
cho phát hiện đối tượng đa tỉ lệ.
Chúng ta sẽ minh họa cách hoạt động này bằng một ví dụ cụ thể
trong [sec_ssd](#sec_ssd).


## Tóm Tắt

* Ở nhiều tỉ lệ, chúng ta có thể sinh anchor box với kích thước khác nhau để phát hiện các đối tượng có kích thước khác nhau.
* Bằng cách định nghĩa hình dạng của feature map, chúng ta có thể xác định tâm của các anchor box được lấy mẫu đều trên bất kỳ ảnh nào.
* Chúng ta dùng thông tin của ảnh đầu vào trong một trường tiếp nhận nhất định để dự đoán lớp và offset của các anchor box nằm gần trường tiếp nhận đó trên ảnh đầu vào.
* Thông qua deep learning, chúng ta có thể tận dụng các biểu diễn theo tầng của ảnh ở nhiều mức cho phát hiện đối tượng đa tỉ lệ.


## Bài Tập

1. Theo các thảo luận của chúng ta trong [sec_alexnet](#sec_alexnet), mạng nơ-ron sâu học các đặc trưng phân cấp với mức độ trừu tượng tăng dần cho ảnh. Trong phát hiện đối tượng đa tỉ lệ, các feature map ở các tỉ lệ khác nhau có tương ứng với các mức trừu tượng khác nhau không? Vì sao hoặc vì sao không?
1. Ở tỉ lệ đầu tiên (`fmap_w=4, fmap_h=4`) trong các thí nghiệm ở [subsec_multiscale-anchor-boxes](#subsec_multiscale-anchor-boxes), hãy sinh các anchor box phân bố đều và có thể chồng lấp.
1. Cho một biến feature map có hình dạng $1 \times c \times h \times w$, trong đó $c$, $h$, và $w$ lần lượt là số kênh, chiều cao, và chiều rộng của các feature map. Làm thế nào để biến đổi biến này thành lớp và offset của anchor box? Hình dạng của đầu ra là gì?


[Thảo luận](https://discuss.d2l.ai/t/1607)
