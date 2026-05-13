# Thị Giác Máy Tính
<a id="chap_cv"></a>

Dù là chẩn đoán y khoa, xe tự lái, giám sát bằng camera hay bộ lọc thông minh, nhiều ứng dụng trong lĩnh vực thị giác máy tính đều liên quan chặt chẽ đến cuộc sống hiện tại và tương lai của chúng ta.
Trong những năm gần đây, deep learning đã là
sức mạnh chuyển đổi thúc đẩy hiệu năng của các hệ thống thị giác máy tính.
Có thể nói rằng các ứng dụng thị giác máy tính tiên tiến nhất gần như không thể tách rời khỏi deep learning.
Vì vậy, chương này sẽ tập trung vào lĩnh vực thị giác máy tính, đồng thời khảo sát các phương pháp và ứng dụng gần đây có ảnh hưởng trong học thuật và công nghiệp.


Trong [chap_cnn](#chap_cnn) và [chap_modern_cnn](#chap_modern_cnn), chúng ta đã nghiên cứu nhiều mạng nơ-ron tích chập
thường được dùng trong thị giác máy tính, và áp dụng chúng
cho các tác vụ phân loại ảnh đơn giản.
Ở đầu chương này, chúng ta sẽ mô tả
hai phương pháp
có thể cải thiện khả năng khái quát hóa của mô hình, cụ thể là *tăng cường ảnh* và *tinh chỉnh*,
rồi áp dụng chúng cho phân loại ảnh.
Vì các mạng nơ-ron sâu có thể biểu diễn ảnh hiệu quả ở nhiều cấp độ,
những biểu diễn theo từng lớp như vậy đã được dùng thành công
trong nhiều tác vụ thị giác máy tính như *phát hiện đối tượng*, *phân đoạn ngữ nghĩa* và *chuyển phong cách*.
Theo ý tưởng chính là tận dụng các biểu diễn theo từng lớp trong thị giác máy tính,
chúng ta sẽ bắt đầu với các thành phần và kỹ thuật chính cho phát hiện đối tượng. Tiếp theo, chúng ta sẽ chỉ ra cách dùng *mạng tích chập đầy đủ* cho phân đoạn ngữ nghĩa ảnh. Sau đó, chúng ta sẽ giải thích cách dùng các kỹ thuật chuyển phong cách để sinh ảnh giống bìa của cuốn sách này.
Cuối cùng, chúng ta kết thúc chương này
bằng cách áp dụng các nội dung của chương này và vài chương trước trên hai tập dữ liệu benchmark thị giác máy tính phổ biến.

```toc
:maxdepth: 2

image-augmentation
fine-tuning
bounding-box
anchor
multiscale-object-detection
object-detection-dataset
ssd
rcnn
semantic-segmentation-and-dataset
transposed-conv
fcn
neural-style
kaggle-cifar10
kaggle-dog
```
