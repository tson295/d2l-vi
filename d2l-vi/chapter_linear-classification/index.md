# Mạng Nơ-ron Tuyến tính cho Phân loại
<a id="chap_classification"></a>

Bây giờ bạn đã nắm vững tất cả các cơ chế,
bạn đã sẵn sàng áp dụng các kỹ năng đã học cho các loại tác vụ rộng hơn.
Ngay cả khi ta chuyển hướng sang phân loại,
hầu hết các cơ sở hạ tầng vẫn giống nhau:
tải dữ liệu, truyền qua mô hình,
tạo đầu ra, tính hàm mất mát,
lấy gradient theo trọng số,
và cập nhật mô hình.
Tuy nhiên, dạng chính xác của mục tiêu,
tham số hóa lớp đầu ra,
và lựa chọn hàm mất mát sẽ thích nghi
để phù hợp với cài đặt *phân loại*.

```toc
:maxdepth: 2

softmax-regression
image-classification-dataset
classification
softmax-regression-scratch
softmax-regression-concise
generalization-classification
environment-and-distribution-shift
```
