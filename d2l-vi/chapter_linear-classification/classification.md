# Mô hình Phân loại Cơ sở
<a id="sec_classification"></a>

Bạn có thể nhận thấy rằng các cài đặt từ đầu và cài đặt gọn bằng chức năng của framework khá giống nhau trong trường hợp hồi quy. Điều tương tự cũng đúng với phân loại. Vì nhiều mô hình trong cuốn sách này liên quan đến phân loại, nên việc thêm các chức năng để hỗ trợ thiết lập này một cách cụ thể là xứng đáng. Phần này cung cấp một lớp cơ sở cho các mô hình phân loại để đơn giản hóa code trong tương lai.


```python
from d2l import torch as d2l
import torch
```


## Lớp `Classifier`


Theo mặc định, chúng ta sử dụng bộ tối ưu hóa stochastic gradient descent, hoạt động trên minibatch, giống như cách chúng ta đã làm trong ngữ cảnh hồi quy tuyến tính.


```python
@d2l.add_to_class(d2l.Module)  
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```


## Độ chính xác

Cho trước phân phối xác suất dự đoán `y_hat`,
chúng ta thường chọn lớp có xác suất dự đoán cao nhất
mỗi khi phải đưa ra dự đoán cứng.
Thực vậy, nhiều ứng dụng yêu cầu chúng ta phải đưa ra lựa chọn.
Ví dụ, Gmail phải phân loại email vào "Chính", "Mạng xã hội", "Cập nhật", "Diễn đàn", hoặc "Spam".
Nó có thể ước tính xác suất nội bộ,
nhưng cuối cùng nó phải chọn một trong các lớp.

Khi dự đoán nhất quán với lớp nhãn `y`, chúng là đúng.
Độ chính xác phân loại là tỷ lệ tất cả các dự đoán đúng.
Mặc dù có thể khó tối ưu hóa độ chính xác trực tiếp (nó không khả vi),
nhưng đây thường là thước đo hiệu suất mà chúng ta quan tâm nhất. Đây thường là
đại lượng liên quan trong các benchmark. Vì vậy, chúng ta hầu như luôn báo cáo nó khi huấn luyện bộ phân loại.

Độ chính xác được tính như sau.
Đầu tiên, nếu `y_hat` là ma trận,
chúng ta giả định rằng chiều thứ hai lưu trữ điểm dự đoán cho mỗi lớp.
Chúng ta dùng `argmax` để thu được lớp dự đoán bằng chỉ số có giá trị lớn nhất trong mỗi hàng.
Sau đó chúng ta [**so sánh lớp dự đoán với nhãn thực `y` theo từng phần tử.**]
Vì toán tử bằng `==` nhạy cảm với kiểu dữ liệu,
chúng ta chuyển đổi kiểu dữ liệu của `y_hat` để khớp với `y`.
Kết quả là tensor chứa các phần tử 0 (sai) và 1 (đúng).
Lấy tổng cho ta số dự đoán đúng.


## Tóm tắt

Phân loại là một bài toán phổ biến đủ để xứng đáng có các hàm tiện ích riêng. Điều quan trọng trung tâm trong phân loại là *độ chính xác* của bộ phân loại. Lưu ý rằng mặc dù chúng ta thường quan tâm chủ yếu đến độ chính xác, chúng ta huấn luyện các bộ phân loại để tối ưu hóa nhiều mục tiêu khác vì lý do thống kê và tính toán. Tuy nhiên, bất kể hàm mất mát nào được tối thiểu hóa trong quá trình huấn luyện, việc có một phương thức tiện ích để đánh giá thực nghiệm độ chính xác của bộ phân loại là hữu ích.


## Bài tập

1. Ký hiệu $L_\textrm{v}$ là mất mát kiểm định, và đặt $L_\textrm{v}^\textrm{q}$ là ước tính nhanh và đơn giản của nó được tính bằng cách lấy trung bình hàm mất mát trong phần này. Cuối cùng, ký hiệu $l_\textrm{v}^\textrm{b}$ là mất mát trên minibatch cuối cùng. Biểu diễn $L_\textrm{v}$ theo $L_\textrm{v}^\textrm{q}$, $l_\textrm{v}^\textrm{b}$, và kích thước mẫu và minibatch.
1. Chứng minh rằng ước tính nhanh và đơn giản $L_\textrm{v}^\textrm{q}$ là không thiên lệch. Tức là, chứng minh rằng $E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$. Tại sao bạn vẫn muốn sử dụng $L_\textrm{v}$ thay thế?
1. Cho một hàm mất mát phân loại đa lớp, ký hiệu $l(y,y')$ là hình phạt khi ước tính $y'$ khi ta thấy $y$ và cho một xác suất $p(y \mid x)$, đặt ra quy tắc để chọn $y'$ tối ưu. Gợi ý: biểu diễn kỳ vọng mất mát, dùng $l$ và $p(y \mid x)$.


[Discussions](https://discuss.d2l.ai/t/6809)
