# Tiền xử lý dữ liệu
<a id="sec_pandas"></a>

Cho đến nay, chúng ta đã làm việc với dữ liệu tổng hợp
đến dưới dạng tensor sẵn sàng sử dụng.
Tuy nhiên, để áp dụng deep learning trong thực tế
chúng ta phải trích xuất dữ liệu lộn xộn
được lưu trữ ở các định dạng tùy ý,
và tiền xử lý nó để phù hợp với nhu cầu của chúng ta.
May mắn thay, thư viện *pandas* [library](https://pandas.pydata.org/)
có thể làm phần lớn công việc nặng nhọc.
Phần này, dù không thay thế được
một hướng dẫn *pandas* [tutorial](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) đầy đủ,
sẽ cung cấp cho bạn một khóa học nhanh
về một số thao tác phổ biến nhất.

## Đọc tập dữ liệu

Tệp giá trị phân tách bằng dấu phẩy (CSV) rất phổ biến
để lưu trữ dữ liệu dạng bảng (giống bảng tính).
Trong đó, mỗi dòng tương ứng với một bản ghi
và bao gồm một số trường (phân tách bằng dấu phẩy), ví dụ:
"Albert Einstein,14 tháng 3 1879,Ulm,Trường bách khoa liên bang,lĩnh vực vật lý hấp dẫn".
Để minh họa cách tải tệp CSV với `pandas`,
chúng ta (**tạo một tệp CSV bên dưới**) `../data/house_tiny.csv`.
Tệp này đại diện cho một tập dữ liệu nhà ở,
trong đó mỗi hàng tương ứng với một ngôi nhà riêng biệt
và các cột tương ứng với số phòng (`NumRooms`),
loại mái (`RoofType`) và giá (`Price`).

```python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

Bây giờ hãy nhập `pandas` và tải tập dữ liệu với `read_csv`.

```python
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Chuẩn bị dữ liệu

Trong học có giám sát, chúng ta huấn luyện mô hình
để dự đoán một giá trị *mục tiêu* được chỉ định,
cho một tập hợp giá trị *đầu vào* nào đó.
Bước đầu tiên trong xử lý tập dữ liệu
là tách các cột tương ứng
với giá trị đầu vào và giá trị mục tiêu.
Chúng ta có thể chọn cột theo tên hoặc
thông qua lập chỉ mục dựa trên vị trí nguyên (`iloc`).

Bạn có thể nhận thấy rằng `pandas` đã thay thế
tất cả các mục CSV có giá trị `NA`
bằng giá trị `NaN` đặc biệt (*not a number* - không phải số).
Điều này cũng có thể xảy ra khi một mục trống,
ví dụ: "3,,,270000".
Chúng được gọi là *giá trị bị thiếu*
và chúng là "bọ giường" của khoa học dữ liệu,
một mối nguy hiểm dai dẳng mà bạn sẽ đối mặt
trong suốt sự nghiệp của mình.
Tùy thuộc vào ngữ cảnh,
các giá trị bị thiếu có thể được xử lý
thông qua *điền vào* hoặc *xóa*.
Điền vào thay thế các giá trị bị thiếu
bằng ước tính của giá trị của chúng
trong khi xóa đơn giản bỏ qua
các hàng hoặc cột
chứa giá trị bị thiếu.

Dưới đây là một số phương pháp điền vào phổ biến.
[**Đối với các trường đầu vào phân loại,
chúng ta có thể coi `NaN` như một danh mục.**]
Vì cột `RoofType` nhận các giá trị `Slate` và `NaN`,
`pandas` có thể chuyển đổi cột này
thành hai cột `RoofType_Slate` và `RoofType_nan`.
Một hàng có loại mái là `Slate` sẽ đặt giá trị
của `RoofType_Slate` và `RoofType_nan` lần lượt thành 1 và 0.
Điều ngược lại đúng với hàng có giá trị `RoofType` bị thiếu.

```python
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

Đối với các giá trị số bị thiếu,
một phương pháp phổ biến là
[**thay thế các mục `NaN` bằng
giá trị trung bình của cột tương ứng**].

```python
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## Chuyển đổi sang định dạng Tensor

Bây giờ [**tất cả các mục trong `inputs` và `targets` đều là số,
chúng ta có thể tải chúng vào tensor**] (nhớ lại [sec_ndarray](#sec_ndarray)).


```python
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```


## Thảo luận

Bây giờ bạn đã biết cách phân tách các cột dữ liệu,
điền vào các biến bị thiếu,
và tải dữ liệu `pandas` vào tensor.
Trong [sec_kaggle_house](#sec_kaggle_house), bạn sẽ
học thêm một số kỹ năng xử lý dữ liệu.
Mặc dù khóa học nhanh này giữ mọi thứ đơn giản,
xử lý dữ liệu có thể trở nên phức tạp.
Ví dụ, thay vì đến trong một tệp CSV duy nhất,
tập dữ liệu của chúng ta có thể được trải rộng trên nhiều tệp
được trích xuất từ cơ sở dữ liệu quan hệ.
Chẳng hạn, trong một ứng dụng thương mại điện tử,
địa chỉ khách hàng có thể nằm trong một bảng
và dữ liệu mua hàng trong bảng khác.
Hơn nữa, người thực hành phải đối mặt với vô số kiểu dữ liệu
ngoài phân loại và số, ví dụ:
chuỗi văn bản, hình ảnh,
dữ liệu âm thanh và đám mây điểm.
Thường thì các công cụ nâng cao và thuật toán hiệu quả
được yêu cầu để ngăn xử lý dữ liệu trở thành
nút thắt cổ chai lớn nhất trong pipeline machine learning.
Những vấn đề này sẽ phát sinh khi chúng ta đến
thị giác máy tính và xử lý ngôn ngữ tự nhiên.
Cuối cùng, chúng ta phải chú ý đến chất lượng dữ liệu.
Các tập dữ liệu thực tế thường bị ảnh hưởng bởi
các ngoại lệ, các phép đo lỗi từ cảm biến và lỗi ghi,
những thứ phải được giải quyết trước
khi đưa dữ liệu vào bất kỳ mô hình nào.
Các công cụ trực quan hóa dữ liệu như [seaborn](https://seaborn.pydata.org/),
[Bokeh](https://docs.bokeh.org/) hoặc [matplotlib](https://matplotlib.org/)
có thể giúp bạn kiểm tra dữ liệu thủ công
và phát triển trực giác về
loại vấn đề bạn có thể cần giải quyết.


## Bài tập

1. Thử tải các tập dữ liệu, ví dụ: Abalone từ [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets) và kiểm tra các thuộc tính của chúng. Bao nhiêu phần trong số chúng có giá trị bị thiếu? Bao nhiêu phần của các biến là số, phân loại hay văn bản?
1. Thử lập chỉ mục và chọn các cột dữ liệu theo tên thay vì theo số cột. Tài liệu pandas về [indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) có thêm chi tiết về cách thực hiện điều này.
1. Bạn nghĩ bạn có thể tải một tập dữ liệu lớn đến mức nào theo cách này? Những hạn chế có thể là gì? Gợi ý: xem xét thời gian đọc dữ liệu, biểu diễn, xử lý và dấu vết bộ nhớ. Thử nghiệm trên laptop của bạn. Điều gì xảy ra nếu bạn thử trên server?
1. Làm thế nào bạn sẽ xử lý dữ liệu có số lượng danh mục rất lớn? Điều gì nếu các nhãn danh mục đều là duy nhất? Bạn có nên bao gồm cái sau không?
1. Bạn có thể nghĩ đến những lựa chọn thay thế nào cho pandas? Còn [tải tensor NumPy từ tệp](https://numpy.org/doc/stable/reference/generated/numpy.load.html) thì sao? Hãy xem [Pillow](https://python-pillow.org/), Thư viện hình ảnh Python.


[Thảo luận](https://discuss.d2l.ai/t/29)
